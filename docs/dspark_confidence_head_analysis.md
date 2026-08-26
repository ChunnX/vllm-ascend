# DSpark Confidence Head 与自适应验证：现状调研与适配设计

> 目标模型：Qwen3.6-27B DSpark（mamba/GDN 混合），MRV2
> 基线：`v0.26.0rc_dspark_dev`（vllm-ascend `releases/v0.26.0rc` + 11 commit，配套 vLLM 0.26.0）
> 调研日期：2026-08-26

## 摘要

上游 vLLM 已经在主线实现了完整的 confidence head + 自适应验证子系统（`7f7a32cfec`，
PR #47808，2026-08-12 合入），**并且是做在 MRV2 上的**。它比我们此前的设计更完整，
解决了图捕获这个最棘手的问题。

但有两个硬约束：

1. **该 PR 只在 vLLM main，未进任何 release tag。** 我们配套的是 vLLM 0.26.0，
   相关地基一个都没有。
2. **SSM backend 被明确排除，`gdn_attn` 在排除名单里。** Qwen3.6-27B 的 GDN 层
   正是这个 backend，按上游当前逻辑会在启动期被拒。

结论：**不建议现在整体跟进**。建议先做"能力对齐"——把 GDN spec 路径设备化、
让 Ascend attention backend 容忍 CPU/设备 query length 不一致。这两件事无论
后续走哪条路都要做，且与我们已有的"消除 draft 热路径同步"工作同源。

---

## 一、Confidence Head 是什么

一个很小的线性层：

```
输入：concat(draft backbone 的 hidden_state,
             上一个 draft token 的 markov embedding)
      维度 = hidden_size + markov_rank
输出：1 个标量 → sigmoid → 该位置 draft token 被接受的概率
```

Qwen3 版本与 DeepSeek-V4 版本有三处实质差异，不可混用：`bias=True`
（released `dspark_qwen3_*_block7` ckpt 带 `proj.bias`）、`params_dtype=float32`
且前向 `x.float()`、外层套 `sigmoid`。

**head 本身只是入口。真正的特性是自适应验证（Adaptive Verification）**：
不再固定投机 K 个 token，而是按置信度决定每个请求这一步实际验证几个。

上游的完整链路：

```
confidence head → 每位置接受概率 [B, D]
    → cumprod → survival（前缀累积存活概率）[B, D]
    → 全局预算（由启动时 profile 的步耗时代价模型决定）
    → 跨请求全局 topk 分配 → 每请求验证长度 [B]
```

关键设计：**槽位跨请求竞争**。一个高置信请求的第 5 个位置可以排在一个低置信
请求的第 1 个位置之前——所以某些请求保留完整 block，另一些可能只验证一两个 token。

### 我们的 checkpoint

已确认支持：

```
enable_confidence_head: True
confidence_head_with_markov: True
markov_rank: 256
```

---

## 二、三方现状对照

| | vLLM main | vllm-ascend main | 我们的分支（vLLM 0.26.0） |
|---|---|---|---|
| confidence head | ✅ 完整实现 | ⚠️ 仅 0.27.1 的 shim | ❌ 权重被 skip |
| 自适应验证 | ✅ MRV2 | ⚠️ 仅 MRV1，算法不同 | ❌ 无 |
| GDN/SSM 支持 | ❌ 明确排除 | — | — |

### vLLM main（PR #47808）

落地范围 21 个文件、约 1200 行，全部在 vLLM core：

```
vllm/v1/worker/gpu/spec_decode/adaptive_verification.py   +477  (新增)
vllm/v1/worker/gpu/model_runner.py                        +150
vllm/v1/worker/gpu/async_utils.py                          +94
vllm/v1/worker/gpu/input_batch.py                          +73
vllm/v1/attention/backend.py                               +45   ← 基类
vllm/v1/worker/gpu/cudagraph_utils.py                      +45
vllm/v1/worker/gpu/spec_decode/dspark/speculator.py        +41
vllm/v1/attention/selector.py                              +15
vllm/model_executor/models/qwen3_dspark.py                 +62
```

配套文档：`docs/features/speculative_decoding/adaptive_verification.md`。

**已核实**：`git tag --contains 7f7a32cfec` 为空——未进任何 release，
最新 tag 是 `v0.26.1rc0`。

### vllm-ascend main

`adaptive_verification` **零引用**，尚未开始接。自己那套 MRV1 的
`DynamicSpecScheduler`（`vllm_ascend/spec_decode/utils.py`）还在，且与
上游算法不同：固定阈值 `budget_threshold=0.3`，每 16 步一次 `.item()`，
而非代价模型。

**已核实**：最近 50 个 commit 里 `qwen3_dspark.py` / `spec_decode/utils.py` /
`ascend_config.py` 各 0 次改动；`vllm_ascend/worker/v2/` 下无任何 confidence
或 dynamic_spec 引用。

### 我们的分支

vLLM 0.26.1rc0 的 `Qwen3DSparkForCausalLM` 明确跳过权重：

```python
# confidence_head is not wired into inference yet; skip its weights.
skip_substrs = ["mask_embedding", "confidence_head"]
```

vllm-ascend main 的 shim 门控是 `if vllm_version_is("0.27.1") and ...`，
在 0.26.0 上为 False，走 `else` 分支（假设 vLLM 主线自己管理），而 0.26
也不管。**两条路在 0.26.0 上都是死的**，照搬上游文件不会生效。

---

## 三、`is_ssm()` 与 SSM 排除

### 是什么

**State Space Model，状态空间模型。不是稀疏 attention。**

指 Mamba、GDN（Gated Delta Net）、linear attention、RWKV 这类**循环模型**。
vLLM 中声明 `is_ssm() -> True` 的 backend：`mamba1_attn`、`mamba2_attn`、
`linear_attn`、`short_conv_attn`、**`gdn_attn`**。

（稀疏 attention 是另一套：`sparse_mla`、`sparse_swa` 等，`is_ssm()` 为 False，
不受此排除影响。）

`is_ssm()` 全仓只有两个消费点：`platforms/interface.py` 的
`_find_non_ssm_backend`（混合模型选 block_size 时跳过状态层），以及 #47808
新增的能力开关默认值。

### 为什么排除

区别在于**有没有跨 token 的顺序状态**：

- **Attention**：每个 query 独立查 KV cache。少算几个 query，剩下的结果不变。
  一个 step 内 token 之间无前后依赖。
- **SSM**：循环状态沿序列前进，token *i* 的输出依赖 token *i-1* 更新后的状态。
  一个请求的状态从哪开始、推进几步、写回哪个 slot，全按**每请求 token 边界**规划。

自适应验证的做法是：CPU 上按请求**均匀**分配 draft 预算（占位），真实切分在
**设备上**决定。上游注释明确了哪些不变量成立：

> On the CPU the draft budget is evenly distributed across requests, so the
> **total** draft budget, the **decode/prefill split point** and the **CPU
> prefill query lengths** all stay correct.

即：总量对、prefill 对、split 点对，**只有 spec decode 请求的每请求 query
长度对不上**。

对 attention 无所谓；对 SSM 则会让状态推进走错 token 区间、写错 state slot，
**而且不报错**，只是结果悄悄变错。

### 重要：这是可覆写的默认值，不是禁令

```python
@classmethod
def supports_device_cpu_query_lens_mismatch(cls) -> bool:
    """...SSM backends opt out: their recurrent-state planning is built from
    the CPU per-request boundaries, which the trimmed batch no longer matches."""
    return not cls.is_ssm()
```

任何 backend 都可以覆写它返回 `True`。上游的语义是"SSM 默认假定从 CPU 边界
规划，要用请自证"，而非架构上不可能。

---

## 四、GDN 审计结果

对照上述不变量，逐条检查 `vllm_ascend/ops/gdn_attn_builder.py`
（本分支版本，含我们的 `_GDNSharedBatchPlan` 改造）：

| 量 | 来源 | 裁剪后 |
|---|---|---|
| `spec_query_start_loc` | `query_start_loc[:n+1]`（**设备**，L687） | ✅ 本来就对 |
| `spec_token_indx` | `repeat_interleave(mask, query_lens)`，`query_lens` 设备派生 | ✅ |
| `num_spec_decode_tokens` | `query_lens_cpu.sum()` 减法（**总量**，L663） | ✅ 总量守恒 |
| `spec_state_indices_tensor` | `block_table[mask, :num_spec+1]`，取最大宽度 | ✅ 上界 |
| `num_prefills/decodes/...` | CPU 标量，用于分支与张量尺寸 | ✅ 请求数与总量不变 |
| 非 spec chunked prefill | 大量 `.tolist()` host tuple | ✅ **不受影响** |
| `_fold_spec_sized_prefill_chunks_into_spec` | `query_lens_cpu == 1` 分类（L69-75） | ⚠️ **会误判** |

**关键发现**：那一大堆看着最棘手的 host tuple（`cu_seqlens_host`、
`chunk_indices_chunk64_host`，喂给 AscendC kernel、无设备等价物）全部位于
`_build_non_spec_chunked_prefill_metadata` —— **non-spec 路径，裁剪管不着**。
反向 grep spec 路径的 `host` 依赖，除一句注释外为空。

**因此：GDN spec 路径已经基本是设备侧的，上游那个 blanket opt-out 对 GDN
偏保守。** （mamba1/mamba2/short_conv 可能确实更糟，未检查。）

---

## 五、图捕获：上游的解法

这是此前我们最担心的问题。vLLM 0.26 MRV2 的判定：

```python
# vllm/v1/worker/gpu/cudagraph_utils.py
def get_uniform_token_count(num_reqs, num_tokens, max_query_len):
    """A batch is uniform if all requests have the same number of tokens."""
    if (max_query_len == num_tokens // num_reqs) and (num_tokens == max_query_len * num_reqs):
        return max_query_len
    return None
```

per-request 变长 K → 返回 `None` → FULL 模式不匹配 → **退回 eager**。

上游没有退让为"统一 K"，而是引入 **varlen decode graph**：

```
# Varlen decode graphs take any mix of 1..decode_query_len tokens per
# request, worst case 1 token per request (or max_num_reqs)
```

不按统一 query 宽度捕获多套图，而是按**总 token 数**捕获，
`num_reqs = min(num_tokens, max_num_reqs)`，`max_query_len` 作为匹配上界，
每请求的切分**在设备侧决定**。既无 shape 爆炸，也无 CPU 同步，且保住请求级粒度。

文档中 full cudagraph 是**必需**的（`--enforce-eager` 启动即拒），因为预算
代价模型要靠捕获图 profile 步耗时。

> 备注：我们此前"降级为每步统一 K 以保住 full graph"的方案被此实现淘汰。

### 上游的其他限制

- 不支持 LoRA（per-token LoRA 映射建自 CPU 边界）
- 不支持 PP（代价曲线与置信度只存在于最后一个 PP rank）
- 不支持 output logprobs（标注为待修）

---

## 六、自建路径与难点

### 最根本的难点

#47808 的改动在 **vLLM core**，不在 vllm-ascend。而 vllm-ascend 是**插件**，
靠 `patch/` 猴补丁与继承覆写改变 vLLM 行为。这次要动的是 `AttentionBackend`
基类、`InputBatch`、`ModelRunner`、`CudaGraphManager`——**都是被继承和被广泛
引用的地基**。

0.26.x 的地基情况（已核实）：

| 符号 | v0.26.1rc0 |
|---|---|
| `adaptive_verification.py` | 不存在 |
| `varlen_decode` | 0 处 |
| `supports_device_cpu_query_lens_mismatch` | 0 处 |

### 三条路

**A. 升级到 vLLM main**
整条 dev 分支与 vllm-ascend `releases/v0.26.0rc` 都建在 0.26.0 上。
这是版本策略问题，需团队决策，非技术选择。

**B. 将 #47808 backport 进打过补丁的 vLLM**
已量化漂移：`model_runner.py` 从 v0.26.1rc0 到 main 为 +473/-141，其中 150 行
即 #47808 本身，无关漂移约 320 行。技术可行，**但从此维护一个私有 vLLM 分支**，
每次跟版都要重做。

**C. 在 vllm-ascend 自己的 MRV2 层重新实现**
绕开 vLLM 基类，只用继承与覆写能触及的地方。省去维护私有 vLLM 的代价，
但能力受限于覆写点。

### vllm-ascend 特有难点（A/B/C 都躲不掉）

**1. varlen decode 图 vs ACLGraph —— 最大的一块，不确定性最高**

vllm-ascend 有自己一整套：`ModelAclGraphManager`、`set_graph_params(capture_sizes)`、
`collect_sorted_captured_token_sizes`，以及 attention backend 内的
`graph_task_update_begin/end`——**后者用 host 侧参数更新图内算子 tiling**。

varlen decode 意味着 replay 时每请求切分由设备决定。这套 host 参数更新路径
能否容忍，**无法静态判断，必须上机验证**。

**2. attention backend 需真正容忍 CPU/设备 query length 不一致**

我们的 fia_sink 工作在此是顺风——该算子本就接受设备侧 seq_lens、tiling 在
AICPU 完成，正是所需能力。其他 backend 需逐个审计。

**3. GDN metadata —— 反而最小**

见第四节。真正要修的是 `_fold_spec_sized_prefill_chunks_into_spec` 的 CPU 分类。

**4. cost model profiling**

上游预算来自启动时 profile 的各 shape 步耗时，非固定阈值。Ascend 需重做一套。
不难，但是新工作。

**5. 验证成本才是大头**

失败模式是**静默算错**——形状对得上、不报错、精度悄悄掉。真实成本大头在
验证：贪心无损等价、accepted-count 序列一致性、各并发下的 TPOT。
此类工作编辑量与验证量常为 1:5。

---

## 七、建议的推进顺序

**先做"能力对齐"，不做"功能对齐"。**

### 第一步（建议现在做）

1. **GDN spec 路径设备化**：修 `_fold_spec_sized_prefill_chunks_into_spec` 的
   CPU 分类，清掉残留的每请求 CPU 派生，覆写
   `supports_device_cpu_query_lens_mismatch()` 返回 `True`，补单测。
2. **Ascend attention backend 容忍 CPU/设备 query length 不一致**：逐个审计，
   fia_sink 路径优先（已具备设备侧 seq_lens 能力）。

理由：

- 这两件事**无论走 A/B/C 哪条路都要做**，不会白费
- 与我们已进行数月的"消除 draft 热路径同步"工作同源，团队熟悉
- 完成后手握硬证据，可就上游 `is_ssm()` opt-out 提 issue——作者注释写得很具体，
  说明其清楚缺什么，大概率愿意讨论

### 第二步（模型侧，可并行）

把 confidence head 建起来、权重加载对齐 0.26.0（**重写那个
`vllm_version_is("0.27.1")` 门控**）、与我们的 vocab_mapping `load_weights`
组合好。

注意上游存在静默降级：

```python
if not confidence_weights:
    self.enable_confidence_head = False
    return
```

无权重就悄悄关闭、不报错。建议改为显式告警，避免出现"跑得好好的但
DSD 从未生效"的版本。

这部分无论上层用哪套调度都是必需的，且能先验证 checkpoint 权重能否加载、
confidence 分数分布是否合理。

### 暂不建议做

**varlen decode 图**。不确定性最大、最可能推倒重来，且上游已有实现——
等版本跟上去白拿，比自己造一个不一样的划算。

---

## 八、待核实项

以下判断来自**读代码，未运行验证**：

- **AscendC 的 GDN kernel 本身是否从 host 参数读每请求边界。**
  `_build_non_spec_chunked_prefill_metadata` 中的 host tuple 说明**至少
  non-spec kernel 是这么做的**；spec kernel 未找到对应证据，但"未找到"
  不等于"没有"。需查 kernel 源码或上机验证。
- **SSM 排除对混合模型的具体表现**：是整模型启动期拒绝，还是仅 GDN 层被拒。
  文档表述为"excluded by the attention selector, and rejected at startup for
  models that hard-wire their backend"，GDN 层无备选 backend 即属 hard-wire，
  但未实际触发验证。
- **第六节难点 1 的工作量区间（2~6 周）** 主要来自此项不确定性。

---

## 附：关键引用

| 内容 | 位置 |
|---|---|
| PR #47808 | vLLM `7f7a32cfec`，2026-08-12 |
| 上游设计文档 | `docs/features/speculative_decoding/adaptive_verification.md` |
| 能力开关 | `vllm/v1/attention/backend.py` `supports_device_cpu_query_lens_mismatch()` |
| SSM 标记 | `vllm/v1/attention/backends/gdn_attn.py` `is_ssm()` |
| 均匀性判定 | `vllm/v1/worker/gpu/cudagraph_utils.py` `get_uniform_token_count()` |
| 启动期校验 | `vllm/config/vllm.py`（LoRA / eager / PP） |
| 我方 GDN builder | `vllm_ascend/ops/gdn_attn_builder.py` |
| vllm-ascend MRV1 DSD | `vllm_ascend/spec_decode/utils.py` `DynamicSpecScheduler` |
