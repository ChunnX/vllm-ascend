# 交付件 3：`fia_sink impl`（c357e30bb）代码评审

> 评审对象：commit `c357e30bb` "fia_sink impl"
> 参考基线：`omni-ops` 算子源码、`omniinfer`（`feature/vllm_0.25.1`）的 sink 使能实现、
> 本仓交付件 1/2 两份文档
> 结论：**方向正确，但存在 2 处会崩的缺陷，不建议直接上机端到端**

---

## 0. 结论摘要

| 级别 | 问题 | 后果 |
|---|---|---|
| P0-1 | builder 与 impl 的 sink 判定条件不等价，不一致时 fall through 到已被置 `None` 的 host 路径 | `TypeError: 'NoneType' object is not subscriptable`，崩溃 |
| P0-2 | `update_graph_params` 的过滤器用 `hasattr` 判断，拦不住 `seq_lens_list=None` 的 sink 层 | 崩溃，或与普通层的捕获参数错位配对（静默算错） |
| P1-1 | 入图路径未实现交付件 2 第 3.4 节要求的 `full_graph_fia_sink` / `update_graph_params` 分支 | 与文档不符（但与 omniinfer 一致，见 §3） |
| P1-2 | metadata 每层重算、每次新分配输出，参考实现是每步一次 + 地址稳定 buffer | AICPU 派发 ×L，可能吃掉全部收益 |
| P2-1 | 未用 `.out` / `_get_max_workspace` 变体 | 多一次 copy，偏离文档 4.3 与 omniinfer `OP_FIA_SINK` |
| P2-2 | `num_tokens` 用 `num_actual_tokens`，与 `sum(actual_seq_qlen)` 在图 padding 下可能不等 | TND 约束校验失败 |
| P2-3 | 文档面向 mrv2，但改动落在两 runner 共用的 `attention_v1.py` | mrv1 未经设计即被启用 |

值得肯定的是：env 开关默认关闭、双保险判定的思路正确、编译文档（交付件 1）可直接执行。

---

## 1. P0-1：双保险的两个判定不等价，不一致时必崩

`build()` 里做了**不可逆**的决定（`attention_v1.py`）：

```python
if use_fia_sink:          # builder 侧
    query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
    actual_seq_lengths_q = None
    seq_lens_list = None
```

builder 侧条件：

```python
_DSPARK_FIA_SINK_ENABLED
and self.speculative_config is not None
and self.speculative_config.parallel_drafting
and not common_attn_metadata.causal
and getattr(self.kv_cache_spec, "head_size", None) == 128
```

impl 侧 `_use_fia_sink` **额外**要求四个条件：

```python
and self.head_size == 128
and self.num_heads != self.num_kv_heads      # ← 强制 GQA，MHA 会被拒
and self.num_heads % self.num_kv_heads == 0
and self.sliding_window is None
and self.sinks is None
```

**builder 已经承诺"不提供 host list"，impl 却可以拒绝走 sink**。此时代码 fall through 到普通
FIA 路径，而那里有 12 个消费点在读这两个已被置 `None` 的字段：

| 位置 | 表达式 | 后果 |
|---|---|---|
| `full_graph_fia` | `attn_metadata.actual_seq_lengths_q[-1]` | TypeError |
| `full_graph_fia_v2` | 同上 | TypeError |
| `forward_fused_infer_attention` | 同上 | TypeError |
| `_get_fia_params`（3 处） | `actual_seq_lengths_kv = attn_metadata.seq_lens_list` | None 传给算子 |
| decode 分支 | `len(attn_metadata.seq_lens_list)` | TypeError |

**最现实的触发场景**：草稿模型是 MHA（`num_heads == num_kv_heads`）。builder 只查
`head_size == 128` 就置位，impl 因 `num_heads != num_kv_heads` 拒绝 → 崩。

两个判定天生无法等价：builder 是 per-kv-cache-group，拿不到 per-layer 的 `num_heads`。

**建议修复**（二选一）：

1. **最小改动**：impl 侧发现 `attn_metadata.use_fia_sink is True` 但自身条件不满足时，
   直接 `raise RuntimeError`，附明确原因，而不是静默 fall through 到必然 NoneType 的路径。
   把"难以定位的 NoneType"变成"一眼可读的配置错误"。
2. **更彻底**：把 per-layer 能力检查提前到模型初始化时做一次，结论回传给 builder，
   使两侧判定由构造保证一致。

无论选哪个，**上机前必须先确认草稿模型的 GQA 配置**：

```bash
python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<draft模型路径>'); \
print('heads=', c.num_attention_heads, 'kv_heads=', c.num_key_value_heads, \
'head_dim=', c.hidden_size // c.num_attention_heads)"
```

`num_attention_heads == num_key_value_heads` 时，这个开关一打开就崩。

---

## 2. P0-2：`update_graph_params` 的过滤器拦不住 sink 层

`attention_v1.py` 中筛选"有 FIA 图参数的层"用的是：

```python
attn_keys = [k for k in attn_metadata if hasattr(attn_metadata[k], "seq_lens_list")]
```

`hasattr` 对**值为 `None` 的属性仍返回 `True`**，因此 sink 层不会被过滤掉。随后：

```python
seq_lens = attn_metadata[draft_step][key].seq_lens_list   # None
...
torch_npu.npu_fused_infer_attention_score_v2.out(..., actual_seq_kvlen=seq_lens, ...)
```

比崩溃更糟的情况：同一张图里若同时存在 sink 层与普通层，
`zip(attn_keys, captured_attn_params, handles, events)` 会把 **sink 层的 key 与普通层的捕获参数
错位配对**——不报错，直接算错。

**建议修复**：过滤条件改为 `getattr(attn_metadata[k], "seq_lens_list", None) is not None`，
或显式排除 `use_fia_sink` 为真的层。

---

## 3. P1-1：入图路径未实现，但需要为 commit 正名

交付件 2 第 3.4 节明确要求新增 `full_graph_fia_sink` 与 `update_graph_params` 的 sink 分支，
commit 一条都没做，且把 sink 判定放在 `_EXTRA_CTX.capturing` 检查**之前**，捕获期直接内联执行。

**但对照 omniinfer 参考实现，commit 的做法反而是对的**
（`omni/attention/backends/attention.py`）：

```python
if forward_context.capturing and not use_aicpu_fa_tiling:   # 注意 not
    capture_graph_task(op_desc=OP_FIA_SINK, ...)
else:
    if use_aicpu_fa_tiling:
        ...  # 算 meta_data
    attn_output = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]
```

即：**启用 AICPU tiling 时，omniinfer 在捕获期同样走内联执行**，不走 `capture_graph_task`。
原因是 AICPU metadata 算子随图一起捕获、replay 时读取地址稳定的 device buffer 中的新值，
本身就能得到正确结果，不需要 `graph_task_group` 的可更新任务机制。

**因此本条从 P0 降为 P1**，需要做的是：

- 修改 `_forward_fia_sink` 的 docstring，说明"与 omniinfer 一致：AICPU tiling 模式下捕获期内联执行，
  依赖 device buffer 地址稳定性而非 graph_task_update"；
- 更新交付件 2 第 3.4 节，说明该节的方案偏保守，实际以 omniinfer 为准。

---

## 4. P1-2：metadata 每层重算，参考实现是每步一次

omniinfer 用 `CrossLayerSharedOp`（`omni/attention/backends/utils.py:335`）包装 metadata 算子，
其 docstring 原文：

> The first layer in a step that calls with `recompute=True` runs the underlying op and copies
> the result into the buffer for its caller; later layers in the same step call with
> `recompute=False` and read the buffer directly. **Buffer addresses are stable across steps so
> aclgraph / cudagraph captures them safely.**

即两个关键性质：

1. **每步只算一次**，其余层读同一 buffer；
2. **buffer 地址跨步稳定**，这是 aclgraph 捕获安全的前提。

commit 的 `_forward_fia_sink` 两者都没有：每层都调
`torch.ops.custom._npu_fused_infer_attention_sink_metadata(...)`，且每次**新分配**输出张量。

后果：

- L 层草稿模型 = 每步 L 次 AICPU 派发（参考实现为 1 次）。AICPU 派发开销不可忽略，
  很可能是"同步消失了但端到端没变快"的直接原因；
- 无地址稳定性保证，aclgraph 捕获的安全性依赖 graph memory pool 的分配行为，脆弱。

**建议修复**：仿 `CrossLayerSharedOp` 实现精简版——impl 持有
`self._fia_sink_meta = torch.empty((1024,), dtype=torch.int32, device="npu")`，
由每步第一个草稿层写入、其余层直接读。

---

## 5. P2 项

### 5.1 未用 `.out` / `_get_max_workspace` 变体

交付件 2 第 4.3 节要求图捕获使用 `.out` + `_npu_fused_infer_attention_sink_get_max_workspace`；
omniinfer 的 `OP_FIA_SINK` 也是 `op_out_fn=...sink.out` + `workspace_fn=..._get_max_workspace`。
commit 用的是分配式变体，并额外做了一次 `output[:num_tokens] = attn_output[:num_tokens]` 拷贝。

### 5.2 `num_tokens` 语义

commit 取 `num_tokens = attn_metadata.num_actual_tokens`，而算子的 TND 约束是
`sum(actual_seq_qlen) == query.shape[0]`，其中 `actual_seq_qlen = query_start_loc[1:]`。
sink 分支跳过了原有的 padding 补齐逻辑（dummy 请求的 `seq_lens_list` 补 1、`block_table` 补零行），
在 FULL 图 padding 场景下两者可能不等，需要专门构造用例验证。

### 5.3 mrv1 / mrv2 范围

交付件 2 通篇针对 **mrv2** 撰写（引用 `worker/v2/spec_decode/dflash/speculator.py`、
`worker/v2/attn_utils.py`、`DFlashAclGraphManager`），改动文件清单也列的是 `worker/v2/...`。
但实际改动落在 **两个 runner 共用的** `attention_v1.py`，判定条件
（`parallel_drafting + 非因果 + head_size==128`）在 **mrv1 上同样会触发**。

mrv1 有自己的 padding 逻辑、`seq_lens_group` 持久 buffer、
`_adjust_parallel_draft_seq_lens_for_graph` 等机制，均未在本次设计中考虑。
若近期主力在 mrv1，需要补充 mrv1 场景的设计与验证；或在判定条件中显式限定 runner。

---

## 6. 上机验证建议

**分两阶段，不要一上来跑端到端。**

### 阶段 1：先验证算子本身（不涉及 vllm-ascend）

按交付件 1 编译安装（路径按实际替换）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd <omni-ops>/inference/ascendc
bash build.sh \
  -n 'ai_infra_fused_infer_attention_sink;ai_infra_fused_infer_attention_sink_metadata' \
  -c ascend910b --disable-check-compatible      # A3 用 ascend910_93

cd output && chmod +x CANN-omni_custom_ops--linux.*.run
./CANN-omni_custom_ops--linux.*.run --quiet \
  --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/omni_custom_transformer/bin/set_env.bash

cd <omni-ops>/inference/ascendc/torch_ops_extension
bash build_and_install.sh
```

两个算子必须一起编译（metadata 的 AICPU 依赖 sink 的公共代码）。

随后运行交付件 1 第 4 节的单算子脚本。**建议补充一项文档未覆盖的验证**：
用相同输入与 `torch_npu.npu_fused_infer_attention_score` 做**精度比对**
（非因果 / GQA / 分页 KV），确认数值等价。否则端到端出问题时无法区分是算子还是适配层。

### 阶段 2：修复 P0 后再跑端到端

1. 先执行 §1 的 GQA 配置检查；
2. 修复 P0-1、P0-2；
3. `VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=1`，**先 eager 后入图**；
4. 判据（沿用既有约定）：
   - greedy（temperature=0）下与无投机逐 token 一致；
   - 每步 accepted counts 序列与改动前**逐步相同**（本改动理论上是恒等变换，接受率不应变化）；
   - profile 确认 `seq_lens.tolist()` / `aten::to` / `_to_copy` 消失；
   - **端到端 TPOT / 吞吐**——这是唯一的性能判据，单个火焰图帧的长短不能作数。

若 TPOT 无改善，优先怀疑 §4 的 metadata 每层重算。
