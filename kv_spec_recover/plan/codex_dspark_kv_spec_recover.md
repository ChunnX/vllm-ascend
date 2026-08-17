# Qwen3.6 DSpark KV Metadata 构建耗时恢复方案（评审后修订版）

## 1. 文档目标与修订结论

本文针对 Qwen3.6 + DSpark 场景中 KV cache group 数量显著多于 MTP、进而放大 `build_attn_metadata` 耗时的问题，给出可实施、可验证、可回退的修改方案。

本版是在综合以下材料并复核当前源码后，对 commit `70f1d583` 中原方案的修订：

- `kv_spec_recover/dspark_kv_group_issue_analysis.md`
- `kv_spec_recover/dspark_kv_log_evidence.md`
- `kv_spec_recover/mtp_vs_dspark_log_evidence.md`
- `kv_spec_recover/dspark_kv_analyze.info`
- `kv_spec_recover/mtp_kv_analyze.info`
- 诊断打点 commit `5d69f096225b96aa3b95d4a7f4a9e4328b8972db`
- commit `0c187fce` 中的 DSV4 review
- 新增 profiling 评审材料

实际日志文件名以本仓库为准，不采用早期文档里的 `/opt/z00830407/*.log`。

修订后的结论是：

1. **接受优先级调整。** 第一阶段不改 KV 分组，优先消除 10 个 GDN group 之间重复的 batch-shape 计算。
2. **不接受“完整 metadata 浅拷贝后只替换少数索引”的实现。** GDN 中随 group 变化的不只有两类 state index，还包括 `prefill_state_indices` 和三个嵌套 causal-conv1d metadata 中的 `cache_indices`；FULL ACLGraph 下各 builder 还拥有自己的持久 buffer。
3. 第一阶段采用 **batch-scope 的不可变公共模板 + 每 group 完整重绑定 KV 地址字段**，group 数仍为 15，scheduler/worker/KV allocator 契约完全不变。
4. 第二阶段才考虑改分组；若实施，使用全局通用的 Pareto 规则：**在 padding 不超过现有 heuristic 的前提下最小化 group 数**。对 `[16, 5, 48]` 选择 `group_size=8`、9 groups，而不是原方案的 `group_size=16`、5 groups。
5. 删除原方案的 `DSPARK_KV_GROUP_MAX_PADDING_RATIO=0.20`。它既引入 DSpark gating 风险，又和“容量损失不超过 10%”验收标准不一致。

本文是修改计划，不表示代码、性能结果或 NPU 验证已经完成。

---

## 2. 为什么 DSpark 比 MTP 多很多 KV group

### 2.1 已确认的 spec 桶

MTP 的 KV spec 桶为：

```text
[17, 48]

17 = 16 个 base FullAttentionSpec + 1 个 MTP FullAttentionSpec
48 = 48 个 MambaSpec
```

MTP 层和 base attention 层的关键规格相同：

```text
num_kv_heads = 1
head_size = 256
head_size_v = 256
```

所以 MTP 层能与 base 层合并成 17 层的同类桶。现有 heuristic 选择 `group_size=17`：

```text
ceil(17 / 17) + ceil(48 / 17) = 1 + 3 = 4 groups
```

DSpark 的 KV spec 桶为：

```text
[16, 5, 48]

16 = base FullAttentionSpec
5  = DSpark draft FullAttentionSpec
48 = MambaSpec
```

DSpark draft attention 的关键规格是：

```text
num_kv_heads = 2
head_size = 128
head_size_v = 128
```

它与 base 的 `1 x 256` tensor 布局不同，不能伪装成同一种 `FullAttentionSpec`，因此形成独立的 5 层小桶。当前 `_get_kv_cache_groups_uniform_page_size()` 选择最小桶大小 5：

```text
ceil(16 / 5) + ceil(5 / 5) + ceil(48 / 5)
= 4 + 1 + 10
= 15 groups
```

所以 DSpark 多 group 的直接原因不是“总层数只多了几层”，而是 **5 层异构 draft spec 把统一 group width 从 17 拉低到 5**，48 个 Mamba 层随之从 3 组被切成 10 组。

### 2.2 通俗例子

可以把三类 KV 层看作三种不能混装的货物，仓库要求所有运输车使用相同数量的货位：

- MTP 有 17 箱 A 货和 48 箱 B 货，车宽取 17，需要 `1 + 3 = 4` 车；
- DSpark 有 16 箱 A、5 箱 C、48 箱 B。C 虽然总体积接近 A，但内部固定架不同，不能与 A 混装；
- 当前规则看到最小一堆只有 5 箱，就要求所有车都改成 5 个货位，于是需要 `4 + 1 + 10 = 15` 车。

每辆车在出发前都要填写一套“本批次有多少订单、哪些是预填充、哪些是推测解码”的公共表格，同时再填写“本车货物放在哪些仓位”的专属表格。10 辆 Mamba 车重复填写前一套公共表格，正是本次优先优化的部分；但每辆车的仓位表不能共用。

### 2.3 `build_attn_metadata` 的放大关系

`vllm_ascend/worker/v2/attn_utils.py::build_attn_metadata()` 按 `kv_cache_groups` 外层循环，每个 group 都有自己的：

- `block_tables[i]`
- `slot_mappings[i]`
- `AscendCommonAttentionMetadata`
- attention metadata builder

因此 DSpark 会执行 15 个 metadata 构建单元，其中 10 个是 GDN/Mamba group；MTP 只有 4 个 group。

新增 profiling 给出的单次观测是：

```text
_build_attention_metadata 总耗时 约 17 ms
  GDN builder build       约 14 ms（10 次，每次约 1.4 ms）
  其余                    约  3 ms
```

这个数据足以改变优化优先级，但目前仓库没有对应原始 profile，尚不能把 `17 ms -> 5 ms` 写成已验证结论。实施前必须补充 batch size、序列长度、prefill/decode 混合、是否稳态、eager/ACLGraph 模式、采样次数和 p50/p90/p99。

---

## 3. GDN 中哪些内容能共享，哪些绝不能共享

### 3.1 同一批次、兼容 GDN builder 间可复用的逻辑结果

当前 `AscendGDNAttentionMetadataBuilder.build()` 的下列计算主要由 batch 请求形状和 speculative decode 输入决定，与 group block table 无关：

- prefill、decode、spec-decode 的请求数和 token 数；
- `spec_sequence_masks` 及 CPU mask；
- spec/non-spec sequence index；
- `query_lens`；
- spec/non-spec token index；
- spec/non-spec query start location；
- 过滤后的 `num_accepted_tokens`；
- prefill chunk indices/offsets；
- context length、`has_initial_state` 的逻辑值；
- causal-conv1d 的 `nums_dict`、`batch_ptr`、`token_chunk_offset_ptr`。

这些结果适合提取为一次 `build_attn_metadata` 调用内部的公共模板。

### 3.2 每个 group 必须重新计算或重绑定的字段

下列字段由当前 group 的 `common_attn_metadata.block_table_tensor` 产生，不能直接复用首个 group 的值：

- `mamba_get_block_table_tensor(...)` 的结果；
- `spec_state_indices_tensor`；
- `non_spec_state_indices_tensor`；
- `prefill_state_indices`；
- `non_spec_conv1d_cache_indices`；
- `spec_decode_metadata.spec_causal_conv1d.cache_indices`；
- `non_spec_prefill_metadata.causal_conv1d.cache_indices`；
- `non_spec_decode_metadata.causal_conv1d.cache_indices`。

`slot_mapping` 虽然不直接进入当前 GDN builder 的主要分支，也必须继续保持每 group 的 `AscendCommonAttentionMetadata` 独立，不能改变上层契约。

### 3.3 FULL ACLGraph 下的额外约束

每个 GDN builder 实例拥有自己的持久 tensor buffer，例如：

- `spec_state_indices_tensor`
- `non_spec_state_indices_tensor`
- `spec_sequence_masks`
- spec/non-spec token index buffer
- spec/non-spec query start buffer
- `num_accepted_tokens`

FULL graph 路径会把当前批次数据复制到这些 builder-local buffer，并让返回 metadata 引用稳定地址。若让后续 group 直接引用第一个 builder 的 buffer，可能造成：

- 不同 group metadata 非预期 alias；
- graph capture/replay 地址所有权改变；
- 后续 builder 或下一 step 覆盖前一 group 的内容；
- 精度问题或 replay 时序问题。

所以公共模板应该表示 **不可变的逻辑批次结果**，而不是第一个 builder 返回的完整 `GDNAttentionMetadata`，更不能把第一个 builder 的持久 buffer 当作全局共享结果。

---

## 4. 第一阶段：batch-scope GDN 公共模板复用

### 4.1 目标与非目标

目标：

1. DSpark 仍保持 15 个 KV groups，所有 KV 分配和地址映射完全不变。
2. 10 个兼容 GDN builder 的公共 batch 计算只做一次。
3. `mamba_get_block_table_tensor()` 和全部 KV 地址派生字段仍按 group 执行 10 次。
4. 每个 group 仍返回独立、完整、类型不变的 metadata。
5. eager、prefill、decode、spec decode、FULL_DECODE_ONLY ACLGraph 均正确。

非目标：

- 不合并 KV groups；
- 不合并不同 group 的 block table 或 slot mapping；
- 不修改 scheduler、KV allocator 或 vLLM core；
- 不跨 engine step 保存 cache；
- 不新增环境变量；
- 不在热路径增加新的 `.item()` 或 CPU-NPU 同步。

### 4.2 修改文件

生产代码：

```text
vllm_ascend/ops/gdn_attn_builder.py
vllm_ascend/worker/v2/attn_utils.py
```

测试代码：

```text
tests/ut/ops/test_gdn_attn_builder.py
tests/ut/ops/test_gdn_layerwise_kv.py
```

如现有测试职责不适合，可新增：

```text
tests/ut/worker/v2/test_attn_utils_gdn_cache.py
```

### 4.3 数据结构设计

在 `gdn_attn_builder.py` 中新增私有 dataclass，名称可采用：

```python
@dataclass(frozen=True)
class _GDNBatchedMetadataTemplate:
    # 仅保存与 group block table 无关的逻辑结果。
    ...
```

模板字段必须按源码数据依赖逐项列出，禁止直接存一个完整 `GDNAttentionMetadata` 后依赖“排除字段列表”。建议拆为三类：

1. Python 标量：各类 request/token count；
2. 与 batch/spec 分流相关的 immutable tensor/view；
3. causal-conv1d 和 prefill chunk 的公共逻辑描述。

对于 FULL graph 需要稳定地址的字段，模板保存源逻辑值，由当前 builder 将其 materialize/copy 到自己的 persistent buffer。只有确认不会改变 graph 地址契约的 immutable tensor 才允许直接共享引用。

### 4.4 Builder API

推荐把当前 `build()` 内部逻辑拆成三个明确步骤：

```python
def _build_batch_template(
    self,
    common_attn_metadata,
    num_accepted_tokens,
    num_decode_draft_tokens_cpu,
) -> _GDNBatchedMetadataTemplate:
    ...

def _materialize_group_metadata(
    self,
    template,
    common_attn_metadata,
) -> GDNAttentionMetadata:
    ...

def build(..., batch_template=None) -> GDNAttentionMetadata:
    template = batch_template or self._build_batch_template(...)
    return self._materialize_group_metadata(template, common_attn_metadata)
```

若为了兼容上游 builder 接口不希望扩展公开 `build()` 参数，可在 Ascend builder 上增加私有方法，由 `attn_utils.py` 在确认 builder 类型后调用。无论采用哪种 API，都应保留无 cache 的完整构建路径，便于等价性测试和回退。

### 4.5 Batch-scope 生命周期

在 `vllm_ascend/worker/v2/attn_utils.py::build_attn_metadata()` 的 group 循环外创建局部 cache：

```python
gdn_batch_templates: dict[_GDNTemplateKey, _GDNBatchedMetadataTemplate] = {}
```

函数返回后 cache 自然销毁，不跨 step、不跨 request batch、不成为 mutable global state。这样无需用 `num_reqs` 或 tensor `id()` 解决跨 step 失效问题，也避免 Python 复用对象地址导致误命中。

这与同一函数中 DSA metadata 的 batch-builder scope cache 思路一致，但 GDN 不能照搬 DSA 的字段和 key。

### 4.6 兼容性 key 与 fallback

同一次 group 循环里仍可能出现配置不同的 GDN builder，cache key 至少要覆盖真正改变公共计算语义的 builder invariant：

- builder 类型；
- `num_spec` / 是否启用 spec decode；
- `use_full_cuda_graph`；
- GDN prefill backend；
- 会改变公共 metadata 结构的 `MambaSpec` 属性；
- extra kwargs 的存在性、shape/dtype/device 和当前批次切片一致性。

由于 batch 输入本身来自同一次 `build_attn_metadata()`，不应把 block table 放入 key；它本来就要求逐 group 重算。

命中前做轻量、无同步的结构兼容检查。无法证明兼容时必须 cache miss 并走完整构建，不允许“尽量复用”。第一版可以先只复用完全同构的 10 个 GDN builders。

### 4.7 每 group materialize 的硬要求

每次 `_materialize_group_metadata()` 必须：

1. 使用当前 group 的 `common_attn_metadata.block_table_tensor` 调用 `mamba_get_block_table_tensor()`；
2. 从当前 block table 派生所有 state/cache indices；
3. 重新构造嵌套的 spec decode、non-spec prefill、non-spec decode metadata；
4. 将当前 group 的 cache indices 注入所有 causal-conv1d metadata；
5. 在 FULL graph 分支只写当前 builder 自己的 persistent buffer；
6. 返回新的 metadata 容器，禁止复用另一个 group 的可变容器。

### 4.8 临时诊断断言

调试阶段不采用“除了 4 个字段外全部相同”的模糊断言，而是：

- 对公共模板字段逐项与无 cache 基线比较；
- 对 group-dependent 字段逐项与当前 group 无 cache 构建结果比较；
- 检查 builder-local persistent buffers 在 group 间没有非法 alias；
- 对完整返回对象做递归的 shape、dtype、device、数值比较。

断言只用于测试或 debug，不能留在生产热路径造成 device sync。

---

## 5. 第一阶段测试方案

### 5.1 纯单元等价性测试

对同一输入分别执行：

```text
baseline：每个 GDN builder 完整 build
optimized：第一个 builder 建 template，后续 builder 复用 template
```

覆盖以下场景：

- 纯 prefill；
- 普通 decode；
- 全部 speculative decode；
- spec/non-spec 混合 batch；
- 有/无 accepted tokens；
- eager；
- FULL graph 对应的 padded batch 分支；
- 不同 group block table；
- align 和非 align mamba cache mode（若当前版本支持）。

逐字段断言 baseline 与 optimized 相等，尤其覆盖第 3.2 节列出的全部地址字段和嵌套 metadata。

### 5.2 调用次数测试

通过 monkeypatch/counter 验证 10 个兼容 GDN groups：

```text
公共 batch-template 构建：1 次
mamba_get_block_table_tensor：10 次
最终 GDNAttentionMetadata：10 份
```

第二次调用顶层 `build_attn_metadata()` 时，公共模板必须重新构建 1 次，证明没有跨 step 误命中。

### 5.3 不兼容与 fallback 测试

构造不同 `num_spec`、graph mode、prefill backend、MambaSpec 或 extra kwargs 的 builder，断言：

- 不复用不兼容模板；
- fallback 结果与原始完整 build 相同；
- 不抛出 block table shape 错误；
- 不产生跨 builder buffer alias。

### 5.4 顶层映射测试

保持 15 groups，断言：

- `attn_groups`、`block_tables`、`slot_mappings` 外层长度仍为 15；
- 返回字典覆盖全部真实层名且每层一次；
- 10 个 GDN group 的 metadata 使用各自 block table 派生的地址；
- base/draft FullAttention groups 行为完全不变。

---

## 6. 第一阶段性能与 NPU 验证

### 6.1 先补齐可复现 profiling

基线和修改后必须记录：

- 模型、draft 模型和对应 commit；
- TP、max model len、max num seqs、spec token 数；
- batch size 和每条请求的 query/context length；
- prefill/decode/spec/mixed 类型；
- eager 或 `FULL_DECODE_ONLY`；
- warmup 次数、采样次数；
- `_build_attention_metadata` p50/p90/p99；
- 各 backend builder 的调用次数和累计耗时；
- GDN 公共模板、per-group materialize、`mamba_get_block_table_tensor` 分项耗时。

计时应使用不会污染正式热路径的临时 profiling 方式。commit `5d69f096` 的诊断日志可作为定位参考，验收后删除逐 step INFO 或降为默认关闭的 debug。

### 6.2 正确性验证顺序

1. CPU/Mock UT；
2. NPU eager，小 batch、短上下文；
3. NPU eager，prefill/decode/spec 混合；
4. `FULL_DECODE_ONLY` capture + replay；
5. 真实权重 `/v1/chat/completions` 请求；
6. 稳态性能；
7. 长上下文和较大 batch 压测。

仅看到 `Application startup complete` 不算成功，必须有真实请求输出、EngineCore 存活和 graph replay 证据。

### 6.3 第一阶段验收门槛

正确性硬门槛：

- spec buckets 仍为 `[16, 5, 48]`；
- KV group 数仍为 15，GDN group 数仍为 10；
- KV cache token capacity、`num_blocks` 和 maximum concurrency 与同环境基线一致；
- 所有 metadata 字段与关闭 cache 的基线一致；
- 真实请求输出正确，DSpark acceptance 不异常归零；
- ACLGraph capture/replay 正常；
- MTP 和非 DSpark 路径无行为变化。

性能门槛：

- 兼容 GDN groups 的公共完整计算从 10 次降为 1 次；
- `mamba_get_block_table_tensor()` 仍为 10 次；
- 若新增 profiling 的 14/17 ms 基线成立，稳态 `build_attn_metadata` p50 目标下降至少 50%；
- p90/p99 不出现新长尾；
- TPOT/吞吐不回归超过 3%；
- 无新增 NPU `.item()` 同步或逐 step INFO。

`17 ms -> 5 ms` 只作为待验证的性能假设，不作为文档中的承诺值。

---

## 7. 第二阶段（可选）：通用 Pareto KV 分组选择

第一阶段已经直接解决 GDN 重复计算，第二阶段不是上线第一阶段的前置条件。只有在以下条件同时满足时才实施：

- 第一阶段稳定通过；
- 15 次 group 外层框架或非 GDN metadata 的剩余耗时仍值得优化；
- 能在实际 runtime 对应的 vLLM core 中修改并完整验证；
- 有明确的 upstream 或固定 fork 承载方案。

### 7.1 选择规则

先按当前 heuristic 得到 baseline `group_size`，计算 baseline 的概念 padding：

```python
num_groups = sum(cdiv(count, group_size) for count in layer_counts)
padded_slots = num_groups * group_size
padding = padded_slots - sum(layer_counts)
```

枚举合法候选，以当前 heuristic 结果作为候选之一，只允许：

```text
candidate_padding <= baseline_padding
```

在合法候选中按以下 tuple 最小化：

```text
(num_groups, padding, -group_size)
```

这保证按同一容量代理：

- group 数不多于现状；
- padding 不多于现状；
- 结果确定；
- 不需要 DSpark gating 或 padding 百分比 magic number。

应把“全局安全的严格改进”理解为 **在 uniform-page-size 分支和该数学代理上的 Pareto 不退化**，不能在缺少全模型测试时扩展为所有运行时行为都天然安全。

### 7.2 `[16, 5, 48]` 的结果

| group_size | groups | padded slots | padding | 是否满足不劣于当前 padding |
|---:|---:|---:|---:|---|
| 5（当前） | 15 | 75 | 6 | 是 |
| 6 | 12 | 72 | 3 | 是 |
| 8 | 9 | 72 | 3 | 是，group 最少 |
| 10 | 8 | 80 | 11 | 否 |
| 16 | 5 | 80 | 11 | 否 |

所以该规则选择 `group_size=8`：

```text
ceil(16 / 8) + ceil(5 / 8) + ceil(48 / 8)
= 2 + 1 + 6
= 9 groups
```

实际 group 层数为 `[8, 8, 5, 8, 8, 8, 8, 8, 8]`，等效槽位从 75 降为 72。按当前 uniform-page-size 分配模型，这不会造成原方案 `g=16` 的约 6.25% 容量损失；最终仍须以 `get_kv_cache_config_from_groups()` 结果和 NPU 启动日志实测。

### 7.3 修改落点与测试

修改应位于 runtime 对应的 vLLM core：

```text
vllm/v1/core/kv_cache_utils.py
tests/v1/core/test_kv_cache_utils.py
```

不在 vllm-ascend worker 中 monkey-patch 分组结果，因为 scheduler 和 worker 必须消费同一份 group list。

参数化测试至少覆盖：

- `[16,5,48] -> group_size=8, 9 groups`；
- `[17,48]` MTP 不退化；
- 单桶、等长桶、极小桶、互质桶；
- 当前 heuristic 本来最优的输入保持不变；
- 候选不得增加 baseline padding；
- 输入桶顺序不改变结果；
- stride 拆分后每个层名恰好一次；
- 组内 spec 可合并；
- PP rank 下的层分布；
- prefix-cache hit、KV config 构造和最大并发计算；
- uniform-page-size 分支以外的 DSV4/uniform-groups 路径不变。

第二阶段验收：

- DSpark groups 15 -> 9；
- GDN groups 10 -> 6；
- effective slots 75 -> 72；
- 实测 KV capacity 不低于第一阶段；
- scheduler/worker group 列表一致；
- 第一阶段公共模板仍只构建一次；
- correctness、acceptance、ACLGraph、TPOT/吞吐均满足第一阶段门槛。

---

## 8. 不采用的方案

### 8.1 强行统一 base 和 draft FullAttentionSpec

不可采用。`1 x 256` 与 `2 x 128` 即使每 token 元素数量接近，tensor shape、KV head 语义、view 和 kernel 参数仍不同，伪造 spec 会破坏 KV layout 或注意力结果。

### 8.2 合并不同 group 的完整 metadata

不可采用。不同 group 有独立 block table 和 slot mapping，worker 私自合并会破坏 scheduler/worker 的 group/block ID 契约。

### 8.3 完整 metadata 浅拷贝并替换 4 个字段

不可采用。group-dependent 字段多于 4 个，嵌套 causal-conv1d metadata 也携带 cache indices；FULL graph 还有 builder-local persistent buffer 的地址稳定性要求。

### 8.4 DSpark 专用 `group_size=16`

不再推荐。它把 groups 降到 5，但 padded slots 从 75 增至 80，理论 KV capacity 约下降 6.25%；有了公共模板后，9 -> 5 groups 的额外收益需要重新实测，不能预先用容量交换。

### 8.5 全局固定 `MIN_GROUP_SIZE`

不可采用。固定常数不能适配不同 hybrid layer 分布，可能显著扩大 padding。

### 8.6 跨 step cache

第一版不可采用。跨 step 需要处理请求重排、batch size、spec accepted tokens、graph padding、tensor 生命周期和缓存失效，收益不足以抵消正确性风险。

---

## 9. 实施步骤

### 阶段 A：基线与证据补齐

1. 确认实际 import 的 vLLM/vllm-ascend 路径和 commit。
2. 用真实权重复现 `[16,5,48]`、15 groups、10 GDN builders。
3. 附原始 profiling，确认 14/17 ms 数据的测量条件和稳定性。
4. 记录 KV capacity、maximum concurrency、acceptance、TTFT、TPOT、吞吐。

### 阶段 B：公共模板重构

1. 从 GDN `build()` 提取 group-independent 数据依赖。
2. 定义 `_GDNBatchedMetadataTemplate`。
3. 保留原始 full-build 路径。
4. 实现 per-group KV 地址 materialize。
5. 处理 FULL graph builder-local buffer。

### 阶段 C：batch-scope 接入

1. 在 `attn_utils.py` group 循环外创建局部 cache。
2. 实现严格 compatibility key/check。
3. 不兼容时自动 full build。
4. 不改变 group 顺序、block table 或 slot mapping。

### 阶段 D：UT 与 NPU 验证

1. 运行字段等价、调用次数、fallback、cache 生命周期测试。
2. 运行 ruff、仓库 format 检查和相关回归测试。
3. 执行 eager -> FULL_DECODE_ONLY -> 真实请求 -> 稳态性能验证。
4. 删除临时断言和逐 step 诊断日志。

### 阶段 E：决定是否进入通用分组优化

1. 用第一阶段 profile 判断剩余 group 框架耗时。
2. 若收益足够，向匹配 runtime 的 vLLM core 实现 Pareto 选择器。
3. 先做通用 core 单测，再做 Ascend 端到端验证。
4. 不引入 `use_dspark()` gating，不引入环境变量。

---

## 10. 回退策略

第一阶段应保持无 cache full-build 路径。出现以下任一情况，关闭/回退公共模板接入：

- baseline 与 optimized 任一 metadata 字段不一致；
- block table/cache index 越界；
- builder buffer 非预期 alias；
- eager 正确但 ACLGraph capture/replay 失败；
- acceptance 或输出精度异常；
- p90/p99、TPOT 或吞吐超过门槛；
- profiling 证明公共计算不是主要耗时，优化收益不足。

第二阶段独立提交，出现以下任一情况只回退分组选择器，不回退已验证的第一阶段：

- 非目标模型分组或容量退化；
- scheduler/worker group 不一致；
- prefix cache、PP、KV config 构造异常；
- NPU 实测 capacity 低于第一阶段；
- 收益不足以支撑 core 改动维护成本。

---

## 11. 预期最终状态

第一阶段目标状态：

```text
spec buckets:                  [16, 5, 48]
kv_cache_groups:               15（不变）
GDN group metadata objects:    10（保持独立）
GDN common template builds:     1
GDN block-table materialize:   10
KV capacity:                   与基线完全一致
```

可选第二阶段目标状态：

```text
spec buckets:                  [16, 5, 48]
selected group_size:            8
kv_cache_groups:                9
GDN groups:                     6
GDN common template builds:     1
effective slots:               72（当前为 75）
```

最终成功标准不是单独看到 group 数下降，也不是仅看到一次 profile 变快，而是同时满足：

```text
公共计算只做一次
+ 每组 KV 地址仍独立正确
+ graph buffer 生命周期正确
+ 真实推理与 acceptance 正确
+ KV capacity 不退化
+ 稳态 metadata/TPOT/吞吐确有收益
```
