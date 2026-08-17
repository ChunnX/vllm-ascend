# Qwen3.6 DSpark KV Cache Group 过度拆分恢复方案

## 1. 文档目标

本文给出 Qwen3.6 + DSpark 场景下 KV cache group 从 15 个收敛到 5 个的详细修改方案，目标是在不伪造 KV cache spec、不改变模型计算语义、不破坏 scheduler/worker KV 地址映射的前提下，降低 `build_attn_metadata` 的循环次数和 CPU 侧构建开销。

本文是实施计划，不表示代码已经修改或 NPU 验证已经完成。实施完成后必须以真实权重请求、KV 容量、metadata 耗时和输出正确性作为最终验收依据。

相关证据：

- `kv_spec_recover/dspark_kv_group_issue_analysis.md`
- `kv_spec_recover/dspark_kv_log_evidence.md`
- `kv_spec_recover/mtp_vs_dspark_log_evidence.md`
- `kv_spec_recover/dspark_kv_analyze.info`
- `kv_spec_recover/mtp_kv_analyze.info`
- 诊断 commit：`5d69f096225b96aa3b95d4a7f4a9e4328b8972db`

实际日志文件名以本仓库为准，分别是 `dspark_kv_analyze.info` 和 `mtp_kv_analyze.info`，不是早期分析文档中写的 `/opt/z00830407/*.log`。

---

## 2. 已确认的问题与证据边界

### 2.1 已确认事实

Qwen3.6 + MTP 的 KV spec 桶大小为：

```text
[17, 48]

17 = 16 个 base FullAttentionSpec + 1 个 MTP FullAttentionSpec
48 = 48 个 MambaSpec
```

MTP 层与 base attention 层的关键规格一致：

```text
num_kv_heads = 1
head_size = 256
head_size_v = 256
```

因此当前算法选择 `group_size=17`：

```text
ceil(17 / 17) + ceil(48 / 17) = 1 + 3 = 4 groups
```

Qwen3.6 + DSpark 的 KV spec 桶大小为：

```text
[16, 5, 48]

16 = base FullAttentionSpec
5  = DSpark draft FullAttentionSpec
48 = MambaSpec
```

DSpark draft attention 的关键规格为：

```text
num_kv_heads = 2
head_size = 128
head_size_v = 128
```

它与 base 的 `1 x 256` 不相等，因而形成独立的 5 层 spec 小桶。当前 `_get_kv_cache_groups_uniform_page_size()` 直接选择最小桶大小：

```text
group_size = min(16, 5, 48) = 5
```

最终得到：

```text
ceil(16 / 5) + ceil(5 / 5) + ceil(48 / 5)
= 4 + 1 + 10
= 15 groups
```

### 2.2 `build_attn_metadata` 的放大关系

vLLM Ascend v2 的 `build_attn_metadata()` 按 `kv_cache_groups` 外层循环，每个 group 使用自己的：

- `block_tables[i]`
- `slot_mappings[i]`
- `AscendCommonAttentionMetadata`
- attention metadata builder

因此 group 数从 4 增加到 15，必然将外层构建单元从 4 增加到 15。

但必须区分以下两件事：

- 日志已证明 DSpark 有 15 个 group、MTP 有 4 个 group；
- DSpark 日志没有记录 `[DIAG-BUILD-META-V2]` 和逐次耗时，因此不能把“3.75 倍构建单元”直接写成“实测耗时精确增加 3.75 倍”。

实施时需要补充完整计时证据。

### 2.3 排除项

以下因素不是这次 group 膨胀的主因：

1. Mamba 层重复收集：两条路径均为 48 个 MambaSpec。
2. `indexes_kv_by_block_stride`：它在 MTP 与 DSpark 间不同，但 DSpark 内部两个 FullAttentionSpec 桶都为 `True`，两个桶仍然因为 KV head/head size 不同而分离。
3. 单纯层数多 4 层：如果 DSpark 的 5 层能与 base spec 合桶，即 `[21, 48]`，仍只会产生 4 个 group。

---

## 3. 修改目标和非目标

### 3.1 修改目标

第一阶段目标：

1. DSpark `[16, 5, 48]` 从 15 个 group 降到 5 个。
2. `build_attn_metadata` 外层循环从 15 次降到 5 次。
3. MTP `[17, 48]` 保持 4 个 group，不发生行为变化。
4. 非 DSpark 模型完全沿用当前 group-size heuristic。
5. 每个真实 KV 层恰好出现一次，组内所有层保持相同 KV spec。
6. scheduler、worker、KV cache allocator 使用同一份 group 列表。
7. 真实权重推理结果、DSpark acceptance 和 ACLGraph 行为不回归。

### 3.2 明确非目标

本次不做以下改动：

- 不把 DSpark 的 `2 x 128` FullAttentionSpec 强行改成 base 的 `1 x 256`。
- 不修改模型权重、attention head 拓扑或 draft 模型结构。
- 不在 `build_attn_metadata()` 内盲目合并拥有不同 block table 的 group。
- 不按 `model.layers.64` 等层名字符串识别 draft 层。
- 不增加全局固定的 `MIN_GROUP_SIZE=16`。
- 不默认修改所有 hybrid 模型的分组策略。
- 不新增环境变量；如果后续确实需要开关，必须按 `vllm_ascend/envs.py` 的评审要求单独设计。
- 不在 metadata 热路径加入 `tensor.item()` 或额外 CPU-NPU 同步。

---

## 4. 方案选择

### 4.1 推荐方案：带 padding 预算的 DSpark group-size 搜索

当前算法只考虑“最小桶”，没有同时优化：

- group 数量；
- padding 数量；
- metadata 构建成本。

推荐在 DSpark 场景枚举候选 `group_size`，在总 padding 比例不超过上限的候选中：

1. 优先选择 group 数最少的候选；
2. group 数相同时选择 padding 最少的候选；
3. 仍相同时选择更大的 `group_size`，保证结果确定。

第一版使用命名常量：

```python
DSPARK_KV_GROUP_MAX_PADDING_RATIO = 0.20
```

`0.20` 只应用于 `speculative_config.use_dspark()` 的通用 hybrid KV 分组分支，不改变非 DSpark 行为。

### 4.2 当前模型的候选结果

对于 `[16, 5, 48]`：

| group_size | group 数 | 等效槽位 | padding | 总 padding 比例 | 说明 |
|---:|---:|---:|---:|---:|---|
| 5 | 15 | 75 | 6 | 8.70% | 当前实现 |
| 8 | 9 | 72 | 3 | 4.35% | KV 容量优先的备选 |
| 10 | 8 | 80 | 11 | 15.94% | 被 16 支配 |
| 12 | 7 | 84 | 15 | 21.74% | 超出 20% 上限 |
| 16 | 5 | 80 | 11 | 15.94% | 推荐结果 |
| 17 | 5 | 85 | 16 | 23.19% | 超出 20% 上限 |
| 24 | 4 | 96 | 27 | 39.13% | padding 代价过高 |

因此在 20% 总 padding 上限内，`group_size=16` 是 group 数最少的合法候选，结果为：

```text
base FullAttentionSpec:    16 -> 1 group
DSpark FullAttentionSpec:   5 -> 1 group
MambaSpec:                 48 -> 3 groups

total = 5 groups
```

结果 group 的实际层数预计是：

```text
[16, 5, 16, 16, 16]
```

DSpark group 不需要伪造 11 个层名。下游 `get_kv_cache_config_from_groups()` 使用最大真实 group 长度作为物理池槽位数，并通过 `i < len(group.layer_names)` 表达短 group 的隐式 padding。

### 4.3 内存代价预估

当前 `group_size=5` 的等效槽位是 75；推荐 `group_size=16` 的等效槽位是 80。相同可用内存下，KV block/token capacity 粗略比例为：

```text
75 / 80 = 93.75%
```

即相对当前 DSpark 配置，理论上可能损失约 6.25% KV 容量。日志中的当前容量为 186,413 tokens，按比例粗估修改后约为：

```text
186,413 * 75 / 80 ~= 174,762 tokens
```

该数字仅用于制定验收阈值，最终必须读取修改后启动日志的实际 `GPU KV cache size` 和 `Maximum concurrency`，不能把估算作为实测结果。

### 4.4 为什么不采用其他方案

#### 强行统一 FullAttentionSpec

不可采用。虽然 `1 x 256` 和 `2 x 128` 每 token 的元素总量接近，但 tensor shape、KV head 语义、view 和 kernel 参数不同。伪造 spec 可能产生错误 KV layout 或错误注意力结果。

#### 仅在 Ascend worker 合并 metadata

不可采用。不同 group 拥有独立的 block table 和 slot mapping。worker 私自合并后会与 scheduler 分配的 group/block ID 不一致。

#### 全局 `MIN_GROUP_SIZE=16`

不可采用。它会改变所有 hybrid 模型，并可能对小模型或其他 attention pattern 产生不可控 padding。

#### 只把 draft 层从输入中摘出再追加

它在当前模型上也能得到 5 个 group，但依赖可靠的 draft 层身份传递。当前 `KVCacheSpec` 没有通用的 `is_draft` 标签，使用层名匹配又很脆弱。padding-budget 搜索不依赖层名和具体层编号，改动更小。

---

## 5. 修改范围与仓库边界

### 5.1 生产逻辑修改：vLLM core

真实运行环境应修改：

```text
/vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py
```

本地分析副本对应：

```text
/opt/zsy/vllm-0.26.0/vllm/v1/core/kv_cache_utils.py
```

必须在 vLLM core 修改，因为 `get_kv_cache_groups(vllm_config, kv_cache_spec)` 是 scheduler/worker 生成统一分组的入口。不能只在 vLLM Ascend worker 中修改返回结果。

### 5.2 vLLM 单元测试

建议新增或扩展：

```text
/vllm-workspace/vllm/tests/v1/core/test_kv_cache_utils.py
```

测试纯 Python 分组、KV 配置构造和非 DSpark 回归，不依赖 NPU。

### 5.3 vLLM Ascend 修改

生产路径不应增加第二套分组算法。vLLM Ascend 只需要：

1. 在验证阶段补充 metadata 总耗时和 builder 分类耗时；
2. 补充 DSpark 端到端回归；
3. 验证 `attn_groups`、`block_tables`、`slot_mappings` 与新 group 数一致；
4. 最终移除或降级 commit `5d69f096` 引入的逐层/逐请求 INFO 诊断日志，避免日志本身污染性能。

涉及文件：

```text
/vllm-workspace/vllm-ascend/vllm_ascend/worker/v2/attn_utils.py
/vllm-workspace/vllm-ascend/tests/ut/worker/v2/...（按现有测试目录落位）
```

如果上游 vLLM 修改暂时不能合入，可在运行镜像固定的 vLLM fork 中携带补丁；不建议用长期 monkey patch 覆盖 core 函数。

---

## 6. 详细代码设计

### 6.1 新增命名常量

在 `kv_cache_utils.py` 模块常量区新增：

```python
DSPARK_KV_GROUP_MAX_PADDING_RATIO = 0.20
```

该常量说明应明确：

- 这是总 padding 层数相对真实 KV 层数的上限；
- 只用于 DSpark 的 general uniform-page-size hybrid 路径；
- 目的是限制为减少 metadata group 而付出的 KV 容量代价。

### 6.2 抽取旧 heuristic

将当前逻辑抽为一个小函数，保持非 DSpark 行为完全一致：

```python
def _get_default_kv_group_size(layer_counts: Sequence[int]) -> int:
    min_num_layers = min(layer_counts)
    max_num_layers = max(layer_counts)
    if max_num_layers < min_num_layers * 1.5:
        return max_num_layers
    return min_num_layers
```

注意：不能顺手改变 `<` 为 `<=`，也不能调整 1.5 heuristic，否则会扩大回归范围。

### 6.3 新增候选评价数据

建议使用局部 tuple，不需要新增可变全局对象：

```python
def _evaluate_group_size(
    layer_counts: Sequence[int],
    group_size: int,
) -> tuple[int, int, float]:
    num_groups = sum(cdiv(count, group_size) for count in layer_counts)
    padded_slots = num_groups * group_size
    total_layers = sum(layer_counts)
    padding_layers = padded_slots - total_layers
    padding_ratio = padding_layers / total_layers
    return num_groups, padding_layers, padding_ratio
```

### 6.4 DSpark 搜索函数

建议实现：

```python
def _select_dspark_kv_group_size(
    layer_counts: Sequence[int],
    max_padding_ratio: float = DSPARK_KV_GROUP_MAX_PADDING_RATIO,
) -> int:
    default_group_size = _get_default_kv_group_size(layer_counts)
    best_group_size = default_group_size
    best_num_groups, best_padding, _ = _evaluate_group_size(
        layer_counts, default_group_size
    )

    # 不枚举比默认 group_size 更小的候选，因为它只可能增加 group 数。
    for candidate in range(default_group_size + 1, max(layer_counts) + 1):
        num_groups, padding_layers, padding_ratio = _evaluate_group_size(
            layer_counts, candidate
        )
        if padding_ratio > max_padding_ratio:
            continue

        candidate_score = (num_groups, padding_layers, -candidate)
        best_score = (best_num_groups, best_padding, -best_group_size)
        if candidate_score < best_score:
            best_group_size = candidate
            best_num_groups = num_groups
            best_padding = padding_layers

    return best_group_size
```

边界要求：

- `layer_counts` 非空且所有元素大于 0；
- 默认 heuristic 永远是合法 fallback，即使它超过 padding budget 也不能返回空结果；
- 只在模型初始化阶段执行，最大枚举范围是层数，时间复杂度可忽略；
- 结果必须确定，不依赖 dict hash 随机顺序。

### 6.5 修改分组函数签名

推荐为 `_get_kv_cache_groups_uniform_page_size()` 增加显式可选参数：

```python
def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
    *,
    optimize_for_fewer_groups: bool = False,
) -> list[KVCacheGroupSpec]:
```

内部逻辑：

```python
layer_counts = [len(layers) for layers in same_type_layers.values()]
if optimize_for_fewer_groups:
    group_size = _select_dspark_kv_group_size(layer_counts)
else:
    group_size = _get_default_kv_group_size(layer_counts)
```

后续 stride 拆分逻辑保持原样：

```python
num_groups = cdiv(len(layers), group_size)
for i in range(num_groups):
    grouped_layers.append(layers[i::num_groups])
```

保持 stride 拆分可以继续兼容原有 PP 均衡语义。当前 DSpark 本身不支持 PP，但不应为本修复改坏公共函数。

### 6.6 在统一入口识别 DSpark

`get_kv_cache_groups()` 已经接收 `vllm_config`，使用配置接口识别，不使用层名：

```python
speculative_config = vllm_config.speculative_config
optimize_for_fewer_groups = bool(
    speculative_config is not None
    and speculative_config.use_dspark()
)

groups = _get_kv_cache_groups_uniform_page_size(
    filtered_spec,
    optimize_for_fewer_groups=optimize_for_fewer_groups,
)
```

这个判断只能放在 general uniform-page-size 分支。以下提前返回路径继续保持现状：

- attention-free；
- uniform spec；
- `UniformTypeKVCacheSpecs`；
- DeepSeek V4 `group_and_unify_kv_cache_specs()` 特殊路径。

特别注意：DeepSeek V4 DSpark 可能走已有 tuple packing/UniformType 分支，不应被 Qwen3.6 的 general hybrid heuristic 意外覆盖。

### 6.7 日志设计

初始化阶段增加一条汇总 INFO 日志即可：

```text
DSpark KV grouping selected group_size=16 for layer_counts=[16, 5, 48],
groups=5, padding_layers=11, padding_ratio=15.94%
```

要求：

- 只记录汇总，不逐层打印；
- 排序后输出 `layer_counts`，避免日志顺序不稳定；
- 非 DSpark 不增加新 INFO；
- 不在每个 decode step 打 INFO。

commit `5d69f096` 中的 `[DIAG-BUILD-META-*]` 当前位于热路径。性能验证完成后应删除，或降级为仅临时 profiling 使用，不能作为生产默认日志保留。

---

## 7. 单元测试方案

### 7.1 vLLM core：搜索函数测试

在 `tests/v1/core/test_kv_cache_utils.py` 增加参数化测试：

| layer_counts | 模式 | 预期 group_size | 预期 group 数 | 目的 |
|---|---|---:|---:|---|
| `[16, 5, 48]` | DSpark 优化 | 16 | 5 | 本问题回归 |
| `[17, 48]` | 默认 | 17 | 4 | MTP/非 DSpark 不变 |
| `[12, 13]` | 默认 | 13 | 2 | 保留现有 1.5 heuristic |
| `[10, 20]` | 默认 | 10 | 3 | 边界不改变 |
| `[1]` | DSpark 优化 | 1 | 1 | 单桶边界 |
| `[5, 5, 5]` | DSpark 优化 | 5 | 3 | 无需扩大 group size |

额外断言：

- 选择结果的 padding ratio 不超过 20%，除非默认 fallback 本身已经超过预算；
- 输入顺序变化不影响结果；
- 空数组和非正数输入触发明确断言或 `ValueError`。

### 7.2 vLLM core：真实 spec 分组测试

构造三个不同的 KV spec identity：

1. 16 个 base FullAttentionSpec：`num_kv_heads=1, head_size=256`；
2. 5 个 draft FullAttentionSpec：`num_kv_heads=2, head_size=128`；
3. 48 个 MambaSpec；

确保三者 `page_size_bytes` 已统一，然后调用 `get_kv_cache_groups()` 或内部函数。

DSpark 模式断言：

```text
len(groups) == 5
sorted(len(group.layer_names) for group in groups) == [5, 16, 16, 16, 16]
```

还必须断言：

- 69 个真实层名的集合与输入完全相同；
- 没有重复层名；
- 每组所有层的原始 spec 与该组 merged spec 兼容；
- attention group 在 Mamba group 之前的稳定顺序不被破坏；
- stride 后每个 Mamba group 为 16 层。

非 DSpark 模式断言仍产生当前 15 个 group，证明开关范围受控。

### 7.3 KV 配置构造测试

对 `[16, 5, 16, 16, 16]` 调用 `get_kv_cache_config_from_groups()`，断言：

- `len(kv_cache_tensors) == 16`；
- 每个 tensor 的 `shared_by` 只包含该 slot 上实际存在的层；
- slot 5 到 15 不包含 draft 层，但包含其他 group 的对应层；
- `num_blocks` 使用 `group_size=16` 计算；
- 所有 group 的 page size 一致；
- 没有空 `shared_by` tensor。

### 7.4 vLLM Ascend metadata 测试

为 `build_attn_metadata()` 构造 5 个 group 的 mock 输入，断言：

- `block_tables` 长度为 5；
- `slot_mappings` 第一维/序列长度符合现有接口；
- `attn_groups` 外层长度为 5；
- 每个 group 取对应下标的 block table 和 slot mapping；
- 返回字典覆盖全部 69 个 layer name；
- 同一 group 内允许共享 metadata，不同 group 不共享带有不同 block table 的 metadata。

---

## 8. 性能观测设计

### 8.1 必须补充的计时点

在临时 profiling 版本中使用 `time.perf_counter_ns()` 记录：

1. 整个 `build_attn_metadata()` 总耗时；
2. 每个 KV group 的 common metadata 构造耗时；
3. 每类 metadata builder 的 `build()` 耗时；
4. Python 映射 `for layer_name in attn_group.layer_names` 的耗时。

计时必须满足：

- 不调用设备 tensor 的 `.item()`；
- 不做 NPU synchronize；
- 不在每步直接 INFO 打印；
- 先累计 CPU 标量，测试结束或固定低频周期再汇总；
- 正式提交前移除临时 instrumentation，或接入现有 observability 体系且默认关闭。

### 8.2 对比方法

使用同一台机器、相同镜像、相同模型、相同 TP、相同请求和相同图模式，对比：

```text
baseline: group_size=5,  15 groups
candidate: group_size=16, 5 groups
```

分别采集：

- warmup 后至少 200 个 decode step；
- `build_attn_metadata` p50/p90/p99；
- TTFT、TPOT、端到端吞吐；
- DSpark acceptance length/rate；
- CPU 利用率；
- `GPU KV cache size` 和最大并发；
- ACLGraph replay 是否正常。

不要把 graph capture 首轮或模型编译时间混入稳态 metadata 数据。

---

## 9. NPU 端到端验证方案

### 9.1 运行环境检查

在容器中确认实际导入路径：

```bash
cd /workspace
python - <<'PY'
import vllm
import vllm_ascend
print(vllm.__file__)
print(vllm_ascend.__file__)
PY
```

期望来自 `/vllm-workspace/vllm` 和 `/vllm-workspace/vllm-ascend`，不能只修改 `/opt/zsy/vllm-0.26.0` 的分析副本后直接宣称验证通过。

### 9.2 第一阶段：复现配置

先复用日志中的容量配置，降低变量数量：

```bash
cd /workspace

HCCL_OP_EXPANSION_MODE=AIV \
vllm serve /opt/foundation_model/Qwen3.6-27B \
  --served-model-name qwen3 \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --async-scheduling \
  --speculative-config \
  '{"num_speculative_tokens":7,"method":"dspark","model":"/opt/foundation_model/Qwen3.6-27B-Dspark-0810-64000"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 8000
```

如果实际模型路径变化，以运行环境为准更新命令和报告。

### 9.3 必须执行真实请求

服务 ready 后执行：

```bash
curl -sf http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3",
    "messages":[{"role":"user","content":"用一句话解释 KV cache。"}],
    "temperature":0,
    "max_tokens":128
  }'
```

验收要求：HTTP 200、输出非空、EngineCore 不退出。`Application startup complete` 不能单独作为成功依据。

### 9.4 图模式与 eager 隔离

默认验收 FULL_DECODE_ONLY ACLGraph 路径，并保留 `run_fullgraph`/graph replay 证据。

如果图模式失败，增加 `--enforce-eager` 仅用于判断错误是否来自 graph capture。eager 成功不能替代图模式最终验收，除非明确接受 eager 作为临时回退。

### 9.5 容量扩展

在复现配置通过后，按硬件实际能力扩展：

1. `max-model-len=16k, max-num-seqs=4`；
2. 保持 16k，增加 `max-num-seqs`；
3. 尝试技能要求的 `128k + bs16` 基线；
4. 如果 128k + bs16 不可行，记录 NPU 内存、TP、KV capacity 或模型限制，不得只写“未测试”。

---

## 10. 验收标准

### 10.1 正确性硬门槛

- [ ] DSpark spec 桶仍为 `[16, 5, 48]`，没有伪造或合并不同 FullAttentionSpec。
- [ ] 最终 `kv_cache_groups == 5`。
- [ ] group 层数为 `[16, 5, 16, 16, 16]`（顺序按 attention/Mamba 稳定规则）。
- [ ] `attn_groups`、`block_tables`、`slot_mappings` 外层长度一致。
- [ ] 所有 69 个层名恰好出现一次。
- [ ] 真实权重请求 HTTP 200 且输出非空。
- [ ] DSpark acceptance 指标正常，无明显异常归零。
- [ ] ACLGraph capture/replay 正常。
- [ ] MTP 仍为 4 个 group。
- [ ] 非 DSpark grouping 单测全部通过。

### 10.2 性能门槛

- [ ] `build_attn_metadata` 外层构建次数从 15 降到 5。
- [ ] 稳态 `build_attn_metadata` p50 至少下降 50%。
- [ ] p90/p99 不出现新的长尾退化。
- [ ] TPOT 不退化超过 3%。
- [ ] 吞吐不退化超过 3%，预期应有改善。
- [ ] KV token capacity 相对当前 DSpark 基线损失不超过 10%。
- [ ] 无新增 CPU-NPU 同步和热路径 INFO 日志。

### 10.3 容量门槛

当前 DSpark 日志容量为：

```text
GPU KV cache size: 186,413 tokens
Maximum concurrency for 16,384 tokens per request: 11.38x
```

`group_size=16` 的粗略容量下限按 90% 验收：

```text
GPU KV cache size >= 167,771 tokens
```

该门槛是相对当前相同配置的回归界限，不是跨机器固定值。

---

## 11. 回退与分阶段策略

### 11.1 推荐路径

如果 `group_size=16` 满足正确性、metadata 和容量门槛，则作为最终方案：

```text
15 groups -> 5 groups
```

### 11.2 容量优先回退

如果 `group_size=16` 的实际 KV capacity 损失超过 10%，使用 `group_size=8` 做第二候选：

```text
groups = ceil(16/8) + ceil(5/8) + ceil(48/8)
       = 2 + 1 + 6
       = 9

effective slots = 9 * 8 = 72
```

它相对当前 75 个等效槽位理论上不损失容量，同时仍将 metadata group 从 15 降到 9。此路径需要通过显式策略参数或调整候选目标实现，不能临时硬编码后直接提交。

### 11.3 完整回滚条件

出现以下任何情况应回滚分组改动：

- 输出正确性变化；
- KV cache shape/view 错误；
- block table 或 slot mapping 越界；
- ACLGraph capture/replay 失败且确认由新 grouping 引起；
- DSpark acceptance 明显退化；
- KV capacity 损失超过约定阈值且 8-slot 备选也不满足要求；
- 非 DSpark 模型 grouping 行为变化。

回滚仅恢复 group-size 选择逻辑，不应删除本次补充的无副作用单元测试和根因文档。

---

## 12. 实施步骤

### 阶段 A：建立可重复基线

1. 确认 runtime import 路径。
2. 使用真实权重复现 15 groups。
3. 记录 KV capacity、acceptance、metadata p50/p90/p99、TTFT、TPOT、吞吐。
4. 保存完整启动命令、commit 和日志路径。

### 阶段 B：实现 vLLM core 选择器

1. 抽取默认 heuristic，确保行为不变。
2. 新增候选评价与 DSpark 选择器。
3. 给 uniform-page-size 分组函数增加显式开关。
4. 在 `get_kv_cache_groups()` 使用 `use_dspark()` 激活。
5. 增加一条初始化汇总日志。

### 阶段 C：补齐 CPU 单测

1. 搜索函数参数化测试。
2. `[16,5,48]` 真实 spec 回归测试。
3. KV tensor `shared_by`/隐式 padding 测试。
4. MTP 和非 DSpark 不变测试。
5. 运行目标测试文件、ruff 和格式检查。

### 阶段 D：补齐 Ascend 验证

1. dummy 可用于快速验证结构路径，但不能作为最终结论。
2. 使用真实权重启动。
3. 执行首请求 smoke，排除 false-ready。
4. 验证 ACLGraph。
5. 采集稳态性能和 KV capacity。
6. 如失败，用 eager 隔离，但继续定位图模式根因。

### 阶段 E：清理和交付

1. 删除临时逐步计时/逐层 INFO 日志。
2. 保留一条初始化分组汇总日志。
3. 更新本目录的日志证据和结果文档。
4. 运行 `bash format.sh ci`。
5. 按仓库要求生成一个 signed-off Conventional Commit。
6. 在最终报告中明确 upstream vLLM commit 与 vLLM Ascend commit/依赖关系。

---

## 13. 建议测试命令

vLLM 单元测试：

```bash
cd /vllm-workspace/vllm
pytest -sv tests/v1/core/test_kv_cache_utils.py -k 'dspark or group_size'
ruff check vllm/v1/core/kv_cache_utils.py tests/v1/core/test_kv_cache_utils.py
```

vLLM Ascend 目标单测按最终落位路径执行，例如：

```bash
cd /vllm-workspace/vllm-ascend
pytest -sv tests/ut/worker/v2 -k 'attn_metadata and dspark'
ruff check vllm_ascend/worker/v2/attn_utils.py tests/ut/worker/v2
```

全类型格式检查：

```bash
cd /vllm-workspace/vllm-ascend
bash format.sh ci
```

---

## 14. 预期最终结果

成功落地后，日志应体现：

```text
spec buckets: [16, 5, 48]
DSpark KV grouping selected group_size=16
padding_layers=11
padding_ratio=15.94%
get_kv_cache_groups final result: 5 kv_cache_groups
build_attn_metadata: 5 kv_cache_groups, 5 total attn_groups
```

与原始实现相比：

| 指标 | 修改前 | 目标值 |
|---|---:|---:|
| KV spec 桶 | 3 | 3，不改变真实 spec |
| group_size | 5 | 16 |
| kv_cache_groups | 15 | 5 |
| metadata 外层构建单元 | 15 | 5 |
| 等效 KV 槽位 | 75 | 80 |
| 理论 KV capacity | 100% | 约 93.75% |
| 真实权重请求 | 成功 | 必须继续成功 |
| ACLGraph | 成功 | 必须继续成功 |

最终判断标准不是“成功打印 5 groups”，而是同时满足：

```text
分组正确 + KV 地址正确 + 真实推理正确 + metadata 明显加速 + 容量损失受控
```
