# Qwen3.6 DSpark KV Cache Group 过度拆分恢复方案（评审与实施计划）

## 0. 对 codex_dspark_kv_spec_recover.md 的结论

**结论：该计划技术方向正确、可行，核心机制经源码逐行核实无误。本计划认同其"padding 预算搜索"方案，在此基础上补充范围澄清、把"改 vLLM core"作为第一闸口、并把 scheduler/worker 一致性作为头号失败风险显式化。**

已逐条核实的关键点（非空口背书）：

| 计划声明 | 源码/日志核实结果 |
|---|---|
| `use_dspark()` 存在 | ✅ `vllm/config/speculative.py:1333`，且已在 `scheduler.py:259`、`sparse_swa.py:380` 使用 |
| `group_size=16 → 5 组` | ✅ 手推：`cdiv(16,16)+cdiv(5,16)+cdiv(48,16)=1+1+3=5` |
| 分组结果 `[16,5,16,16,16]` | ✅ stride 拆分：base 1 组 16 层、draft 1 组 5 层、mamba 3 组各 16 层 |
| 短组（draft 5 层）被下游容忍 | ✅ `get_kv_cache_config_from_groups` 用 `group_size=max(len(...))` + `i < len(...)` 做隐式 padding |
| 容量损失约 6.25% | ✅ 与 `get_max_concurrency_for_kv_cache_config` 推导一致（等效槽位 75→80，`num_block_per_request = num_groups × max_model_len / block_size`） |
| 必须改 core 而非插件 | ✅ `get_kv_cache_groups` 是 scheduler(`kv_cache_utils.py:2091`)与 worker(`gpu_model_runner.py:6447`)共用入口 |
| `speculative_config` 在 core 侧可用 | ✅ 同文件 `_annotate_eagle_groups_deepseek_v4` 已按 `vllm_config.speculative_config.use_eagle()` 分派，先例成立 |

对计划的 3 处保留（不影响方向，需在执行时显式处理）：

1. **"改 vLLM core"是真正的第一闸口。** vllm-ascend 是硬件插件，其 AGENTS.md 明确"不直接新增模型文件/须走 patch 或继承"。`_get_kv_cache_groups_uniform_page_size` 是 core 内部函数，插件无法干净地改变其内部行为；计划的正确做法是 core 改动，但必须二选一：**上游合入 vllm-project/vllm（首选）或 fork 打补丁（过渡）**。这一点计划只写在 5.3 末尾，本计划将其提升为前置决策。
2. **scheduler 与 worker 的 group 一致性是头号失败风险。** gating 依赖 `vllm_config.speculative_config.use_dspark()` 在两处都返回 True。若任一侧 `speculative_config` 为空，会出现 15 组 vs 5 组不一致，直接导致 block table / slot_mapping 越界。必须先验 + 加显式一致性断言。
3. **`padding_ratio` 是"概念 padding"，不是 stride 拆分的精确 padding。** 计划用 `num_groups × group_size − total` 作排序依据是合理代理，但真实 padding 由 `layers[i::num_groups]` 决定（组内层数为 `ceil/floor`）。它只用于候选排序、不用于精确记账，可接受；精确容量必须靠 7.3 的构造测试 + NPU 实测 `GPU KV cache size`。

---

## 1. 范围澄清（重要，回应文件名里的 "dsv4"）

本方案针对的是**已由日志证实**的 Qwen3.6 + DSpark 问题，它走的是 `_get_kv_cache_groups_uniform_page_size`（general hybrid 分支，日志 `Path: uniform_page_size`）。

**DeepSeek V4 + DSpark 不在此方案的覆盖范围内**，原因：

- DSV4 走 `group_and_unify_kv_cache_specs()` → `_get_kv_cache_groups_uniform_groups()`（`kv_cache_utils.py:1525/1603`），使用 `_approximate_gcd` 最小化 padding，是**另一套分组算法**，没有本问题里的"最小桶拖低 group_size"机制。
- DSV4 的 EAGLE 层标记走 `_annotate_eagle_groups_deepseek_v4()`（`use_eagle()` 且 `model_version=="deepseek_v4"`）。

本方案的 gating 只放在 general uniform-page-size 分支，天然不会触碰 DSV4 路径。**若后续确认 DSV4 + DSpark 也有分组膨胀，需另立方案分析其 `_approximate_gcd` 行为，不能复用本方案。**

---

## 2. 已确认的事实与证据边界

（与 codex 计划一致，此处只列结论，证据详见 `dspark_kv_log_evidence.md` / `mtp_vs_dspark_log_evidence.md`）

- MTP（v1）：spec 桶 `[17, 48]` → `group_size=17` → 4 组（`1+3`）。
- DSpark（v2）：spec 桶 `[16, 5, 48]` → `group_size=min=5` → 15 组（`4+1+10`）。
- 根因：draft 层 `num_kv_heads=2/head_size=128` 与 base `1/256` 不同，`FullAttentionSpec.merge()` 断言（`kv_cache_interface.py:313-318`）硬性禁止同组，形成 5 层异构小桶，把 `group_size` 从 17 拖到 5。
- 关键补充：draft 与 base 的 `page_size_bytes` 完全相同（`1536×2×256×2 = 1536×1×512×2 = 1,572,864`），只是 K/V head 布局不同。因此"单独成组"不引入新的 page_size 不对齐。
- `build_attn_metadata`（`vllm_ascend/worker/v2/attn_utils.py:206`）外层按 `kv_cache_groups` 循环，组数 15→5 即构建单元 15→5。

---

## 3. 修改目标与非目标

### 3.1 目标

1. DSpark `[16, 5, 48]`：15 组 → **5 组**，`build_attn_metadata` 外层循环 15 → **5**。
2. MTP `[17, 48]`：保持 4 组，行为零变化。
3. 非 DSpark、非 general-hybrid 模型：完全沿用现有 heuristic。
4. 每个真实 KV 层恰好出现一次；组内所有层 spec 一致；不伪造 spec。
5. scheduler / worker / KV allocator 使用**同一份** group 列表。
6. 真实权重推理、DSpark acceptance、ACLGraph 不回归。

### 3.2 非目标

- 不把 draft `2×128` 伪造成 base `1×256`（`merge` 断言禁止，且会破坏 K/V 计算布局）。
- 不改模型结构 / 权重 / head 拓扑。
- 不按层名字符串（`model.layers.64`）识别 draft。
- 不设全局 `MIN_GROUP_SIZE`。
- 不默认改变所有 hybrid 模型分组策略。
- 不新增环境变量（若需要开关，按 `vllm_ascend/envs.py` 评审要求另行设计）。
- 不在热路径引入 `tensor.item()` / CPU-NPU 同步 / 每步 INFO。

---

## 4. 方案选择

### 4.1 选定方案：DSpark 场景下的 padding-budget group_size 搜索

在 DSpark（`use_dspark()`）的 general uniform-page-size 分支，枚举候选 `group_size`，在"总 padding 比例 ≤ 上限"的候选中，按以下优先级选：

1. group 数最少；
2. group 数相同则 padding 最少；
3. 仍相同则 `group_size` 更大（保证确定性）。

上限命名常量 `DSPARK_KV_GROUP_MAX_PADDING_RATIO = 0.20`，仅作用于 `use_dspark()` 分支。

### 4.2 对 `[16, 5, 48]` 的候选结果

| group_size | group 数 | 等效槽位 | padding 层 | padding 比例 | 说明 |
|---:|---:|---:|---:|---:|---|
| 5 | 15 | 75 | 6 | 8.70% | 当前实现 |
| 8 | 9 | 72 | 3 | 4.35% | 容量优先备选 |
| 10 | 8 | 80 | 11 | 15.94% | 被 16 支配 |
| 16 | **5** | 80 | 11 | 15.94% | **选定** |
| 24 | 4 | 96 | 27 | 39.13% | 超出预算 |

选定 `group_size=16`：base 1 组、draft 1 组、mamba 3 组，共 **5 组**，实际层数 `[16, 5, 16, 16, 16]`。

### 4.3 容量代价（约 6.25%，须实测）

等效槽位 75 → 80，理论上 KV 容量约为当前的 `75/80 = 93.75%`。日志当前 `GPU KV cache size ≈ 186,413 tokens`，粗估修改后约 `174,762 tokens`。**该数字只用于定验收阈值，最终以启动日志实际 `GPU KV cache size` / `Maximum concurrency` 为准。**

### 4.4 为何不选其他方案

- **统一 FullAttentionSpec**：`merge` 断言 + 计算布局错误，不可行。
- **仅在 Ascend worker 合并 metadata**：不同 group 有独立 block table / slot mapping，私自合并与 scheduler 分配不一致，不可行。
- **全局 `MIN_GROUP_SIZE`**：影响所有 hybrid 模型，不可控。
- **按 draft 层名摘出再追加**：结果同为 5 组，但需可靠 draft 身份；虽然 `Qwen3DSparkForCausalLM.get_draft_kv_cache_layer_names()`（上游 `qwen3_dspark.py:129`）能提供，但引入了对模型结构的依赖，比纯 `layer_counts` 搜索更脆。**搜索方案不依赖层名，更通用、改动更小。**

---

## 5. 第一闸口：改动落点决策（必须先定）

`get_kv_cache_groups` 是 core 函数，插件侧无干净的接入点（monkey-patch core 函数是长期隐患，codex 计划已正确否决）。因此本方案**必须先决策**：

- **首选：向上游 vllm-project/vllm 提交 core 改动。** 改动是通用、无副作用的（新增可选参数 + DSpark gating），适合上游化。需遵守上游 AGENTS.md：先做重复 PR 检查（`gh issue/pr search`）、补齐单测、人类提交者逐行 review。
- **过渡：在固定 vLLM fork 中携带同一 patch。** 在 vllm-ascend 镜像锁定 fork 提交，直到上游合入。

两选一必须在本方案"阶段 B"开始前由提交者确认，并写入最终报告（明确 upstream commit 与 vllm-ascend 的依赖关系）。

---

## 6. 代码设计（core 侧，`vllm/v1/core/kv_cache_utils.py`）

### 6.1 命名常量

```python
DSPARK_KV_GROUP_MAX_PADDING_RATIO = 0.20
```

### 6.2 抽取默认 heuristic（保持行为完全一致）

```python
def _get_default_kv_group_size(layer_counts: Sequence[int]) -> int:
    min_num_layers = min(layer_counts)
    max_num_layers = max(layer_counts)
    if max_num_layers < min_num_layers * 1.5:
        return max_num_layers
    return min_num_layers
```

注意：不得改动 `<` 为 `<=`，不得动 1.5 系数。

### 6.3 候选评价

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

`padding_ratio` 是概念 padding（排序代理），非 stride 拆分的精确 padding。

### 6.4 DSpark 搜索

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

边界要求：`layer_counts` 非空且全为正；默认 heuristic 永远是合法 fallback；结果确定（不依赖 dict hash 顺序）。

### 6.5 分组函数增加可选参数

```python
def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
    *,
    optimize_for_fewer_groups: bool = False,
) -> list[KVCacheGroupSpec]:
    ...
    layer_counts = [len(layers) for layers in same_type_layers.values()]
    if optimize_for_fewer_groups:
        group_size = _select_dspark_kv_group_size(layer_counts)
    else:
        group_size = _get_default_kv_group_size(layer_counts)
    # 后续 stride 拆分保持原样，兼容 PP 均衡语义
```

### 6.6 入口分派（只放 general 分支）

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

保持现状的提前返回路径：attention-free、uniform spec、`UniformTypeKVCacheSpecs`、DSV4 `group_and_unify_kv_cache_specs`。

### 6.7 日志

初始化期一条汇总 INFO（不逐层、不在热路径）：

```text
DSpark KV grouping selected group_size=16 for layer_counts=[16, 5, 48],
groups=5, padding_layers=11, padding_ratio=15.94%
```

`layer_counts` 排序后输出。非 DSpark 不打。commit `5d69f096` 的 `[DIAG-BUILD-META-*]` 位于热路径，验证完成后删除或降级。

---

## 7. 测试

### 7.1 core 单测（`tests/v1/core/test_kv_cache_utils.py`）

搜索函数参数化：

| layer_counts | 模式 | 预期 group_size | 预期组数 | 目的 |
|---|---|---:|---:|---|
| `[16, 5, 48]` | DSpark | 16 | 5 | 本问题回归 |
| `[17, 48]` | 默认 | 17 | 4 | MTP 不变 |
| `[12, 13]` | 默认 | 13 | 2 | 1.5 heuristic 不变 |
| `[10, 20]` | 默认 | 10 | 3 | 边界不变 |
| `[1]` | DSpark | 1 | 1 | 单桶 |
| `[5, 5, 5]` | DSpark | 5 | 3 | 无需放大 |

额外断言：padding ratio 不超预算（除非 fallback 本身超）；输入顺序不影响结果；空/非正输入抛 `ValueError`。

真实 spec 分组：构造 16 个 base（1×256）、5 个 draft（2×128）、48 个 Mamba，先统一 page_size，断言 `len(groups)==5`、`sorted(len(g.layer_names)) == [5,16,16,16,16]`、69 层名恰好一次、无重复、组内 spec 兼容、Mamba 每组 16 层、非 DSpark 仍 15 组。

### 7.2 KV 配置构造测试

对 `[16,5,16,16,16]` 调 `get_kv_cache_config_from_groups`，断言：`len(kv_cache_tensors)==16`；slot 5..15 不含 draft 层；`num_blocks` 用 `group_size=16`；无空 `shared_by` tensor。

### 7.3 Ascend metadata 测试

构造 5 组 mock，断言 `block_tables`/`slot_mappings`/`attn_groups` 外层长度均为 5，返回字典覆盖 69 层名，不同 group 不共享带不同 block table 的 metadata。

---

## 8. 验证

### 8.1 头号风险：scheduler/worker 一致性

启动即断言：scheduler 侧与 worker 侧的 `len(kv_cache_groups)` 一致（都是 5）。**任何不一致立即视为失败**，根因通常是某侧 `speculative_config` 为空导致 gating 分叉。

### 8.2 复现配置（与日志一致）

```bash
HCCL_OP_EXPANSION_MODE=AIV vllm serve /opt/foundation_model/Qwen3.6-27B \
  --max-model-len 16384 --max-num-seqs 4 --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 --trust-remote-code --async-scheduling \
  --speculative-config '{"num_speculative_tokens":7,"method":"dspark","model":"/opt/foundation_model/Qwen3.6-27B-Dspark-0810-64000"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --port 8000
```

### 8.3 必须执行真实请求

`/v1/chat/completions` 返回 200 且输出非空、EngineCore 不退出；`Application startup complete` 不能单独作为成功依据。默认验收 FULL_DECODE_ONLY ACLGraph 路径，保留 replay 证据；图模式失败时用 `--enforce-eager` 隔离定位，但 eager 成功不能替代图模式验收。

### 8.4 容量扩展

复现通过后依次：16k/bs4 → 保持 16k 增 bs → 尝试 128k+bs16 基线；不可行时记录内存/TP/KV 容量/模型限制，不得只写"未测试"。

---

## 9. 验收标准

正确性硬门槛：spec 桶仍 `[16,5,48]`；`kv_cache_groups==5`；层数 `[16,5,16,16,16]`；`attn_groups`/`block_tables`/`slot_mappings` 外层一致；69 层名恰好一次；真实请求正确；acceptance 不异常归零；ACLGraph 正常；MTP 仍 4 组；非 DSpark 单测全过。

性能门槛：构建单元 15→5；稳态 `build_attn_metadata` p50 目标下降 ≥50%；p90/p99 无新长尾；TPOT/吞吐退化 ≤3%；容量损失 ≤10%；无热路径 INFO / 新增同步。

容量门槛：相对当前基线 `GPU KV cache size ≥ 167,771 tokens`（当前 186,413 的 90%），该值仅是相对回归界限，非跨机器固定值。

---

## 10. 回退策略

- 首选 `group_size=16`（5 组）。
- 容量损失超 10% 时用 `group_size=8`（9 组，等效槽位 72，理论上不损容量）。
- 出现输出变化 / KV shape 错误 / block table 越界 / ACLGraph 失败 / acceptance 退化 / 容量超阈值 / 非 DSpark 分组变化 → 回滚 group-size 选择逻辑，但保留无副作用的单测与根因文档。

---

## 11. 实施步骤

- **阶段 A**：确认 runtime import 路径；真实权重复现 15 组；记录 KV 容量、acceptance、metadata p50/p90/p99、TTFT/TPOT/吞吐基线。
- **阶段 B**：**先决策 core 改动落点（上游 vs fork）**；抽取默认 heuristic；新增评价 + 搜索 + 可选参数；`use_dspark()` gating；一条汇总日志。
- **阶段 C**：补 core 单测 + Ascend metadata 单测；跑 `ruff` + `bash format.sh ci`。
- **阶段 D**：真实权重启动；首请求 smoke；scheduler/worker 一致性断言；ACLGraph；稳态性能 + 容量采集。
- **阶段 E**：删除 `[DIAG-*]` 逐步/逐层日志；保留一条汇总日志；更新证据文档；signed-off Conventional Commit；报告 upstream/fork commit 关系。

---

## 12. 预期最终结果

```text
spec buckets: [16, 5, 48]
DSpark KV grouping selected group_size=16, padding_layers=11, padding_ratio=15.94%
get_kv_cache_groups final result: 5 kv_cache_groups
build_attn_metadata: 5 kv_cache_groups, 5 total attn_groups
```

最终判断标准不是"打印 5 groups"，而是同时满足：

```text
分组正确 + KV 地址正确 + 真实推理正确 + metadata 明显加速 + 容量损失受控
```
