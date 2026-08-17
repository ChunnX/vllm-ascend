# Qwen3.6 DSpark v2 路径 kv_cache_groups 过多问题分析

## 1. 问题描述

在 Qwen3.6 模型上：

| 路径 | kv_cache_groups 数量 | build_meta 调用次数 | 效率 |
|------|----------------------|---------------------|------|
| v1 MTP | 4 | 4 | 高 |
| v2 DSpark | 15 | 15 | 低 |

v2 DSpark 路径下 `attn_utils.py:177` 处迭代 15 个 kv_cache_groups，每个 group 独立执行 build meta，效率远低于 v1 MTP 路径的 4 次。预期非 DSpark 部分的分组应与 MTP 一致，但实际上被过度拆分。

---

## 2. 关键代码路径

### 2.1 kv_cache_groups 的生成

```
kv_cache_spec (dict[str, KVCacheSpec])
  → get_kv_cache_groups()                          # kv_cache_utils.py:1611
    → is_kv_cache_spec_uniform?                    # No (FullAttentionSpec + MambaSpec)
    → UniformTypeKVCacheSpecs.from_specs?          # No (不同类型)
    → group_and_unify_kv_cache_specs?              # No (非 DeepSeekV4)
    → _get_kv_cache_groups_uniform_page_size()     # ← 关键函数 (kv_cache_utils.py:1052)
```

### 2.2 v1 build_meta 的调用

文件: `vllm_ascend/worker/model_runner_v1.py:2991`

```python
for kv_cache_gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
    for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
        _build_attn_group_metadata(kv_cache_gid, attn_gid, ...)
```

### 2.3 v2 build_meta 的调用

文件: `vllm_ascend/worker/v2/attn_utils.py:177`

```python
kv_cache_groups = kv_cache_config.kv_cache_groups
for i, kv_cache_spec in enumerate(kv_cache_groups):
    # 每个 kv_cache_group 独立构建 common_attn_metadata
    for attn_group in attn_groups[i]:
        # 每个 attn_group 独立构建 attn_metadata
```

---

## 3. 根因分析

### 3.1 `_get_kv_cache_groups_uniform_page_size` 分组算法

文件: `vllm/v1/core/kv_cache_utils.py:1052-1171`

算法步骤：

1. **按 KVCacheSpec 分桶**：用 dict 的 `__hash__`/`__eq__` 将所有层按 spec 归类
   ```python
   same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
   for layer_name, layer_spec in kv_cache_spec.items():
       same_type_layers[layer_spec].append(layer_name)
   ```

2. **计算 group_size**：
   ```python
   min_num_layers = min([len(layers) for layers in same_type_layers.values()])
   group_size = min_num_layers
   max_num_layers = max([len(layers) for layers in same_type_layers.values()])
   if max_num_layers < min_num_layers * 1.5:
       group_size = max_num_layers
   ```

3. **按 group_size stride 切分**：
   ```python
   num_groups = cdiv(len(layers), group_size)
   for i in range(num_groups):
       grouped_layers.append(layers[i::num_groups])  # stride 切分
   ```

### 3.2 两条路径的 kv_cache_spec 对比

#### v1 (MTP) 路径

`get_kv_cache_spec` 文件: `vllm_ascend/worker/model_runner_v1.py:4516`

| 类型 | 层名 | 数量 | KVCacheSpec |
|------|------|------|-------------|
| base self_attn | language_model.model.layers.{3,7,...,63}.self_attn.attn | 16 | FullAttentionSpec (num_kv_heads=N, head_size=H) |
| base linear_attn | language_model.model.layers.{0,4,...,60}.linear_attn | 48 | MambaSpec |
| MTP self_attn | mtp.layers.0.self_attn.attn | 1 | FullAttentionSpec (**与 base 相同**) |

MTP 层使用 base 模型最后一层的同一套注意力参数（相同 `num_kv_heads`、`head_size`），因此 `FullAttentionSpec` 的 `__hash__`/`__eq__` 判定与 base 层相同，归入同一桶。

**类型桶**：A=17 个 FullAttentionSpec, B=48 个 MambaSpec

**分组计算**：
- `min_num_layers=17`, `max_num_layers=48`
- `48 > 17 * 1.5 = 25.5` → `group_size = 17`
- A(17层): `cdiv(17,17)` = 1 组
- B(48层): `cdiv(48,17)` = 3 组
- **总计 = 4 个 kv_cache_groups** ✓

#### v2 (DSpark) 路径

`get_kv_cache_spec` 文件: `vllm_ascend/worker/v2/attn_utils.py:56`

| 类型 | 层名 | 数量 | KVCacheSpec |
|------|------|------|-------------|
| base self_attn | language_model.model.layers.{3,7,...,63}.self_attn.attn | 16 | FullAttentionSpec (num_kv_heads=N₁, head_size=H₁) |
| base linear_attn | language_model.model.layers.{0,1,2,4,5,...}.linear_attn | 48 | MambaSpec |
| DSpark draft self_attn | model.layers.{64,...,68}.self_attn.attn | 5 | FullAttentionSpec (**num_kv_heads=N₂, head_size=H₂**) |

DSpark draft 模型（如 `dspark_qwen3_8b_block7`）是独立的小模型，其 `num_key_value_heads`、`head_dim` 等参数与 base Qwen3.6 模型不同，因此产生的 `FullAttentionSpec` 在 `__hash__`/`__eq__` 上与 base 不同，归入独立桶。

**类型桶**：A=16 个 FullAttentionSpec, B=48 个 MambaSpec, **C=5 个 FullAttentionSpec (DSpark, 异构)**

**分组计算**：
- `min_num_layers=5`, `max_num_layers=48`
- `48 > 5 * 1.5 = 7.5` → `group_size = 5`
- A(16层): `cdiv(16,5)` = 4 组（stride 切分，每组 4 层）
- B(48层): `cdiv(48,5)` = 10 组（stride 切分，每组约 5 层）
- C(5层): `cdiv(5,5)` = 1 组
- **总计 = 15 个 kv_cache_groups** ✓

### 3.3 根因总结

```
DSpark draft 层产生异构 FullAttentionSpec
  → _get_kv_cache_groups_uniform_page_size 中出现第三个 spec 桶（仅 5 层）
    → group_size = min_num_layers = 5（被最小的桶拖垮）
      → 48 个 MambaSpec 层被拆成 10 组
      → 16 个 base FullAttentionSpec 层被拆成 4 组
        → 总共 15 个 kv_cache_groups，15 次 build_meta
```

对比 v1 (MTP)：MTP 层与 base 共享同一 FullAttentionSpec，只有 2 个桶，`group_size=17`，仅 4 个 kv_cache_groups。

---

## 4. 实际数据验证

### 4.1 v1 (MTP) 的 attn_group 示例

Group 00 — self_attn.attn (17 层，base 16 + MTP 1):
```
language_model.model.layers.3.self_attn.attn
language_model.model.layers.7.self_attn.attn
...
language_model.model.layers.63.self_attn.attn
mtp.layers.0.self_attn.attn
```

Group 01 — linear_attn (16 层):
```
language_model.model.layers.0.linear_attn
language_model.model.layers.4.linear_attn
...
language_model.model.layers.60.linear_attn
```

### 4.2 v2 (DSpark) 的 kv_cache_group 示例

Group A-0 — self_attn.attn (4 层，stride 切分):
```
language_model.model.layers.3.self_attn.attn
language_model.model.layers.19.self_attn.attn
language_model.model.layers.35.self_attn.attn
language_model.model.layers.51.self_attn.attn
```

Group B-3 — linear_attn (4 层，stride 切分):
```
language_model.model.layers.12.linear_attn
language_model.model.layers.25.linear_attn
language_model.model.layers.38.linear_attn
language_model.model.layers.52.linear_attn
```

Group C-0 — DSpark bidirectional (5 层):
```
model.layers.64.self_attn.attn
model.layers.65.self_attn.attn
model.layers.66.self_attn.attn
model.layers.67.self_attn.attn
model.layers.68.self_attn.attn
```

---

## 5. kv_cache_spec 的来源差异

### 5.1 v1 get_kv_cache_spec

文件: `vllm_ascend/worker/model_runner_v1.py:4516`

```python
attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)

for layer_name, attn_module in attn_layers.items():
    if (isinstance(attn_module, Attention)
            and (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None):
        self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
        continue  # Attention 层的 KV sharing 检查
    elif isinstance(attn_module, Attention):
        kv_cache_spec[layer_name] = spec
    elif isinstance(attn_module, MambaBase):
        mamba_layers[layer_name] = attn_module  # 延后处理

# 延后处理 MambaSpec，对齐 page_size
if len(mamba_layers) > 0:
    for layer_name, mamba_module in mamba_layers.items():
        kv_cache_spec[layer_name] = spec
    # 对齐 attn page_size 到 mamba page_size
```

### 5.2 v2 get_kv_cache_spec

文件: `vllm_ascend/worker/v2/attn_utils.py:56`

```python
attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)

for layer_name, attn_module in attn_layers.items():
    if getattr(attn_module, "kv_sharing_target_layer_name", None):
        continue  # 用 getattr 检查所有层类型（MambaBase 无此属性，不会触发）

    spec = attn_module.get_kv_cache_spec(vllm_config)
    if isinstance(spec, MambaSpec):
        mamba_specs[layer_name] = spec  # 分离处理
        continue

    kv_cache_spec[layer_name] = spec

# 对齐 page_size 并设置 indexes_kv_by_block_stride
if mamba_specs:
    common_page_size = max(...)
    for layer_name in attention_layer_names:
        kv_cache_spec[layer_name] = replace(spec, page_size_padded=..., indexes_kv_by_block_stride=True)
    kv_cache_spec.update(mamba_specs)
```

**关键差异**：v2 的 `replace(spec, indexes_kv_by_block_stride=True)` 会修改所有 AttentionSpec，但所有同类型层都被一致修改，不影响分组。两者找到的层数量相同（都使用 `AttentionLayerBase`），48 个 MambaSpec + 16 个 base FullAttentionSpec + DSpark 层在两条路径中一致。

---

## 6. 解决方向

### 方案 A: 修改分组算法，排除 draft 层对 group_size 的影响

在 `_get_kv_cache_groups_uniform_page_size` 中，识别 DSpark draft 层并将其排除在 `min_num_layers` 计算之外：

```python
# 伪代码
draft_layer_names = set(model.get_draft_kv_cache_layer_names())
non_draft_types = {spec: layers for spec, layers in same_type_layers.items()
                   if not any(l in draft_layer_names for l in layers)}
min_num_layers = min([len(layers) for layers in non_draft_types.values()])
group_size = min_num_layers
```

优点：非 draft 部分的分组与 MTP 路径一致。
缺点：需要将 draft 层名信息传递到分组逻辑中，改动范围较大。

### 方案 B: 将 DSpark draft 层单独成组，不参与 group_size 计算

先对非 draft 层执行 `_get_kv_cache_groups_uniform_page_size`，然后将 draft 层追加为独立的 kv_cache_group：

```python
# 伪代码
non_draft_spec = {k: v for k, v in kv_cache_spec.items() if k not in draft_layer_names}
kv_cache_groups = _get_kv_cache_groups_uniform_page_size(non_draft_spec)
# draft 层各自成组
for draft_layer_name in draft_layer_names:
    kv_cache_groups.append(KVCacheGroupSpec([draft_layer_name], kv_cache_spec[draft_layer_name]))
```

优点：改动集中，非 draft 部分完全复用原有逻辑。
缺点：draft 层不参与 stride 切分，可能影响 PP 场景下的均衡性。

### 方案 C: 设置 group_size 下限

在 `_get_kv_cache_groups_uniform_page_size` 中设置 group_size 的最小值，防止小桶拖垮大桶：

```python
# 伪代码
MIN_GROUP_SIZE = 8  # 或其他合理值
group_size = max(min_num_layers, MIN_GROUP_SIZE)
```

优点：改动最小，一行代码。
缺点：需要确定合理的下限值，可能对其他模型产生副作用；draft 层仍然与非 draft 层混合分组。

### 方案 D: Draft 层使用与 base 相同的 FullAttentionSpec（unify spec）

在 `get_kv_cache_spec` 阶段，将 DSpark draft 层的 FullAttentionSpec 替换为与 base 一致的 spec，使 draft 层归入 base 的 spec 桶：

```python
# 伪代码 - 在 get_kv_cache_spec 中
base_full_attn_spec = next(spec for name, spec in kv_cache_spec.items()
                           if name in base_attn_layer_names and isinstance(spec, FullAttentionSpec))
for draft_name in draft_layer_names:
    kv_cache_spec[draft_name] = base_full_attn_spec
```

优点：分组逻辑无需修改，draft 层自然归入 base 桶。
缺点：draft 层的 KV cache 实际参数与 base 不同，强行统一 spec 可能导致 KV cache 内存分配错误。**此方案不可行**。

### 推荐方案

**方案 B** 最合理：将 DSpark draft 层从 `_get_kv_cache_groups_uniform_page_size` 的输入中分离出来，非 draft 部分独立分组后再将 draft 层追加为独立 kv_cache_group。这样非 draft 部分的分组结果与 MTP 路径一致（4 个 kv_cache_groups），draft 层各自独立成组（1-5 个），总计 5-9 个 kv_cache_groups，远优于当前的 15 个。

---

## 7. 相关文件索引

| 文件 | 行号 | 作用 |
|------|------|------|
| `vllm/v1/core/kv_cache_utils.py` | 1052-1171 | `_get_kv_cache_groups_uniform_page_size` 分组算法 |
| `vllm/v1/core/kv_cache_utils.py` | 1611-1659 | `get_kv_cache_groups` 入口分发 |
| `vllm_ascend/worker/model_runner_v1.py` | 4516-4654 | v1 `get_kv_cache_spec` |
| `vllm_ascend/worker/v2/attn_utils.py` | 56-118 | v2 `get_kv_cache_spec` |
| `vllm_ascend/worker/v2/attn_utils.py` | 176-242 | v2 `build_attn_metadata` 迭代 kv_cache_groups |
| `vllm_ascend/worker/model_runner_v1.py` | 2915-3037 | v1 `_build_attn_group_metadata` |
| `vllm/model_executor/models/qwen3_dspark.py` | - | DSpark 模型定义 |
| `vllm/model_executor/models/qwen3_dflash.py` | 72-132 | `_resolve_layer_attention` draft 层 spec 决定逻辑 |
| `vllm/model_executor/layers/mamba/abstract.py` | 16 | `MambaBase(AttentionLayerBase)` — GDN 继承关系 |
| `vllm/model_executor/layers/attention/attention.py` | 621-693 | `Attention.get_kv_cache_spec` — 返回 FullAttentionSpec |

---

## 8. 诊断日志说明

为验证上述根因分析，已在以下位置添加 `[DIAG]` 前缀的诊断日志。运行 Qwen3.6 后，可通过 `grep "\[DIAG"` 过滤日志。

### 8.1 日志标签体系

| 标签 | 文件 | 含义 |
|------|------|------|
| `[DIAG-KV-SPEC-V1]` | `vllm_ascend/worker/model_runner_v1.py` | v1 路径 `get_kv_cache_spec` 输出的逐层 spec 信息 |
| `[DIAG-KV-SPEC-V2]` | `vllm_ascend/worker/v2/attn_utils.py` | v2 路径 `get_kv_cache_spec` 输出的逐层 spec 信息 |
| `[DIAG-KV-GROUP]` | `vllm/v1/core/kv_cache_utils.py` | 分组算法的决策过程 |
| `[DIAG-BUILD-META-V1]` | `vllm_ascend/worker/model_runner_v1.py` | v1 路径 build_meta 阶段的 kv_cache_group 迭代信息 |
| `[DIAG-BUILD-META-V2]` | `vllm_ascend/worker/v2/attn_utils.py` | v2 路径 build_meta 阶段的 kv_cache_group 迭代信息 |

### 8.2 预期日志验证逻辑

#### 步骤 1：验证 DSpark draft 层与 base 层的 FullAttentionSpec 不同

查看 `[DIAG-KV-SPEC-V2]` 中 FullAttentionSpec 层的日志，预期：

```
[DIAG-KV-SPEC-V2] FullAttentionSpec layer: language_model.model.layers.3.self_attn.attn,
    num_kv_heads=X, head_size=Y, ..., spec_hash=H1
...
[DIAG-KV-SPEC-V2] FullAttentionSpec layer: model.layers.64.self_attn.attn,
    num_kv_heads=A, head_size=B, ..., spec_hash=H2   ← 不同！
```

关键证据：base 层的 `num_kv_heads`/`head_size` 与 DSpark draft 层不同，`spec_hash` 也不同。

#### 步骤 2：验证 spec 桶的数量和大小

查看 `[DIAG-KV-GROUP]` 中 `_get_kv_cache_groups_uniform_page_size` 的日志，预期：

```
[DIAG-KV-GROUP] _get_kv_cache_groups_uniform_page_size: 3 spec buckets
[DIAG-KV-GROUP]   bucket[0]: type=FullAttentionSpec, num_layers=16, spec_hash=H1, ...
[DIAG-KV-GROUP]   bucket[1]: type=FullAttentionSpec, num_layers=5, spec_hash=H2, ...  ← DSpark 异构桶
[DIAG-KV-GROUP]   bucket[2]: type=MambaSpec, num_layers=48, spec_hash=H3, ...
```

关键证据：3 个桶，DSpark 桶仅有 5 层。

#### 步骤 3：验证 group_size 被拖低至 5

查看 `[DIAG-KV-GROUP]` 中 group_size 决策的日志，预期：

```
[DIAG-KV-GROUP] group_size decision: min_num_layers=5, max_num_layers=48,
    max/min_ratio=9.60, threshold=1.5, group_size=5
```

关键证据：`min_num_layers=5`（来自 DSpark 桶），导致 `group_size=5`。

对比 v1 (MTP) 路径：

```
[DIAG-KV-GROUP] group_size decision: min_num_layers=17, max_num_layers=48,
    max/min_ratio=2.82, threshold=1.5, group_size=17
```

关键证据：v1 中 MTP 层与 base 共享 spec，最小桶 17 层，`group_size=17`。

#### 步骤 4：验证分组结果

查看 `[DIAG-KV-GROUP]` 中最终分组结果，预期：

v2 (DSpark): 15 个 kv_cache_groups
v1 (MTP): 4 个 kv_cache_groups

#### 步骤 5：验证 build_meta 调用次数

查看 `[DIAG-BUILD-META-V2]` 和 `[DIAG-BUILD-META-V1]`，预期：

```
[DIAG-BUILD-META-V2] build_attn_metadata: 15 kv_cache_groups, 15 total attn_groups
[DIAG-BUILD-META-V1] _build_attention_metadata: 4 kv_cache_groups, 4 total attn_groups
```

### 8.3 日志修改清单

| 文件 | 修改位置 | 新增日志 |
|------|----------|----------|
| `vllm/v1/core/kv_cache_utils.py` | `get_kv_cache_groups` 入口 | 逐层 spec identity 摘要 |
| `vllm/v1/core/kv_cache_utils.py` | `get_kv_cache_groups` 路径分发 | 哪条路径被选中 |
| `vllm/v1/core/kv_cache_utils.py` | `unify_kv_cache_spec_page_size` | page_size 分布和对齐操作 |
| `vllm/v1/core/kv_cache_utils.py` | `_get_kv_cache_groups_uniform_page_size` | spec 桶详情、group_size 决策、分组结果 |
| `vllm/v1/core/kv_cache_utils.py` | `get_kv_cache_groups` 最终返回 | 最终 kv_cache_group 列表 |
| `vllm_ascend/worker/model_runner_v1.py` | `get_kv_cache_spec` | 逐层 FullAttentionSpec/SlidingWindowSpec/MambaSpec 字段 |
| `vllm_ascend/worker/model_runner_v1.py` | `get_kv_cache_spec` 返回前 | spec hash 桶汇总 |
| `vllm_ascend/worker/model_runner_v1.py` | `_build_attention_metadata` 外层循环 | kv_cache_group/attn_group 迭代信息 |
| `vllm_ascend/worker/v2/attn_utils.py` | `get_kv_cache_spec` | 逐层 spec 字段 + kv_sharing 跳过 + page_size 对齐 |
| `vllm_ascend/worker/v2/attn_utils.py` | `get_kv_cache_spec` 返回前 | spec hash 桶汇总 |
| `vllm_ascend/worker/v2/attn_utils.py` | `build_attn_metadata` | kv_cache_group 迭代信息 |
