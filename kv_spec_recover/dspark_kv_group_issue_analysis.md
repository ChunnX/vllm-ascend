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
  → get_kv_cache_groups()                          # kv_cache_utils.py:1728
    → is_kv_cache_spec_uniform?                    # No (FullAttentionSpec + MambaSpec)
    → UniformTypeKVCacheSpecs.from_specs?          # No (不同类型)
    → group_and_unify_kv_cache_specs?              # No (非 DeepSeekV4)
    → _get_kv_cache_groups_uniform_page_size()     # ← 关键函数 (kv_cache_utils.py:1137)
```

两条路径都走到 `_get_kv_cache_groups_uniform_page_size`，使用同一套分组算法。

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

文件: `vllm/v1/core/kv_cache_utils.py:1137-1256`

**核心约束**：所有 kv_cache_group 的单个 block 物理内存大小必须相同（避免内存碎片）。因为 block 大小 = `组内层数 × 每层 page_size`，且 page_size 已统一，所以 **所有组的层数必须相同**。

算法步骤：

1. **按 KVCacheSpec 分桶**：用 `frozen dataclass` 的 `__hash__`/`__eq__` 将所有层按 spec 归类
   ```python
   same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
   for layer_name, layer_spec in kv_cache_spec.items():
       same_type_layers[layer_spec].append(layer_name)
   ```

2. **计算 group_size**（每组放多少层）：
   ```python
   min_num_layers = min([len(layers) for layers in same_type_layers.values()])
   group_size = min_num_layers
   max_num_layers = max([len(layers) for layers in same_type_layers.values()])
   if max_num_layers < min_num_layers * 1.5:
       group_size = max_num_layers
   ```
   取 `min_num_layers` 是为了让最小的桶能恰好整除；1.5 倍阈值是为了在桶大小接近时避免过多 padding。

3. **按 group_size stride 切分**：
   ```python
   num_groups = cdiv(len(layers), group_size)
   for i in range(num_groups):
       grouped_layers.append(layers[i::num_groups])  # stride 切分（非连续切分，为 PP 均衡）
   ```

### 3.2 两条路径的 kv_cache_spec 对比（日志实测数据）

#### v1 (MTP) 路径 — 65 层，2 个 spec 桶

`get_kv_cache_spec` 文件: `vllm_ascend/worker/model_runner_v1.py:4516`

| 类型 | 层名 | 数量 | KVCacheSpec 关键字段 |
|------|------|------|---------------------|
| base self_attn | language_model.model.layers.{3,7,...,63}.self_attn.attn | 16 | FullAttentionSpec(num_kv_heads=**1**, head_size=**256**) |
| base linear_attn | language_model.model.layers.{0,1,2,4,5,6,...}.linear_attn | 48 | MambaSpec(GDN_ATTN) |
| MTP self_attn | mtp.layers.0.self_attn.attn | 1 | FullAttentionSpec(num_kv_heads=**1**, head_size=**256**) ← 与 base 相同 |

MTP 层使用 base 模型同一套注意力参数（`num_kv_heads=1, head_size=256`），因此 `FullAttentionSpec` 的 `__hash__`/`__eq__` 判定与 base 层相同，**归入同一桶**。

**类型桶**：A=17 个 FullAttentionSpec (16 base + 1 MTP), B=48 个 MambaSpec

**分组计算（日志实测）**：
- `min_num_layers=17`, `max_num_layers=48`, `max/min_ratio=2.82`
- `2.82 > 1.5` → `group_size = 17`
- A(17层): `cdiv(17,17)` = 1 组
- B(48层): `cdiv(48,17)` = 3 组
- **总计 = 4 个 kv_cache_groups** ✓

#### v2 (DSpark) 路径 — 69 层，3 个 spec 桶

`get_kv_cache_spec` 文件: `vllm_ascend/worker/v2/attn_utils.py:56`

| 类型 | 层名 | 数量 | KVCacheSpec 关键字段 |
|------|------|------|---------------------|
| base self_attn | language_model.model.layers.{3,7,...,63}.self_attn.attn | 16 | FullAttentionSpec(num_kv_heads=**1**, head_size=**256**) |
| base linear_attn | language_model.model.layers.{0,1,2,4,5,6,...}.linear_attn | 48 | MambaSpec(GDN_ATTN) |
| DSpark draft self_attn | model.layers.{64,65,66,67,68}.self_attn.attn | 5 | FullAttentionSpec(num_kv_heads=**2**, head_size=**128**) ← 与 base 不同 |

DSpark draft 模型（如 `dspark_qwen3_8b`）是独立的小模型，其 `num_key_value_heads=2, head_dim=128` 与 base Qwen3.6 的 `num_key_value_heads=1, head_dim=256` 不同，因此产生的 `FullAttentionSpec` 在 `__hash__`/`__eq__` 上与 base 不同，**归入独立桶**。

**类型桶**：A=16 个 FullAttentionSpec (base), B=48 个 MambaSpec, **C=5 个 FullAttentionSpec (DSpark, 异构)**

**分组计算（日志实测）**：
- `min_num_layers=5`, `max_num_layers=48`, `max/min_ratio=9.60`
- `9.60 > 1.5` → `group_size = 5`
- A(16层): `cdiv(16,5)` = 4 组（stride 切分，每组 4 层 + 1 padding）
- B(48层): `cdiv(48,5)` = 10 组（stride 切分，8 组 5 层 + 2 组 4 层）
- C(5层): `cdiv(5,5)` = 1 组
- **总计 = 15 个 kv_cache_groups** ✓

### 3.3 根因总结

```
DSpark draft 层产生异构 FullAttentionSpec (num_kv_heads=2, head_size=128)
  vs base 层 FullAttentionSpec (num_kv_heads=1, head_size=256)
    → _get_kv_cache_groups_uniform_page_size 中出现第三个 spec 桶（仅 5 层）
      → group_size = min_num_layers = 5（被最小的桶拖垮）
        → 48 个 MambaSpec 层被拆成 10 组
        → 16 个 base FullAttentionSpec 层被拆成 4 组
          → 总共 15 个 kv_cache_groups，15 次 build_meta
```

对比 v1 (MTP)：MTP 层与 base 共享同一 FullAttentionSpec (num_kv_heads=1, head_size=256)，只有 2 个桶，`group_size=17`，仅 4 个 kv_cache_groups。

### 3.4 对比图示

```
                  MTP (v1)                            DSpark (v2)
               ┌───────────────┐                  ┌───────────────┐
  spec 桶      │ A: 17层 Full  │                  │ A: 16层 Full  │
               │   (kv=1,h=256)│                  │   (kv=1,h=256)│
               │ B: 48层 Mamba │                  │ B:  5层 Full  │ ← 异构小桶
               └───────┬───────┘                  │   (kv=2,h=128)│
                       │                          │ C: 48层 Mamba │
                       │                          └───────┬───────┘
             min=17, group_size=17               min=5, group_size=5
                       │                                  │
                ┌──────┴───────┐                    ┌──────┴───────┐
  分组结果      │ 1组 × 17层   │                    │ 4组 × 4层    │ base 被拆碎
               │ 3组 × 16层   │                    │ 1组 × 5层    │ DSpark
               └──────┬───────┘                    │10组 × ~5层   │ Mamba 被拆碎
                       │                            └──────┬───────┘
                 4 个 group                            15 个 group
                 4 次 build_meta                      15 次 build_meta
```

---

## 4. 日志实测数据验证

### 4.1 v1 (MTP) 分组结果

来源: `/opt/z00830407/mtp_kv_analyze.log`

```
get_kv_cache_groups called with 65 layers, type counts: {'FullAttentionSpec': 17, 'MambaSpec': 48}
  spec_identity: FullAttentionSpec(num_kv_heads=1, head_size=256, ...) -> 17 layers  ← 含 mtp
  spec_identity: MambaSpec(shapes=((10, 2560), (12, 128, 128)), ...) -> 48 layers

group_size decision: min_num_layers=17, max_num_layers=48, max/min_ratio=2.82, group_size=17

kv_cache_group[0]: 17 layers, FullAttentionSpec
  [layers.7,11,15,19,23,27,31,35,39,43,47,51,55,59,63.self_attn.attn, mtp.layers.0.self_attn.attn]

kv_cache_group[1]: 16 layers, MambaSpec  [layers.0,8,12,16,...stride=3...]
kv_cache_group[2]: 16 layers, MambaSpec  [layers.1,9,13,17,...stride=3...]
kv_cache_group[3]: 16 layers, MambaSpec  [layers.2,10,14,18,...stride=3...]

build_meta: 4 kv_cache_groups, 4 total attn_groups (每组 1 个 attn_group)
```

### 4.2 v2 (DSpark) 分组结果

来源: `/opt/z00830407/dspark_kv_analyze.log`

```
get_kv_cache_groups called with 69 layers, type_counts: {'FullAttentionSpec': 21, 'MambaSpec': 48}
  spec_identity: FullAttentionSpec(num_kv_heads=1, head_size=256, ...) -> 16 layers  ← base
  spec_identity: FullAttentionSpec(num_kv_heads=2, head_size=128, ...) -> 5 layers   ← DSpark 异构
  spec_identity: MambaSpec(shapes=((10, 2560), (12, 128, 128)), ...) -> 48 layers

3 spec buckets:
  bucket[0]: FullAttentionSpec, num_layers=16, spec_hash=4457966727369825676
  bucket[1]: FullAttentionSpec, num_layers=5,  spec_hash=-5539063342261036244  ← 异构小桶
  bucket[2]: MambaSpec,         num_layers=48, spec_hash=-4846187972312919609

group_size decision: min_num_layers=5, max_num_layers=48, max/min_ratio=9.60, group_size=5

kv_cache_group[0]:  4 layers, FullAttentionSpec [layers.3,19,35,51]
kv_cache_group[1]:  4 layers, FullAttentionSpec [layers.7,23,39,55]
kv_cache_group[2]:  4 layers, FullAttentionSpec [layers.11,27,43,59]
kv_cache_group[3]:  4 layers, FullAttentionSpec [layers.15,31,47,63]
kv_cache_group[4]:  5 layers, FullAttentionSpec [model.layers.64-68]
kv_cache_group[5-12]: ~5 layers each, MambaSpec (stride=10)
kv_cache_group[13]: 4 layers, MambaSpec
kv_cache_group[14]: 4 layers, MambaSpec

get_kv_cache_groups final result: 15 kv_cache_groups
```

### 4.3 关键对比：MTP vs DSpark 的 spec 差异

| | base self_attn | draft/MTP self_attn | 是否同桶 |
|---|---|---|---|
| **MTP** | num_kv_heads=1, head_size=256 | num_kv_heads=**1**, head_size=**256** | **同桶** → 17层 |
| **DSpark** | num_kv_heads=1, head_size=256 | num_kv_heads=**2**, head_size=**128** | **不同桶** → 16层 + 5层 |

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
    # 对齐 attn page_size 到 mamba page_size（仅设置 page_size_padded）
    for layer_name in attn_layer_names:
        if kv_cache_spec[layer_name].page_size_bytes < mamba_page_size_padded:
            object.__setattr__(kv_cache_spec[layer_name], "page_size_padded", mamba_page_size_padded)
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
        kv_cache_spec[layer_name] = replace(
            spec, page_size_padded=page_size_padded,
            indexes_kv_by_block_stride=True)  # ← v2 特有
    kv_cache_spec.update(mamba_specs)
```

### 5.3 v1 vs v2 的关键行为差异

| 维度 | v1 | v2 |
|------|----|----|
| `indexes_kv_by_block_stride` | **False**（只设 `page_size_padded`） | **True**（用 `replace()` 重设 spec） |
| MambaSpec 层数 | 48 | 48 |
| base FullAttentionSpec 层数 | 16 (+1 MTP) | 16 |
| DSpark FullAttentionSpec 层数 | — | 5（异构） |

`indexes_kv_by_block_stride` 的差异不影响分组结果（同一路径内所有 AttentionSpec 该字段一致），但 v1/v2 的不一致可能导致运行时行为差异，值得注意。

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
缺点：draft 层不参与 stride 切分，可能影响 PP 场景下的均衡性；draft 层单层成组会破坏"所有组层数相同"的约束，需评估对 block_table 内存分配的影响。

**预期效果**：非 draft 部分 = 2 桶 (16 FullAttentionSpec + 48 MambaSpec)，`group_size=16`，1+3=4 组；draft 部分 5 层独立 1 组。**总计 5 个 kv_cache_groups**（vs 当前 15 个）。

### 方案 C: 设置 group_size 下限

在 `_get_kv_cache_groups_uniform_page_size` 中设置 group_size 的最小值，防止小桶拖垮大桶：

```python
# 伪代码
MIN_GROUP_SIZE = 8  # 或其他合理值
group_size = max(min_num_layers, MIN_GROUP_SIZE)
```

优点：改动最小，一行代码。
缺点：需要确定合理的下限值，可能对其他模型产生副作用；draft 层仍然与非 draft 层混合分组，DSpark 的 5 层桶会需要 padding 到 group_size。

**预期效果**（以 `MIN_GROUP_SIZE=16` 为例）：3 桶 (16+5+48)，`group_size=16`，1+1+3=5 组，DSpark 的 5 层桶需 11 层 padding。**总计 5 个 kv_cache_groups**。

### 方案 D: Draft 层使用与 base 相同的 FullAttentionSpec（unify spec） — 不可行

在 `get_kv_cache_spec` 阶段，将 DSpark draft 层的 FullAttentionSpec 替换为与 base 一致的 spec，使 draft 层归入 base 的 spec 桶。

缺点：draft 层的 KV cache 实际参数与 base 不同（`num_kv_heads=2 vs 1, head_size=128 vs 256`），强行统一 spec 会导致 KV cache 内存分配与实际使用不匹配。**此方案不可行**。

### 推荐方案

**方案 B** 最合理：将 DSpark draft 层从分组输入中分离，非 draft 部分独立分组后再将 draft 层追加为独立 kv_cache_group。非 draft 部分的分组结果与 MTP 路径一致（4 个 kv_cache_groups），draft 层独立 1 组，总计 5 个 kv_cache_groups。

但需注意：draft 层单层成组后，该组层数（5 层）与其他组（16 层）不同，会打破 `_get_kv_cache_groups_uniform_page_size` 中"所有组层数相同"的约束。需要评估下游 `KVCacheManager` 的 block 分配逻辑是否支持不同大小的组。如果不支持，可能需要为 draft 组添加 padding 层（11 层 padding），或者修改 `KVCacheManager` 以支持异构组。

---

## 7. 相关文件索引

| 文件 | 行号 | 作用 |
|------|------|------|
| `vllm/v1/core/kv_cache_utils.py` | 1137-1256 | `_get_kv_cache_groups_uniform_page_size` 分组算法 |
| `vllm/v1/core/kv_cache_utils.py` | 1728-1794 | `get_kv_cache_groups` 入口分发 |
| `vllm_ascend/worker/model_runner_v1.py` | 4516-4660 | v1 `get_kv_cache_spec` |
| `vllm_ascend/worker/v2/attn_utils.py` | 56-199 | v2 `get_kv_cache_spec` |
| `vllm_ascend/worker/v2/attn_utils.py` | 202-323 | v2 `build_attn_metadata` 迭代 kv_cache_groups |
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

### 8.2 日志实测验证结果

已通过两条路径的实际运行日志验证，详见：

- v2 DSpark: `/opt/z00830407/dspark_kv_analyze.log`
- v1 MTP: `/opt/z00830407/mtp_kv_analyze.log`

详细分析见 `/vllm-workspace/vllm-ascend/kv_spec_recover/mtp_vs_dspark_log_evidence.md`。

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
