# DSpark kv_cache_groups 问题日志证据分析

## 日志来源

`/opt/z00830407/dspark_kv_analyze.log`，运行 Qwen3.6 + DSpark 的 v2 路径。

---

## 证据链

### 证据 1：kv_cache_spec 共 69 层，分为 3 个 spec identity 桶

```
[DIAG-KV-GROUP] get_kv_cache_groups called with 69 layers,
    type counts: {'FullAttentionSpec': 21, 'MambaSpec': 48}
```

共 69 层，其中 21 个 FullAttentionSpec + 48 个 MambaSpec。

### 证据 2：三个 spec identity 桶的关键差异

**桶 A — base self_attn (16 层):**
```
spec_identity: FullAttentionSpec(
    num_kv_heads=1,       ← 关键差异
    head_size=256,        ← 关键差异
    head_size_v=256,
    sliding_window=None,
    non_causal=False,
    block_size=1536,
    dtype=torch.bfloat16,
    kv_quant_mode=0,
    page_size_padded=1624064,
    indexes_kv_by_block_stride=True
) -> 16 layers: [layers.7,11,15,19,23,27,31,35,39,43,47,51,55,59,63.self_attn.attn]
```

**桶 B — DSpark draft self_attn (5 层):**
```
spec_identity: FullAttentionSpec(
    num_kv_heads=2,       ← 与桶 A 不同！
    head_size=128,        ← 与桶 A 不同！
    head_size_v=128,
    sliding_window=None,
    non_causal=False,
    block_size=1536,
    dtype=torch.bfloat16,
    kv_quant_mode=0,
    page_size_padded=1624064,
    indexes_kv_by_block_stride=True
) -> 5 layers: [model.layers.64,65,66,67,68.self_attn.attn]
```

**桶 C — base linear_attn (48 层):**
```
spec_identity: MambaSpec(
    shapes=((10, 2560), (12, 128, 128)),
    dtypes=(torch.bfloat16, torch.float32),
    block_size=16384,
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    page_size_padded=1624064
) -> 48 layers: [layers.0,1,2,4,5,6,8,9,10,...,60,61,62.linear_attn]
```

**关键发现：base 层 num_kv_heads=1, head_size=256；DSpark draft 层 num_kv_heads=2, head_size=128。**
这是因为 base 模型（Qwen3.6 大模型）与 DSpark draft 模型（小模型 dspark_qwen3_8b）架构参数不同，
导致 FullAttentionSpec 的 `__hash__`/`__eq__` 不一致，被分入不同桶。

### 证据 3：走了 uniform_page_size 分组路径

```
[DIAG-KV-GROUP] Path: uniform_page_size (general hybrid model),
    filtered_spec has 69 layers (excluded 0 HiddenStateCacheSpec)
```

page_size 已统一（1 个 distinct page_size = 1624064），无需额外对齐。

### 证据 4：3 个 spec 桶确认

```
[DIAG-KV-GROUP] _get_kv_cache_groups_uniform_page_size: 3 spec buckets
  bucket[0]: type=FullAttentionSpec, num_layers=16, spec_hash=4457966727369825676
  bucket[1]: type=FullAttentionSpec, num_layers=5,  spec_hash=-5539063342261036244  ← DSpark 异构桶
  bucket[2]: type=MambaSpec,         num_layers=48, spec_hash=-4846187972312919609
```

三个桶的 spec_hash 均不同。DSpark draft 层虽然也是 FullAttentionSpec 类型，
但 spec_hash 与 base 不同（num_kv_heads 和 head_size 差异）。

### 证据 5：group_size 被拖低至 5

```
[DIAG-KV-GROUP] group_size decision: min_num_layers=5, max_num_layers=48,
    max/min_ratio=9.60, threshold=1.5, group_size=5
```

- `min_num_layers=5` 来自 DSpark 桶（仅 5 层）
- `max_num_layers=48` 来自 MambaSpec 桶
- `max/min_ratio=9.60 > 1.5`，不满足取 max 的条件，因此 `group_size=5`

**如果 DSpark draft 层与 base 层 spec 相同（像 MTP 那样），则只有 2 个桶：**
- 桶 A: 21 个 FullAttentionSpec，桶 B: 48 个 MambaSpec
- `min_num_layers=21`, `max/min_ratio=48/21=2.29 > 1.5`, `group_size=21`
- 结果：1 + ceil(48/21)=3 = **4 个 kv_cache_groups**

### 证据 6：15 个 kv_cache_groups 最终分组

```
[DIAG-KV-GROUP] _get_kv_cache_groups_uniform_page_size result: 15 total groups
```

分组明细：

| group | 层数 | 类型 | 层名 |
|-------|------|------|------|
| 0 | 4 | FullAttentionSpec (base) | layers.3,19,35,51.self_attn.attn |
| 1 | 4 | FullAttentionSpec (base) | layers.7,23,39,55.self_attn.attn |
| 2 | 4 | FullAttentionSpec (base) | layers.11,27,43,59.self_attn.attn |
| 3 | 4 | FullAttentionSpec (base) | layers.15,31,47,63.self_attn.attn |
| 4 | 5 | FullAttentionSpec (DSpark) | model.layers.64-68.self_attn.attn |
| 5 | 5 | MambaSpec | linear_attn layers (stride 切分) |
| 6 | 5 | MambaSpec | linear_attn layers |
| 7 | 5 | MambaSpec | linear_attn layers |
| 8 | 5 | MambaSpec | linear_attn layers |
| 9 | 5 | MambaSpec | linear_attn layers |
| 10 | 5 | MambaSpec | linear_attn layers |
| 11 | 5 | MambaSpec | linear_attn layers |
| 12 | 5 | MambaSpec | linear_attn layers |
| 13 | 4 | MambaSpec | linear_attn layers |
| 14 | 4 | MambaSpec | linear_attn layers |

- 16 个 base FullAttentionSpec 层 → 4 组（每组 4 层，stride=4 切分）
- 5 个 DSpark FullAttentionSpec 层 → 1 组
- 48 个 MambaSpec 层 → 10 组（8 组 5 层 + 2 组 4 层，stride=10 切分）
- **总计 15 个 kv_cache_groups**

### 证据 7：最终 kv_cache_groups 确认

```
[DIAG-KV-GROUP] get_kv_cache_groups final result: 15 kv_cache_groups
```

---

## 根因确认

日志完整验证了之前的分析：

1. **DSpark draft 模型的 `num_kv_heads=2, head_size=128` 与 base 模型的 `num_kv_heads=1, head_size=256` 不同** → 产生独立的 FullAttentionSpec 桶（仅 5 层）
2. **`min_num_layers=5` 导致 `group_size=5`** → 48 个 MambaSpec 层被拆成 10 组，16 个 base FullAttentionSpec 层被拆成 4 组
3. **如果 DSpark draft 层与 base 层 spec 相同**（像 MTP 那样），则只有 2 个桶，`group_size=21`，结果仅 4 个 kv_cache_groups

**对比 v1 MTP 路径的推测**：MTP 层使用 base 模型相同的 `num_kv_heads=1, head_size=256`，与 base 层归入同一桶，21+48 两个桶 → `group_size=21` → 1+3=4 个 kv_cache_groups。

---

## 补充发现

### 48 个 MambaSpec 层全部出现

之前 v1 路径的分析中，用户只观察到 16 个 linear_attn 层。但日志显示 v2 路径下所有 48 个 GDN/linear_attention 层都产生了 MambaSpec。

这意味着 v1 和 v2 路径在 GDN 层的 KV cache 处理上存在差异：
- v1 中可能有 GDN 层的状态共享机制（`kv_sharing_target_layer_name`），使得只有 16/48 层产生 MambaSpec
- v2 中所有 48 层都产生 MambaSpec，进一步放大了 group_size 被拖低的影响

这个差异值得进一步调查，但即使 v2 也只有 16 个 MambaSpec 层（同 v1），DSpark 的 5 层异构桶仍会导致 `group_size=5`，产生 4+4+1=9 个 kv_cache_groups，仍然远多于 v1 的 4 个。
