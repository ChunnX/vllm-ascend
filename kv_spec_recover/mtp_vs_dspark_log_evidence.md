# MTP v1 与 DSpark v2 的 kv_cache_groups 对比日志分析

## 一、MTP (v1) 路径日志

### 1. kv_cache_spec 入口

```
get_kv_cache_groups called with 65 layers, type counts: {'FullAttentionSpec': 17, 'MambaSpec': 48}
```

65 层 = 16 base self_attn + 1 MTP self_attn + 48 base linear_attn

### 2. 2 个 spec identity 桶

**桶 A — FullAttentionSpec (17 层，含 MTP):**
```
FullAttentionSpec(
    num_kv_heads=1, head_size=256, head_size_v=256,
    sliding_window=None, non_causal=False,
    block_size=1536, dtype=torch.bfloat16, kv_quant_mode=0,
    page_size_padded=1624064, indexes_kv_by_block_stride=False)
→ 17 layers: [layers.7,11,15,19,23,27,31,35,39,43,47,51,55,59,63.self_attn.attn,
              mtp.layers.0.self_attn.attn]    ← MTP 层与 base 同 spec！
```

**桶 B — MambaSpec (48 层):**
```
MambaSpec(shapes=((10, 2560), (12, 128, 128)), dtypes=(torch.bfloat16, torch.float32),
    block_size=16384, mamba_type=MambaAttentionBackendEnum.GDN_ATTN, page_size_padded=1624064)
→ 48 layers: [layers.0,1,2,4,5,6,8,9,...,60,61,62.linear_attn]
```

**关键：MTP 层的 `num_kv_heads=1, head_size=256` 与 base 层完全一致，归入同一桶。**

v1 的 `get_kv_cache_spec` 逐层确认（以 TP0 为例）：
- 17 层 spec_hash=5628838180901049694 (FullAttentionSpec，含 MTP)
- 48 层 spec_hash=-7959731826289515535 (MambaSpec)

### 3. group_size 计算

```
2 spec buckets
  bucket[0]: FullAttentionSpec, num_layers=17
  bucket[1]: MambaSpec,         num_layers=48

group_size decision: min_num_layers=17, max_num_layers=48,
    max/min_ratio=2.82, threshold=1.5, group_size=17
```

- 只有 2 个桶，`min=17, max=48`
- `48/17=2.82 > 1.5`，不满足取 max 的条件
- **`group_size = 17`**

### 4. 分组结果：4 个 kv_cache_groups

```
group[0]: 17 layers, FullAttentionSpec [base 16层 + MTP 1层]
group[1]: 16 layers, MambaSpec [linear_attn, stride=3 切分]
group[2]: 16 layers, MambaSpec [linear_attn, stride=3 切分]
group[3]: 16 layers, MambaSpec [linear_attn, stride=3 切分]
```

### 5. build_meta 确认

```
_build_attention_metadata: 4 kv_cache_groups, 4 total attn_groups
  kv_cache_group[0]: 17 layers, 1 attn_groups
  kv_cache_group[1]: 16 layers, 1 attn_groups
  kv_cache_group[2]: 16 layers, 1 attn_groups
  kv_cache_group[3]: 16 layers, 1 attn_groups
```

**4 次 build_meta 调用。**

---

## 二、DSpark (v2) 路径日志（回顾）

### 1. kv_cache_spec 入口

```
get_kv_cache_groups called with 69 layers, type counts: {'FullAttentionSpec': 21, 'MambaSpec': 48}
```

69 层 = 16 base self_attn + 5 DSpark self_attn + 48 base linear_attn

### 2. 3 个 spec identity 桶

**桶 A — base FullAttentionSpec (16 层):**
```
num_kv_heads=1, head_size=256, head_size_v=256, indexes_kv_by_block_stride=True
```

**桶 B — DSpark FullAttentionSpec (5 层):**
```
num_kv_heads=2, head_size=128, head_size_v=128, indexes_kv_by_block_stride=True
← 与桶 A 不同！
```

**桶 C — MambaSpec (48 层):**
```
shapes=((10, 2560), (12, 128, 128)), mamba_type=GDN_ATTN
```

**关键：DSPark draft 层 `num_kv_heads=2, head_size=128` ≠ base 的 `num_kv_heads=1, head_size=256`，产生独立小桶。**

### 3. group_size 计算

```
3 spec buckets
  bucket[0]: FullAttentionSpec, num_layers=16
  bucket[1]: FullAttentionSpec, num_layers=5    ← DSpark 异构小桶
  bucket[2]: MambaSpec,         num_layers=48

group_size decision: min_num_layers=5, max_num_layers=48,
    max/min_ratio=9.60, threshold=1.5, group_size=5
```

**`group_size = 5`**（被 DSpark 的 5 层桶拖低）

### 4. 分组结果：15 个 kv_cache_groups

```
group[0]:  4 layers, FullAttentionSpec (base, stride=4)
group[1]:  4 layers, FullAttentionSpec (base, stride=4)
group[2]:  4 layers, FullAttentionSpec (base, stride=4)
group[3]:  4 layers, FullAttentionSpec (base, stride=4)
group[4]:  5 layers, FullAttentionSpec (DSpark)
group[5-12]: ~5 layers each, MambaSpec (stride=10)
group[13]: 4 layers, MambaSpec
group[14]: 4 layers, MambaSpec
```

**15 次 build_meta 调用。**

---

## 三、逐项对比

| 维度 | MTP (v1) | DSpark (v2) | 差异原因 |
|------|----------|-------------|----------|
| 总层数 | 65 | 69 | DSpark 5层 vs MTP 1层 |
| FullAttentionSpec 桶数 | 1 (17层) | 2 (16+5层) | DSpark draft 的 `num_kv_heads`/`head_size` 与 base 不同 |
| MambaSpec 桶数 | 1 (48层) | 1 (48层) | 相同 |
| spec identity 桶总数 | **2** | **3** | DSpark 制造了异构小桶 |
| min_num_layers | 17 | 5 | DSpark 小桶拖低 |
| group_size | **17** | **5** | 直接后果 |
| kv_cache_groups 数 | **4** | **15** | 4+1+10 vs 1+3 |
| build_meta 次数 | **4** | **15** | 效率差距 3.75x |

---

## 四、根因链条（日志验证版）

```
MTP 层:  num_kv_heads=1, head_size=256  → 与 base 相同 → 合入 base 桶(17层)
DSpark 层: num_kv_heads=2, head_size=128 → 与 base 不同 → 独立桶(5层)
                                              │
                                              ▼
                               min_num_layers=5 (DSpark桶)
                                              │
                                              ▼
                                    group_size = 5
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                 base 16层被拆4组     DSpark 5层=1组      Mamba 48层被拆10组
                          │                   │                   │
                          └───────────────────┼───────────────────┘
                                              ▼
                                      15 个 kv_cache_groups
```

如果 DSpark draft 层也用 `num_kv_heads=1, head_size=256`（与 base 一致），则：
- 2 个桶：FullAttentionSpec(21层) + MambaSpec(48层)
- `min_num_layers=21, group_size=21`
- 1 + cdiv(48,21)=3 = **4 个 kv_cache_groups**，与 MTP 一致

---

## 五、补充发现：v1 vs v2 的 MambaSpec 层数一致

两条路径的 MambaSpec 都是 **48 层**（所有 GDN linear_attention 层）。

之前用户观察到 v1 只有 16 个 linear_attn 层，那可能是观察的是 v1 的 `_build_attn_group_metadata` 内的 attn_group 内容，而非 kv_cache_spec 的全量。实际日志确认：

v1 MTP 的 65 层 = 17 FullAttentionSpec + 48 MambaSpec（含全部 48 个 GDN 层）
v2 DSpark 的 69 层 = 21 FullAttentionSpec + 48 MambaSpec（含全部 48 个 GDN 层）

**48 个 MambaSpec 层在两条路径中一致**，问题纯粹来自 DSpark draft 层的异构 FullAttentionSpec。

---

## 六、另一个细微差异：indexes_kv_by_block_stride

| 路径 | FullAttentionSpec 的 indexes_kv_by_block_stride |
|------|------------------------------------------------|
| v1 MTP | **False** |
| v2 DSpark | **True** |

这是因为 v2 的 `get_kv_cache_spec` 在有 MambaSpec 层时，对所有 AttentionSpec 执行了：
```python
kv_cache_spec[layer_name] = replace(spec, page_size_padded=page_size_padded, indexes_kv_by_block_stride=True)
```

而 v1 的处理方式不同：
```python
object.__setattr__(kv_cache_spec[layer_name], "page_size_padded", mamba_page_size_padded)
```
v1 只设置了 `page_size_padded`，没有设置 `indexes_kv_by_block_stride=True`。

这个差异不影响分组结果（因为同一路径内所有 AttentionSpec 的该字段一致，不影响桶的划分），但 v1/v2 的不一致可能导致运行时行为差异，值得注意。
