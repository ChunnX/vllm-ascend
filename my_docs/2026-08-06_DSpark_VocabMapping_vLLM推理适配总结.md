# DSpark 小词表推理:vLLM Ascend 侧适配总结

日期:2026-08-06
仓库:`vllm-ascend`,分支 `dspark-vocab-mapping-inference`(远端 `fg11991`)
基线:`dea3bc377`
提交:`1d9a70c65` → `eb042af60` → `8e341e0f2`
范围:**仅 model runner v1**(v2 由 upstream vLLM 的 `DSparkSpeculator` 负责,已自带小词表支持)

前置文档:

- `my_docs/2026-08-06_DSpark_VocabMapping_vLLM推理代码Review.md` — 两轮独立复审(同目录)
- `specforge_sgl0514/SpecForge-dspark-vocab/my_docs/2026-08-05_DSpark_VocabMapping_端到端适配总结.md` — 训练侧

---

## 0. 三句话

1. 训练侧(SpecForge)已支持 DSpark 词表裁剪,但**服务侧不认**——vLLM Ascend 的 v1 proposer 完全没有消费 `d2t`,裁剪 checkpoint 上线即形状崩溃或静默错 token。本次把这条链路补上。
2. 开关只有 checkpoint 本身:draft config 顶层 `draft_vocab_size` + 权重里的 `d2t`。**启动命令一个字不用改。**
3. 生产代码净增约 130 行执行逻辑,分布在 2 个文件、4 个新方法;全词表路径逐字节不变。

---

## 1. 背景:缺口在哪一段

DSpark 的推理分两段:并行 backbone 出一整块的 base logits,再由 Markov 头串行地对每个位置做偏置修正。词表裁剪只影响后半段和 LM 头。

训练侧(SpecForge)`feat/dspark-vocab-mapping` 已经做完,产出的 checkpoint 有三个与全词表不同的特征:

| 特征 | 说明 |
|---|---|
| `markov_w2` 输出维是 `draft_vocab_size` (K) | 而 `markov_w1` 仍是全 target 词表——它被**上一个真实 token** 索引,那个 token 可以是任意 target id |
| 带 `t2d` / `d2t` 两个 buffer | `d2t[i] = kept_target_ids[i] - i`,即 `target_id = i + d2t[i]` |
| **没有 `lm_head`** | DSpark/DFlash 训练时借用冻结的 target head,裁剪时取 `target.lm_head.weight[t2d]` 的 K 行 |

配套 vLLM 的模型侧其实已经就绪:`Qwen3DSparkForCausalLM` 有 `draft_vocab_size`、`draft_id_to_target_id`、`compute_draft_logits`、`map_draft_to_target`,`load_weights` 也会把 `d2t` 改名接入(`t2d` 训练专用,直接丢弃)。upstream 的 **v2/GPU** speculator 也已正确实现。

**唯独 vLLM Ascend 的 v1 proposer 没接。** 具体两处:

1. `_run_merged_draft` 的 dspark 分支调用 `compute_logits`,它在裁剪时会把 K 维 logits 散射进全 target 词表(其余 `-inf`),而 `markov_bias` 返回 K 维 → **形状不匹配,直接崩**。
2. `_maybe_share_lm_head` 遇到"有 `d2t` 但没有 `lm_head`"这种组合时,会走到"保留 draft 自己的 lm_head"分支——那是个**未初始化的 `ParallelLMHead(K, H)`**,不报错,输出垃圾。

---

## 2. 做了什么

### 2.1 采样改到 draft 空间(热路径)

`vllm_ascend/spec_decode/llm_base_proposer.py`

```
base_logits[K] + markov_bias[K]
    → argmax → draft_id
    → map_draft_to_target → target_id
    → 同时用于:写入 proposal buffer(交给 target verifier)
                下一步 markov_w1[target_id]
```

三个要点:

- **`_dspark_base_logits`**:优先 `compute_draft_logits`(draft 空间,无散射),模型没有这个方法时回退 `compute_logits`。这样 DSV4-DSpark(`deepseek_v4_dspark.py`,全词表、无这两个方法)和全词表 Qwen3-DSpark 都走原来的张量。
- **`_dspark_map_to_target`**:映射**必须发生在 id 被喂回 `markov_w1` 之前**。`markov_w1` 跨越全 target 词表,未映射的 draft id 是个**合法但指错行**的下标——不会崩,只会静默拉低接受率。这是整条链路最隐蔽的错误点。
- **`_dspark_sample_block`**:把原来内联的循环提取成方法,行为不变,目的是让单测能直接打这段逻辑而不必驱动整个 forward。

### 2.2 运行时切出裁剪头(load 期)

`_build_pruned_lm_head_from_target`:按 `kept[i] = i + d2t[i]` 从 target head 取行,复现训练时 `target.lm_head.weight[t2d]` 的同一份切片。

TP 处理:

1. 每个 rank 用 `index_copy_` / `index_select` 填自己 shard 命中的行(避开布尔行索引,NPU 支持性未验证);
2. 在**target head 自己的 `comm_group`** 上做一次 all-reduce,拼出完整 `[K, H]`(K 行不是 V 行,K=32000/H=5120/bf16 约 328 MB 临时);
3. 交给 draft head 自己的 `weight_loader` 做本地分片和 padding。

**不要求两个 head 同组**:`draft_tensor_parallel_size=1` 时 draft head 在 `patch_tensor_parallel_group` 造的单卡组里,而 reduce 只需要覆盖 target 行的那个组;每个 target rank 拼出同一份完整头,各自的单卡 draft head 全量加载。

`get_tp_group()` **不能**当兜底:`load_model` 就跑在那个 patch 里面,全局 TP 组此刻可能是 draft 的单卡组。

### 2.3 校验(load 期)

`_validate_draft_vocab_mapping`,只要 `draft_id_to_target_id is not None` 就跑,**不管 head 是派生的还是 checkpoint 自带的**——自带 head 的 checkpoint 一样要按 `d2t` 解释每个 draft id。

| 检查 | 拦住什么 |
|---|---|
| mapping 是否真的加载 | config 声明裁剪但 checkpoint 没有 `d2t` |
| `i + d2t[i]` 是否在 `[0, V)` | 对着另一个 target 训出来的 draft |
| kept ids 是否严格递增 | offsets 与 head 行序不一致 |

"是否加载"这一条不能看内容:`d2t` 全零既可能是"没装载",也可能是合法的"保留 target ids 0..K-1"。训练侧用**全 False 的 `t2d`** 消歧(真实映射必然选中恰好 K 个),但 vLLM 把 `t2d` 丢了。所以在 `AscendQwen3DSparkForCausalLM.load_weights` 里加了个透传生成器,在权重流经过时记下 `has_draft_id_mapping`——那是**最后一个还能看到原始权重名的地方**,且不需要改 upstream vLLM。

### 2.4 防呆

`enable_reduce_sample` + dspark 直接 `NotImplementedError`。这个组合**改动前就是坏的**:那条分支对 backbone logits 做平铺 argmax、完全跳过 Markov 头,还返回一维张量给按 block 切片的调用方。

---

## 2.5 代码地图

行号取自 `c6b69f34c^`(即 `8e341e0f2`,文档提交之前)。改动只涉及 2 个生产文件。

### `vllm_ascend/spec_decode/llm_base_proposer.py`(+257 / −24)

| 行 | 符号 | 时机 | 做什么 |
|---|---|---|---|
| 22 | `import UnquantizedEmbeddingMethod` | — | 量化判据依赖它 |
| **456–528** | `_maybe_share_lm_head` | load | 唯一被修改的既有方法,见下 |
| 530–536 | `_resolve_target_lm_head` | load | 从既有代码里提出来的 7 行,消除重复的 target head 解析 |
| **538–586** | `_validate_draft_vocab_mapping` | load | d2t 三项校验,返回 kept target ids |
| **588–690** | `_build_pruned_lm_head_from_target` | load | 按 d2t 从 target head 切 K 行 |
| 1248–1262 | `_dspark_base_logits` | **热路径(图内)** | 优先 `compute_draft_logits`,回退 `compute_logits` |
| 1264–1297 | `_dspark_sample_block` | **热路径(图内)** | 原内联 Markov 循环提取而成,行为不变 |
| 1299–1310 | `_dspark_map_to_target` | **热路径(图内)** | draft id → target id,无映射时恒等 |

`_maybe_share_lm_head` 内部的三处改动:

| 行 | 改动 | 为什么 |
|---|---|---|
| 467–472 | 新增:只要 `draft_id_to_target_id is not None` 就调 `_validate_draft_vocab_mapping()` | 自带 head 的 checkpoint 也按 d2t 解释每个 draft id,不能豁免校验(第二轮 review P1) |
| 473–484 | 新增分支:有 d2t 且 `has_own_lm_head=False` → 派生 head | 原来两个分支都不对:共享 target head 词表错,保留自有 head 是未初始化张量 |
| 491–501 | 改写:target head 解析改调 `_resolve_target_lm_head` | 纯提取,行为不变 |

`_run_merged_draft` 内部:

| 行 | 改动 |
|---|---|
| 1382–1390 | 新增 `enable_reduce_sample` + dspark 的 `NotImplementedError` |
| 1418 | 原来 20 行内联的 Markov 循环 → `draft_token_ids = self._dspark_sample_block(...)` |

`_build_pruned_lm_head_from_target` 内部的关键行(这几处是两轮 review 的落点):

| 行 | 内容 |
|---|---|
| 626 | 量化判据:`isinstance(target_quant_method, UnquantizedEmbeddingMethod)` —— **不是** dtype,FP8 是浮点会漏过 |
| 639 | 归约组只取 `target_lm_head.comm_group`,**不用** `get_tp_group()` 兜底 |
| 669 | `index_copy_` + `index_select` 填本 rank 命中的行(避开布尔行索引) |
| 677 | 唯一一次 collective:`target_group.all_reduce(pruned_weight)` |
| 679 | 交给 `draft_lm_head.weight_loader` 做本地分片与 padding |

`_validate_draft_vocab_mapping` 内部:

| 行 | 检查 |
|---|---|
| 556–565 | 是否加载:优先 `has_draft_id_mapping`,无则回退 `d2t.any()` 启发式 |
| 573–578 | `i + d2t[i]` 是否落在 `[0, V)` |
| 581–586 | kept ids 是否严格递增 |

### `vllm_ascend/models/qwen3_dspark.py`(+20 / −3)

| 行 | 符号 | 做什么 |
|---|---|---|
| 30–39 | `has_draft_id_mapping: bool = False` | 类属性 + 10 行注释解释为什么内容判断不了 |
| 45–50 | `_note_draft_id_mapping` | 透传生成器,权重流经过时记下有没有 `d2t` |
| 52–61 | `load_weights` | 唯一改动:末行由 `super().load_weights(weights)` 变成 `super().load_weights(self._note_draft_id_mapping(weights))` |

⚠️ 生成器是**惰性**的,`has_draft_id_mapping` 只在流被消费后才置位。生产路径安全(upstream 的 `load_weights` 会先把整个流灌进 dict,而 `_maybe_share_lm_head` 在 `load_model` 返回之后才跑),但**写测试时必须显式消费**,否则拿到的永远是 `False`。

### 测试

| 文件 | 条数 | 覆盖 |
|---|---|---|
| `tests/ut/spec_decode/test_dspark_vocab_mapping.py`(新增 498 行) | 25 | 采样(4)、派生 head(9)、d2t 校验(5)、ownership 默认值(2)、routing(5) |
| `tests/ut/model_executor/test_qwen3_dspark.py`(+52 / −1) | 4 新 + 1 改 | presence flag 的记录;改的那条是因为权重现在经生成器传递,断言需先 `list()` |

## 3. 什么情况下会失效

### 3.1 算法固有的上界(无法通过工程消除)

**任何落在 draft 词表外的 target token,draft 永远提不出来,那个位置必然被拒。** 接受率天花板 ≈ 训练时报的 `top-K token frequency ratio`。

端到端吞吐 = 加速比 × 接受长度。小词表把 draft 采样段做快了(见 §5),但如果覆盖率把接受长度拉低的幅度超过省下的时间,**净收益可以为负**。这只能真机实测,没有先验答案。

### 3.2 会显式报错的(响亮失败,可诊断)

| 触发条件 | 报错关键词 |
|---|---|
| config 声明 `draft_vocab_size` 但权重无 `d2t` | `carries no d2t mapping` |
| 映射指到 target 词表外 | `outside the target vocabulary` |
| 映射非严格递增 | `not strictly increasing` |
| target `lm_head` 被量化(含 FP8) | `requires an unquantized target lm_head` |
| target head 分片了但报不出 `comm_group` | `no comm_group` |
| 开了 `additional_config.enable_reduce_sample` | `does not support DSpark` |
| `draft_sample_method="probabilistic"` | v1 本来就 raise(非本次引入) |

target head 量化那条值得展开:modelslim 若没把 `lm_head` 放进跳过列表,派生路径会拒绝启动。**拒绝是正确行为**——直接拿量化权重当普通权重会丢 scale,静默算错。绕过办法是离线把 `target_lm_head.weight[t2d]` 的 K 行写进 draft checkpoint 的 `lm_head.weight`,代码会自动改走"自带 head"分支。

### 3.3 会静默失效的(必须靠日志确认)

**架构名不对。** SpecForge 原始 config 写的是 `"architectures": ["DSparkDraftModel"]`,而 `vllm_ascend/models/__init__.py` 把这个名字注册到 **DSV4 的 `DSparkDeepseekV4ForCausalLM`**,它连 `draft_vocab_size` 概念都没有。小词表要求架构名是 `Qwen3DSparkModel`。

唯一可靠的正向信号是启动日志:

```
[spec_decode/base] DSPARK draft uses a reduced vocabulary (32000 of 248320, 7.76x)
and ships no lm_head; derived it from the target lm_head rows kept by d2t.
```

**看不到这行 = 小词表路径没生效。**

⚠️ 这个信号**只覆盖"checkpoint 不带 lm_head"这一种形状**。若 checkpoint 自带切好的 head,日志只会说 "Detected DSpark model with distinct lm_head weights",不提词表——那种情况下没有直接日志能确认映射生效,只能靠校验没报错 + 接受率反推。**这是当前实现的一个可观测性缺口。**

### 3.4 已知的、有意为之的保守取舍

`d2t` 全零 + 模型类不记录 `has_draft_id_mapping`(即非 `AscendQwen3DSparkForCausalLM`)时,会**误拒**一个合法的"保留 target ids 0..K-1"映射。

这是权衡的结果:两种失败代价不对称——把未装载当合法映射是**静默错 token**,而把合法的 first-K 映射当未装载是**启动即报错、一眼可诊断**。而且"最高频 K 个词恰好是 id 0..K-1"不是真实 tokenizer 会产生的词表。

### 3.5 不在覆盖范围

- **model runner v2**:upstream vLLM 的 `DSparkSpeculator._sample_sequential` 已正确实现,本次未评审也未改动。
- **DSV4-DSpark**:全词表专用,`deepseek_v4_dspark.py` 没有 `draft_vocab_size`/`d2t`,本次不支持也不受影响。
- **DFlash**:派生 head 和校验逻辑对 dflash 同样生效(条件相同),但没有 dflash 裁剪 checkpoint 验证过。

---

## 4. 已有功能不变的保证(以及保证的强度)

这一节要分清"**验证过不变**"和"**推理认为不变**"。

### 4.1 有测试守着的

| 不变量 | 守它的测试 |
|---|---|
| 全词表 draft 的映射是恒等 | `test_full_vocabulary_draft_is_an_identity_remap` |
| 没有 `compute_draft_logits` 的模型(DSV4)回退原路径 | `test_falls_back_to_compute_logits_without_compute_draft_logits` |
| 无 `d2t`、无自有 head → 仍然共享 target head | `test_full_vocabulary_checkpoint_still_shares_the_target_head` |
| `has_own_lm_head` / `has_own_embed_tokens` 默认 `False` | `TestCheckpointOwnershipDefaults`(⚠️ 见 §5.2) |

最后一条是本次两轮 review 里唯一被证伪的 finding 的产物:有 reviewer 认为这两个标志不会被赋 `False`、新分支永不命中。事实是 `Qwen3ForCausalLM` 名义继承 `SupportsEagle3 → SupportsEagleBase`,后者把 `has_own_lm_head: bool = False` 定义成真实类属性,经 MRO 传下来。**更硬的证据是全词表 DSpark 今天在跑**——那条路径正是靠 `False` 才会去共享 target head。既然这个默认值是隐式的、且已经骗过一次 review,就把它显式钉住。

### 4.2 靠推理、未实测的

- **ACLGraph 入图不受影响。** 捕获范围是**整个 `_run_merged_draft`**(不只是 `self.model(...)`),所以 Markov 循环和新增的 d2t gather 都在图里。逐项审计:词表相关张量只是**窄了 7.76×**;`getattr` 是 Python、capture 时求值一次且对模型终身不变;d2t gather 形状 `[B]` 静态、无 host sync、`d2t` 地址在 load 后固定;常驻 buffer、返回视图、attn metadata、batch descriptor、`_update_full_graph_params` **一律未改**,所以图的 key 和输入地址契约不变。裁剪还**去掉**了 `compute_logits` 里那次 `[B*N, V]` 的 `-inf` 分配 + scatter,对图更友好。
  高级索引本身在这条路径上已有先例——同一函数里的 `last_hidden_states[token_indices_to_sample]` 今天就在图里跑。若真出问题,失败是响亮的(`torch.npu.graph` 捕获期抛 `RuntimeError`),改成 `torch.index_select` 即可(`compute_draft_token_ids` 的 dflash d2t 重映射就是这么写的)。
- **权重显存净增。** 全词表时 draft 复用 target 的 lm_head(零额外);裁剪后 draft 持有自己的 `[32000, 5120]` = +328 MB,`markov_w2` 省 111 MB,**净 +217 MB / TP**。`gpu_memory_utilization` 卡得紧时会从 KV cache 里挤出来。

---

## 5. 验证做到哪一步

### 5.1 做了什么

**30 条单测**(25 条新文件 + 4 条新增 + 1 条修改),覆盖:draft→target 映射与回喂、不走全词表散射、full-vocab fallback、单卡 / target TP2 / target TP2+draft TP1 的 head 行选择、量化与 FP8 拒绝、越界 / 重复 / 缺失映射、ownership 默认值、三种 lm_head routing、presence flag 的记录。

**每一条新增测试都做了变异验证**——注入对应故障、确认测试真的会红。这是从训练侧那次教训继承下来的规矩:"有测试守着"这句话需要证据,一个恒真的测试比没有测试更糟。共验证 12 种变异,全部捕获,包括:跳过 d2t 映射、退回全词表散射、丢掉 target shard 偏移、跳过 all-reduce、在 draft 组上 reduce、量化判断退回 dtype、校验只在派生分支跑、忽略 loader flag、model 侧不记录 flag。

### 5.2 没做什么(必须说清楚)

**开发机是 macOS,没有 vllm 也没有 NPU。** 单测是用 AST 把源文件里的真实方法体抽出来、绑到桩类上执行的——测的是提交的代码文本,不是抄写版,但:

- `TestCheckpointOwnershipDefaults` 那 2 条**在本地是 deselect 的**,它断言真实 vLLM 类,用桩复刻 MRO 只会测到桩本身。需要在装了 vllm 的机器 / CI 上跑才算数。
- **真实裁剪 checkpoint 的 loader 集成、多卡 HCCL collective、ACLGraph 捕获、acceptance 回归,一条都没跑过。**
- 没有 e2e 覆盖小词表路径(无公开裁剪 checkpoint)。

上机验收建议按这个顺序:先确认 §3.3 那行日志出现 → 首个 block 的 base logits / 每步修正后 logits / draft id / 映射后 target id 与 `specforge_sgl0514/DeepSpec/scripts/eval_real_dspark.py` 同输入逐项比对 → 最后才看 `acceptance_per_pos` 和端到端吞吐。**先保证首 block 逐项一致,再看接受率**,否则最终文本正常会掩盖中间 token-space 的错误。

---

## 6. 性能预期

只有采样段受影响,**backbone(5 层 Qwen3)、context KV 预计算、attention 一分钱不省**。V=248320 → K=32000,比值 7.76×:

| 每次 drafting | 全词表 | 小词表 |
|---|---|---|
| `lm_head` 权重读取 | 2.54 GB | 0.33 GB |
| `markov_w2` × N 步 | 127 MB × N | 16 MB × N |
| logits 加 bias + argmax 访存 | ~B·V·N | ~B·K·N |

按 H=5120、N=16、910B HBM ~1.6 TB/s **粗估**每次 drafting 省 2~3 ms,且这几项都是权重带宽主导,**batch 越小收益占比越大**。

⚠️ 以上是按带宽推的估算,**未实测**。而且如 §3.1 所述,单步变快不等于端到端变快。

---

## 7. 怎么用

启动命令与全词表 DSpark **完全一致**:

```bash
vllm serve <target> --speculative-config '{"method": "dspark", "model": <draft_path>, "num_speculative_tokens": 15}'
```

三个前提:

1. draft config **顶层**有 `draft_vocab_size`,且架构名是 `Qwen3DSparkModel`(不是 SpecForge 原始的 `DSparkDraftModel`);
2. 权重里有 `d2t`(`t2d` 不需要,vLLM 会丢弃;`lm_head` 不需要,没有就从 target 切);
3. 不要开 `enable_reduce_sample`;`draft_tensor_parallel_size` 填 1 或与 target 相同都支持。

⚠️ 别抄用户文档里 DSpark 示例的 `"enforce_eager": true`——`use_cuda_graph = runner._use_aclgraph() and not speculative_config.enforce_eager`,带上它等于**小模型完全不入图**。

---

## 8. 复审历程

两轮独立 review,共 6 条 finding。记录下来是因为**哪些不成立比哪些成立更值得留档**。

| 轮次 | finding | 结论 |
|---|---|---|
| 1 | ownership flags 不会被赋 False,新分支永不命中 | **不成立**(见 §4.1) |
| 1 | 同组限制误拒 target TP>1 + draft TP=1 | 成立,已修 |
| 1 | FP8 绕过"未量化"检查 | 成立,已修 |
| 1 | 全零 d2t 被当合法 identity mapping | 成立,已修 |
| 2 | 校验只跑在派生-head 分支 | 成立,已修 |
| 2 | `d2t.any()` 会误拒合法的 first-K 映射 | 成立,已修 |

两点方法上的收获:

- **第一轮的头号结论是错的,而且它错在一个"看起来很有道理"的推理上**——`process_eagle_weight` 确实只往 True 设,但它忽略了类属性默认值经 MRO 传递。破除它靠的不是再读一遍代码,而是问"如果这是真的,今天在跑的全词表 DSpark 为什么没坏?"**用既有事实去反推,比顺着代码再看一遍更有效。**
- **第二轮指出我照搬训练侧约定时搬错了张量**:SpecForge 判"是否装载"用的是全 False 的 `t2d`,我搬成了全零的 `d2t`,而后者有歧义。**跨仓库借用约定时,要连同"这个约定为什么无歧义"一起搬。**

---

## 9. 遗留

按风险排序:

1. **上机验证全部未做**(§5.2),这是唯一的阻塞项。
2. **`has_draft_id_mapping` 放在 ascend 侧是权宜。** 这个状态本属于 upstream vLLM 的 loader(它才是丢弃 `t2d`、skip 掉 `d2t` 的地方)。建议向 vLLM 提小 PR,ascend 侧的 `getattr` 兜底可无缝切换。
3. **"checkpoint 自带 head"分支缺正向可观测信号**(§3.3)。
4. **v2 runner 未评审。** upstream 已实现,但 ascend 侧的 `AscendDSparkSpeculator` 与之的交互没看过。
5. **DFlash 裁剪路径没有 checkpoint 验证过**,代码上是复用的。
