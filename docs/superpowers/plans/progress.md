# DFlash/DSpark 310P (Qwen3-8B) 开发进度

实施计划：[2026-07-22-dflash-dspark-310p-qwen3-8b.md](2026-07-22-dflash-dspark-310p-qwen3-8b.md)
分支：`dflash_dspark_310p_adapt_20260723`（基于 vllm-ascend `e4c88fb0b`，vllm `752a3a504`）

开发在 Mac 上进行，**验证全部在 310P 服务器上手工执行**。Mac 装不了 torch_npu，
`tests/ut/base.py` → `vllm_ascend/utils.py:34` 依赖它，所以本地跑不了任何单测；
Mac 侧只能做语法、行宽和 import 检查。

---

## 总览

| 阶段 | 状态 | 硬件需求 |
| --- | --- | --- |
| Task 1.1/1.2 无 Triton 输入展开 helper | ✅ 已验证 | CPU 单测 |
| Task 1.3a DFlash dispatch seam | ✅ 已验证 | CPU 单测 |
| Task 1.3b DSpark per-group dispatch | ✅ 已验证 | CPU 单测 |
| Task 2 五层逐层 context RoPE + patch gate | ✅ 已验证 | CPU 单测 |
| Task 3 ADN adapter + 精确路由 | ✅ 已验证 | CPU 单测 |
| Phase 0.3 输入展开算子门禁 | ✅ 已通过 | 310P 真机 |
| Phase 0.2 ADN 算子基线 | ✅ 40/40 通过 | 310P 真机 + ADN |
| Phase 0.4 ADN NZ 直读门禁 | ✅ 通过（含 TP=4 布局用例） | 310P 真机 + ADN |
| Task 4 Qwen3-8B eager E2E（DSpark-only，TP=4） | ✅ 通过（正确）；acceptance 偏低待查 | 310P 真机 |
| Task 6 target 入图 / drafter eager | 🟡 已实现，待真机 | 310P 真机 |
| Task 5 回归、文档、提交拆分 | ⬜ 未开始 | — |

---

## 已完成

### Task 1.1 + 1.2：无 Triton 输入展开 helper

提交：`1707eb90c`，修正 `585d6d6aa`
验证：**7 passed**（服务器）

新增 `vllm_ascend/_310p/spec_decode/parallel_drafting_inputs.py`，是
`copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid` 的向量化 PyTorch 等价实现。
原地写调用方的持久 buffer，全程无 `.cpu()`/`.item()`/`.tolist()`，不 import torch_npu/triton。

测试 golden 是原 Triton kernel 的**逐行转写**而非第二个向量化实现——对不上时能定位到原 kernel 的
具体哪一行。覆盖：DFlash K+1 / DSpark K 两种布局、ragged 调度段（`ctx_lens != seq_lens`，
绝对 position）、有/无 rejected tail、query slot 跨 127/128/129 页边界、乱序非连续 physical page id。
所有输出 buffer 多分配 8 个 `-99` 哨兵位，越界写会被抓到。

首次验证 6 passed / 1 failed，修正见下方「踩过的坑」。

### Task 1.3a：DFlash dispatch seam

提交：`2b9e85096`
验证：**15 passed**（7 原有 + 8 新增）

- `parallel_drafting_inputs.py` 新增 `ADN_BLOCK_SIZE = 128` 与 `resolve_310p_block_size()`；
- `dflash_proposer.py` 新增 `_expand_drafting_inputs()` 共享方法（DSpark 后续继承），
  `set_inputs_first_pass` 原来的裸 kernel 调用改为走它。

**block size 的读取放在 dispatch 内部**：若放在外面，Triton 路径也会收到 selected BlockTable size，
A2/A3 换个 cache block 配置就会被 `!= 128` 的 guard 打挂。现在 310P 分支自己调
`resolve_310p_block_size()`，Triton 分支原样用调用方传来的 `self.kernel_block_size`。

三个 block size 来源各自对 `ADN_BLOCK_SIZE` 比，不互相比——三者一致只能证明彼此相同，
证明不了等于 ADN 验证过的值。`runner.kernel_block_sizes[gid]` 是候选列表，用 `in` 而非 `==`。

测试驱动**真实的** `set_inputs_first_pass` 而非直接调 seam；`_ExplodingLauncher` 正反两用
（310P 不得触达、非 310P 必须触达）；buffer 参数用 `assertIs` 断身份而非比值。

### Task 1.3b：DSpark per-group dispatch

提交：`740275779`，fixture 修正 `e75061925`
验证：**20 passed**（服务器）

DSpark 复用继承来的 `_expand_drafting_inputs`，不重复定义。它与 DFlash 的差异全部保留：
按 KV cache group 循环、per-group buffers、query 数为 K（非 K+1）、`sample_from_anchor=True`。

新增单 group scope guard（仅 310P 生效）：Qwen3-8B 的 5 层同规格会被 vLLM 并成 1 个 group，
多 group 机器是给 DeepSeek-V4 准备的，本期显式拒绝而不是静默只展开其中一个。

`kv_block_size` 那行**没有**加 128 断言——它跑在所有设备上，pin 死会打挂 A2/A3 的合法配置。
310P 的 pin 在 dispatch 内部的 `resolve_310p_block_size()` 里。

删掉了 `dspark_proposer.py` 已不再使用的 Triton import。

### Task 2：五层逐层 context RoPE + patch gate

提交：`924401ff4`
验证：**30 passed**（服务器）

三处改动：

1. **`patch/worker/__init__.py`** — `patch_qwen3_dflash` 移出 `if not is_310p()`。
   `patch_qwen3_5` / `patch_qwen3vl` 仍保持 310P gate，不扩大模型范围。
   DSpark 不需要 worker patch：模型侧 `Qwen3DSparkModel(DFlashQwen3Model)` 继承同一实现，
   mask token 走 platform 层的 `patch_speculative_config`（见 §0.1）。

2. **`patch_qwen3_dflash.py`** — 抽出 `apply_context_rope()` 做设备分支：
   - 310P 逐层旋转，每层只送 `num_ctx` 个 position，结果写回 `all_k_normed`；
   - 非 310P 保持一次 fused `[L * num_ctx]` 调用。

   两条路径都**接住返回值**。核实过：A2/A3 走 `rope_forward_oot`，其 `else` 分支对已连续的
   张量调 `.contiguous()` 返回同一对象、原地改再返回该存储，所以接住返回值是 no-op；
   310P 走 `npu_apply_rotary_pos_emb`，返回**新张量**，不接住就会把未旋转的 K 写进 cache。
   后者是"数值错但不报错"的故障。

   310P 用 `layers[i].self_attn.rotary_emb` 而非 `layers[0]` 的——各层 RoPE 参数本就要求一致
   （`_build_fused_kv_buffers` 有断言），逐层取不依赖该断言且零代价。

3. **两个 proposer 的 `dummy_run`** — `is_profile` 分支整体包进 `_profile_rope_context()`
   （定义在 dflash，DSpark 继承）。这是计划里的 P0：profile 路径绕过 `_run_merged_draft`，
   drafting flag 从未打开，**两个** forward（context KV 预写 + 紧随的 query forward）都会读到
   target dummy run 遗留的 cos/sin slice。故障形态是"用错值"不是崩溃。

测试要点：profile 用例驱动**真实**的 `dummy_run(is_profile=True)`，并让 context 与 query
长度不同（12 vs 18 / 14），这样"只覆盖第一个 forward"的半修复会被抓住。异常路径用例强制
`is_310p=True`，否则 context manager 直接 yield，断言恒真。

### Task 3：ADN adapter + 精确路由

提交：`fe027c4a6`，测试修正 `409207da0`，causal 路径回归修复 `71643f640`
验证：**tests/ut/_310p/ 全绿**（53 条）

新增 `_310p/attention/adn_fused_infer_attention.py`：

- `load_adn()` 惰性导入并缓存。`adn_custom_ops` 的包 `__init__` 顶层 import torchair，
  急加载会让 torchair 成为每次 310P 运行的硬依赖。导入失败时**明确不回退**——
  落到 causal splitfuse 会返回看似合理的错数值；
- `validate_adn_scope()` 锁死本期唯一验证过的配置：method ∈ {dflash:K8, dspark:K7}、
  draft architecture、`enforce_eager`、TP=2、`(Nq,Nkv,D)==(16,4,128)`、FP16、
  rank-4 NZ、`get_npu_format == ACL_FORMAT_FRACTAL_NZ`、cache 物理 block == 128。
  首次调用后置 flag 缓存，不进热路径；
- `forward_parallel_draft_adn()` 每步动态校验 q-len/kv-len/block table，然后固定
  `attn_mask=None`、`inner_precise=2`、`force_call=False`、`input_layout="TND"`。

**block size 一律对 `ADN_BLOCK_SIZE` 常量比，不从 cache 反推**。上一版计划里
`block_size = key_cache.shape[-2]` 之后又断言 `key_cache.shape[-2] != block_size` 是恒真检查。

**返回值严格比 `query_tnd.shape`**，不只比 numel——numel 相同会放过转置或错头的结果。

路由加在 `forward_impl` 最前面：`draft + dflash/dspark + ChunkedPrefill + non-causal` → ADN；
**其他任何 `causal=False` 抛 `NotImplementedError`**，不静默落到 causal splitfuse。
guard 只在 310P 的 `forward_impl`，不上移到共享层（会打挂 A2/A3 的 non-causal 路径）。

顺带清掉 `forward_chunked_prefill_310` 里一个冗余的函数内 `_EXTRA_CTX` import
（已确认 `ascend_forward_context` 不依赖任何 attention 模块，无循环导入风险，
基类 attention_v1 也是模块级导入它）。

### Phase 0：硬件门禁脚本

提交：`ea552d8b6`，fixture 修正待验证
状态：**0.3 通过；0.4 前 3 个数值用例通过，后 2 个因 fixture bug 未跑到**

已得到的实质结论：

| 用例 | max_abs | mean_abs |
| --- | --- | --- |
| DFlash q=9，单请求 2 页 | 0.000657 | 0.000065 |
| DFlash q=9，ragged 1/2/3 页 | 0.001195 | 0.000078 |
| DSpark q=7，ragged 1/2/3 页 | 0.001195 | 0.000079 |

**ADN 能直读 vLLM 分配并写入的 NZ cache，误差在 fp16 分辨率量级（~1.2e-3）。**
adapter"直接传 cache、不 gather、不转 ND"的地基假设成立。

环境：CANN OPP 9.1.0-beta.1、torch 2.10.0+cpu、torch_npu 2.10.0.post2、
vllm 0.25.1（= `752a3a504`，与工作区和 `Dockerfile.310p` 的 `VLLM_TAG` 完全一致，无版本漂移）。

两个手工门禁脚本（`smoke_` 前缀，pytest 默认只收集 `test_*.py`，不会被 CI 误抓）：

`tests/e2e/_310p/spec_decode/smoke_parallel_drafting_inputs.py`
: 同一个 helper 在 NPU 和 CPU 各跑一遍逐字段比对。CPU 单测已证明它等价于 Triton kernel，
  这里证明的是**它用到的每个算子在 310P 上真的能跑**（computed-index 高级索引、rank-2
  gather、整除取模、broadcast、列赋值）。抛异常＝算子不支持，要转小型 AscendC helper；
  静默不一致更糟，说明算子行为不同。

`tests/e2e/_310p/adn/smoke_adn_nz_readback.py`
: 验证 ADN 能直读 vLLM 已分配已写入的 NZ cache。这是整个 adapter 的地基假设。
  - 用 `torch_npu.empty_with_format` **分开分配两个 rank-4 cache**，复刻
    `model_runner_310p.py:845-850`。分配一个 5D 再切片得到的 storage descriptor 不同，
    而那恰恰是本门禁要验的东西；
  - 用 `DeviceOperator.reshape_and_cache` 写入，走生产 dispatch；
  - slot 由 block table 的 **CPU 镜像**算出，避免逐 token `int(npu_tensor)` 同步；
  - **scale 用 `head_dim ** -0.5`**，不是 ATK 的 `1/head_dim`（`fia_common.py:508`）——
    ATK 自洽但跑的是生产永远见不到的数值区间；
  - block table 用打乱且非连续的物理页，并断言无不动点（逻辑页 i → 物理页 i 会让
    "忽略 block table"的实现也通过）；
  - future-token dominance 不用幅值阈值，而是**同时算 causal 和 non-causal 两个 golden**，
    断言输出匹配后者、不匹配前者，并先断言两者本身有显著差异——否则这条用例是空的。

**判据已确定，不再是占位值。** Ascend_Ops 更新后移除了 `atk_test/`，改为
`Ascend_Ops/tests/test_adn_fia.py`，其中 `:172` 定义 `atol = 1e-4` 且
`passed = diff_flatten_mean <= atol`——**按平均绝对误差判定**，max 只打印。
smoke 脚本已改用同一判据（`MEAN_ATOL = 1e-4`）。

按此回算已有实测：mean_abs 分别是 0.000065 / 0.000078 / 0.000079，**三条全部通过，
余量约 1.3 倍**。

另外两点已随更新解决：

- `test_adn_fia.py:173` 用 `scale = head_dim ** -0.5`，与生产一致，旧 ATK 的
  `1/head_dim` 坑不复存在；
- ADN 的 Python ABI 精简了（移除全部 quant/dequant/antiquant 与 `kv_padding_size`）。
  本期 adapter 全程只用关键字实参且从未传过这些，**核对后无需改动**。

---

### Task 4：Qwen3-8B DSpark TP=4 eager E2E

提交：待填
状态：**已写，待真机**

`tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_eager_310p.py`。
只测 DSpark（K=7）+ TP=4，配置：`dtype=float16`、`block_size=128`、`enforce_eager=True`、
`distributed_executor_backend="mp"`、`enable_prefix_caching=False`，顶部 `os.environ.setdefault
("VLLM_USE_V2_MODEL_RUNNER","0")` 锁 MRV1。

两组断言：

1. **正确性（token 一致）** — greedy 投机解码是无损的，所以开投机的输出 token 必须与不开投机的
   baseline **逐 token 相同**。不一致即 draft/verify/reject 环路有数值 bug。这是最硬的正确性门。
2. **投机确实在发生且在拒绝** — `num_drafts > 0`、`total_accepted > 0`、
   **`total_accepted < num_drafts * K`**。第三条关键：光有 token 一致，在"每个 draft 都被拒、
   静默退化成纯 target 解码"时也会通过；加上界才证明真的在接受又在拒绝。

**不**拿 acceptance 去比 A2/A3 的 golden `[1.0,0.8,0.6,...]`——那是 TP=1/graph 模式在 A2/A3 上测的，
不是 310P eager。首轮只打印 310P eager 的 acceptance_per_pos 记录，断言只卡结构性质。

两次 8B 加载（baseline + DSpark）各用独立 `with` 上下文，退出即释放。

已知延后项：revision 未 pin SHA（沿用仓库自身 dspark 测试的惯例，不 pin）；exact-token-id 页边界
prompt 未加（页边界已在 Phase 0.4 NZ 门禁算子层覆盖）。

---

## scope 调整：放开 TP（2026-07-24）

原设计把 ADN scope 锁死在 **TP=2 + 硬编码 (Nq,Nkv)=(16,4)**。实机要用 **TP=4**，据此放开：

- **TP 不再校验。** TP 只是把 head 切到各 rank，每个 rank 跑相同的 attention，不改数值。
  真正要管的是 per-rank head 布局；
- **head 布局改为结构约束**，不硬编码具体数：`0 < head_dim <= 256`、
  `head_dim * block <= 16384`（block=128 时自然把 head_dim 卡在 <=128）、`Nq % Nkv == 0`、
  `Nq/Nkv <= 64`、`Nkv * head_dim` 16 对齐。TP=2(16/4)、TP=4(8/2)、TP=8(4/1) 都自然通过；
- 这也更稳：drafter 的确切 head 数由 checkpoint 决定，结构约束不依赖我猜数字。

`validate_adn_scope` 相应重写，单测把"拒绝 TP=4 / 拒绝 Nq=16 外的布局"两条换成
"TP=4 布局放行 + 非法 GQA/非 16 对齐拒绝"。NZ 直读门禁参数化 head 数，新增一个
**TP=4 布局(8/2, NZ dim1=16)** 用例——因为 TP=4 是实机实际要跑的，dim1 从 32 变 16，
给 writer/reader 路径在该维度一个真机证据，而不是假设它和 dim1=32 一样。

**Task 4 随之收窄**：实机只有 DSpark checkpoint，只测 DSpark（K=7）+ TP=4。DFlash 端到端延后
（无 checkpoint），但其 q=9/skip-anchor 布局有 CPU 单测 + Phase 0.4 的 DFlash q=9 门禁覆盖。

---

## 下一步

**在 310P 真机按顺序执行 Phase 0。**

**0.1 记录版本矩阵**（✅ 已完成，见上）。注意 `version.cfg` 在本容器布局下不存在，
用 `opp/version.info` 代替——它给的是 OPP（算子包）版本，对 ADN 这个 custom OPP 反而更贴题：

```bash
python -c "import torch, torch_npu, vllm, vllm_ascend; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('vllm', vllm.__version__)"
cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
python -c "import adn_custom_ops, torchair; print('ADN + torchair ok')"
```

最后一条失败就先按 `Ascend_Ops/AGENTS.md` 编译安装 custom_opp 和 PTA，其余步骤都依赖它。

**0.2 跑 ATK 基线并抄出真实判据**：

```bash
cd $ASCEND_OPS/atk_test
atk case -f op_fia_tnd_nocausal_hd128_bs128.yaml -p . -dt 1
atk task -c result/op_fia_tnd_nocausal_hd128_bs128/json/all_op_fia_tnd_nocausal_hd128_bs128.json -n nodes.yaml -p . -sp
```

从结果 JSON / 框架配置里找到 FP16 default 标准的实际数值，替换
`smoke_adn_nz_readback.py` 顶部的 `ATOL`/`RTOL` 占位值。

**0.3 输入展开算子门禁**（不需要 ADN，可以先跑）：

```bash
python tests/e2e/_310p/spec_decode/smoke_parallel_drafting_inputs.py
```

**0.4 NZ writer → ADN 直通**（整个 adapter 的地基）：

```bash
python tests/e2e/_310p/adn/smoke_adn_nz_readback.py
```

失败时按这个顺序查：descriptor/shape 报错说明 NZ 分配或 writer 布局与 ADN 不一致；
形状对但数值错，通常是 `num_key_value_heads`、`block_size` 或 scale。
**不要在热路径加 repack 绕过去**——那会让"直读"这个前提悄悄失效。

**0.4 跑不通的话 Task 3 的"直接传 cache 不做 gather"前提就不成立，adapter 要重写。**
所以在它变绿之前不要开 Task 4 的 E2E。

---

## 验证命令

每步完成后在 310P 服务器执行：

```bash
git pull && TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/_310p/spec_decode/test_parallel_drafting_inputs_310p.py
```

回归（应全绿）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q tests/ut/_310p/
```

⚠️ **不要**用 `pytest tests/ut/spec_decode/` 做回归——见下方「已知既有失败」。

---

## 已知既有失败（与本期改动无关）

`tests/ut/spec_decode/a2/test_eagle_proposer.py` 在 310P 机器上有 **9 条固定失败**，
已通过回退 `dflash_proposer.py` 到 `f9a5d8df8` 复现同样的 9 条确认与本期无关。

两类根因：

1. **无 Triton（2 条）** — `llm_base_proposer.py:1949` 的 `prepare_inputs_padded_kernel[grid]`。
   无 Triton 时 `@triton.jit` 退化为恒等装饰器，kernel 成了普通函数，下标访问即
   `TypeError: 'function' object is not subscriptable`。
2. **`patch_idex_310` 的 `super()` 绑定（4 条 + 相关 3 条）** —
   `patch_idex_310.py` 把 `AscendSpecDecodeBaseProposer310.set_inputs_first_pass` 这个函数对象
   赋给基类 `AscendSpecDecodeBaseProposer`。方法体里的零参 `super()` 靠
   `__class__ = AscendSpecDecodeBaseProposer310` 解析，而 Eagle proposer 不是它的实例 →
   `TypeError: super(type, obj): obj must be an instance or subtype of type`
   （`llm_base_proposer_310.py:107`）。同源问题还导致 `assert long_seq_args is None` 失败
   （310 版本返回 `(None, None)`）。

`tests/ut/spec_decode/a2/` 没有任何平台 gate，而 `TestBase.__init__` 无条件
`adapt_patch()`，所以 310P 机器上这个 patch 必然生效。

**这是既有缺陷，不在本期范围。** 第 2 类值得单独修，但应另开分支，不要混进本条线。

---

## compressed mask 升级后的复核（2026-07-24）

升级 torch_npu + ATB 以启用 compressed mask 后的检查结论。

### 诊断成立

`is_compressed_mask_supported()`（`attention_mask.py:28`）是 hasattr 判定：
`_npu_flash_attention_v3` 且 `_npu_paged_attention_splitfuse_v2`。升级后转 True，路径切到
compressed 分支。**旧路径确实无法入图**——`get_splitfuse_mask:90-94` 每次调用做两次
`.to("cpu")` 加 `.tolist()`，是实打实的设备同步；compressed 版用缓存的固定 2048×2048 mask，
无同步。

### 既有代码已经是 compressed-aware，无需改动

- `get_attention_mask:139` 用 `COMPRESSED_MASK_SEQ_LEN` 而非 `max_seqlen`，prefill 的 mask 尺寸自洽；
- `_flash_attention` 与 `forward_chunked_prefill_310` 都有完整的双分支。

### 我的 seq_len 修复与 v3 兼容

两个算子共用同一契约：**非投机时 builder 本来就给 CPU 张量**（`attention_v1.py:283-291` 优先
`_seq_lens_cpu`），所以 host 是既有约定，我的修复只是让投机路径与之一致，不是新引入的要求。

### 实际发现并修的问题：compressed 路径零测试覆盖

`tests/ut/_310p/attention/test_attention_v1_310.py` 里**四处**都把
`self.impl.support_compressed_mask = False` 钉死，`_npu_flash_attention_v3` 和 `splitfuse_v2`
**一条测试都没有**——而升级后生产跑的正是这条路径。补了两条：

1. v3 也必须收到 host seq_len（与非 compressed 分支同契约）；
2. splitfuse 走 v2 + 固定方阵 mask，且**断言 legacy `get_splitfuse_mask` 未被调用**——
   后者带同步，是入图的直接障碍。

### 图友好性：qlens buffer 的设计正好对上

`_fill_query_lens_cpu:95-99`：非 drafting 用**持久 pinned buffer**（地址稳定，
`torch.sub(..., out=buffer)` 原地更新）→ 捕获/重放友好；drafting 才 `.clone()`（地址不稳）。
这与「target 入图 / drafter eager」的分工天然吻合，无需改动。

### 需要留意（未改，但值得知道）

`forward_chunked_prefill_310:283` 的 `seq_lens.to(device=...)`：若 seq_lens 在 CPU 会**每次新建
device 张量**，地址不稳。investigation 结论是 parallel_drafting 下 seq_lens 本就在 device，
这里是 no-op，所以当前无影响——但若将来 seq_lens 的来源变了，这是第一个要查的点。

---

## cudagraph_capture_sizes 的正确取值

`gpu_model_runner.py:842`：`uniform_decode_query_len = 1 + num_spec_tokens`
→ DSpark K=7 时是 **8，不是 7**。

`:3829-3830` 的 uniform decode 判定：
```python
(max_num_scheduled_tokens == uniform_decode_query_len)
and (num_tokens == max_num_scheduled_tokens * num_reqs)
```
**capture size 是 token 数**，所以必须是 `uniform_decode_query_len` 的整数倍：
batch 1..N → `[8, 16, 24, ...]`。

**为什么会想到 7**：A2/A3 的 dspark 测试用 `cudagraph_capture_sizes=[7, 8]` + **PIECEWISE**。
PIECEWISE 下 **drafter 也被捕获**，而 DSpark drafter 每请求正好 K=7 个 query token，所以 7、8 都要。
我们是 FULL_DECODE_ONLY + drafter eager，drafter 永不入图，**7 用不上**。

**修掉的真 bug**：图模式 E2E 原本写死 `[8, 16]`（batch 1、2），但测试有 3 条 prompt，
`max_num_seqs=256` 会把它们批到一起 = **24 tokens**，两个 size 都不匹配 → **静默回退 eager**，
图根本没被验证，测试会"通过"但什么也没证明。改为按 prompt 数推导
`[UNIFORM_DECODE_QUERY_LEN * n for n in 1..len(PROMPTS)]`。

---

## 图模式选择的证据链（PR #11765 复核）

**结论：继续用 `FULL_DECODE_ONLY`。**

PR #11765 就是本基线里的 `41ff81e1a [Feature] Add qwen/glm dspark for mrv1`（2026-07-13）——
它是 **DSpark MRV1 的 PR，不是 310P 图模式的 PR**。而且它自己的性能表两行都写着
`FULL_DECODE_ONLY`（DFlash 4.73 / DSpark 6.30），所以"只支持 eager 和 piecewise"的读法不成立。

**时间线上有个值得知道的细节**：#11765 当时 DSpark **没有**关掉自己的图（继承基类的
`use_cuda_graph = runner._use_aclgraph() and not spec.enforce_eager`），所以那次 6.30 的 benchmark
里 drafter 很可能也在图中。一周后的 `8fe122d95 [Feature][Refactor] DSv4 DSpark (#11431)`
（2026-07-21）才加上 `use_cuda_graph = False`，注释 "Ascend cudagraph unsupported on this path"。

**对我们无影响**：ADN 的 `SymInt[]` 长度参数无论如何都要求 drafter 保持 eager，而当前基线已经
强制了这一点。所以「target FULL_DECODE_ONLY + drafter eager」既是当前代码的既成行为，
也是 ADN 约束下唯一可行的组合。

---

## Task 6：target 入图 + drafter eager（已实现，待真机）

按调查结论实现。**纠正一个前提**：310P 支持的是 **`FULL_DECODE_ONLY`，不是 PIECEWISE**
（`docs/source/locale/.../310p.po` 的 Graph Mode Notes）。同一段还有个对我们直接相关的警告：

> 当启用多个 TP 时，**可捕获的图数量受限**，取决于模型深度（Qwen3-32B 只能捕获重放 **2 个图**）；
> TP=1 时无此限制。原因是**硬件 event-id 资源**。

我们是 TP=4 + 36 层，所以 E2E 的 `cudagraph_capture_sizes` 只给 `[8, 16]`。捕获直接失败的话，
这个限制是第一嫌疑。

### 改动

**把「整机 eager」的检查换成「ADN 不得被捕获」的逐次检查。** 原来的
`if not model_config.enforce_eager: raise` 会误杀 target 入图这个完全合法的配置——它检查的是
引擎全局，而真正的约束只针对 ADN 自己。新判据：

```python
if is_forward_context_available() and _EXTRA_CTX.capturing:
    raise RuntimeError("ADN draft attention was reached during ACLGraph capture...")
```

`is_forward_context_available()` 的短路是必需的——直接读 `_EXTRA_CTX` 在无 forward context 时会抛
"Forward context is not set"，那正是之前 `test_forward_mtp_310` 被我打挂的同一个坑。

**为什么这个判据更好**：它不依赖配置推断，而是在 ADN 真的被捕获的那一刻拦下。将来谁把 drafter
翻进图模式，会在这里响，而不是产出静默错误的数值。

**分工无需额外代码**：proposer 的 `use_cuda_graph` 独立于 runner，且 DSpark 已在
`dspark_proposer.py:57` 把自己钉成 False。

### 310P 已有的 spec-decode 图基建（不用重造）

- `model_runner_310p.py:638` `is_spec_graph_capture` —— "All the spec decoding cases has to run
  splitfuse op on 310P"
- `:692` `update_before_replay` —— spec decode 在 replay 前更新 full graph 参数
- `temporary_modify_uniform_decode_query_len` —— 目前只对 ngram 生效

### 单测

capture guard 三条：capturing=True 必须拒绝、capturing=False（eager 与 **replay** 两种情形）
必须放行、无 forward context 时不得读 `_EXTRA_CTX`（用"一读就炸"的假对象锁住）。
原来那条"graph mode 必须被拒绝"翻转成"必须放行"。

---

## ACLGraph 适配点调查（2026-07-24）

结论：**DSpark drafter 入图被 ADN 的算子 ABI 硬阻塞**，不是 vllm-ascend 侧能绕过的。
可行的是「target 入图 + drafter 保持 eager」。

### 阻塞一（硬，需 Ascend_Ops 改算子）：长度参数是 `SymInt[]`，不是 tensor

`custom_pta/csrc/registration.cpp:56-57`：
```
SymInt[]? actual_seq_lengths_q=None
SymInt[]? actual_seq_lengths_kv=None
```
Python list 在 graph capture 时会被**固化成常量**。而 `actual_seq_lengths_kv` **每个 decode step
都在变**（KV 在增长）——replay 时会按捕获时的旧长度做 attention，**静默算错**，不是崩。

A2/A3 的 FIA 有 tensor 形态可用（`attention_v1.py:599,806` 的 `actual_seq_kvlen=seq_lens` 传的是
**张量**，静态地址、内容原地更新），所以它能入图。**ADN 没有 tensor 重载**，
必须由 Ascend_Ops 侧新增，vllm-ascend 无法绕过。

### 阻塞二（中，需 Ascend_Ops 改算子）：输出每次新分配

`custom_pta/csrc/adn_fused_infer_attention.cpp:26`：
`at::empty_symint(query.sym_sizes(), query.options())` —— 无 `out=` 变体。
图捕获要求输出地址静态。计划 §5.3 早就把「加 `out=`」列为 P1，这里变成入图的前置条件。

### 阻塞三（上游决策）：DSpark 自己关掉了图

`dspark_proposer.py:57`：`self.use_cuda_graph = False`，注释写明
"DSpark runs eager only (Ascend cudagraph unsupported on this path)"。
**这是上游 vllm-ascend 的决定，不是 310P 特有**——A2/A3 上 DSpark 同样跑 eager。

### 可行退路：target 入图，drafter 保持 eager

`llm_base_proposer.py:208`：`use_cuda_graph = runner._use_aclgraph() and not spec.enforce_eager`，
DSpark 在 :57 覆盖为 False。两者独立 → **runner（target）可以走 ACLGraph，drafter 留在 eager**。
这正是 DSpark 在 A2/A3 上的现状。target 是 36 层 8B，drafter 只有 5 层，收益大头在 target。

310P 本身有 ACLGraph 基建（`model_runner_310p.py:70-75` 的 capture/replay），但**所有既有 310P
E2E 都是 `enforce_eager=True`**，没有先例，需要单独验证 ATB 算子在 capture 下的行为。

### 我这边只需一行

`adn_fused_infer_attention.py:100` 的 `enforce_eager` 检查——上面三条解决前它应该留着。

---

## Task 4 通过：DSpark 在 310P 端到端正确

**第四次真机**（`_pass_0724_1650`）：**1 passed**。
```
num_drafts=61  total_accepted=135
acceptance_per_pos=[0.80, 0.46, 0.38, 0.30, 0.11, 0.11, 0.05]
exact token match: 3/3 prompts
```
**正确性确认**：3/3 完全一致（上一轮 prompt 2 分岔是 run-to-run 的 fp 抖动，正印证 token
一致性不是硬门）。代码层面的适配到此功能完整且正确。

**遗留观察（效率，非正确性）**：acceptance 低于 A2/A3。

⚠️ **参照修正**：先前拿 `tests/.../utils.py` 的 `BASELINES["dspark"]` 比是错的——那是测试的
**宽松阈值**，不是真实性能。真实数字在 PR #11765（`41ff81e1a`）的性能表里：

| | Acc Length | Acc Rate per Position |
| --- | --- | --- |
| A2/A3 DSpark（gsm8k 100 req，output 2048） | **6.30** | `[0.94, 0.87, 0.81, 0.74, 0.69, 0.63, 0.59]` |
| 本次 310P（3 条短 prompt，64 token） | ~2.2 | `[0.80, 0.46, 0.38, 0.30, 0.11, 0.11, 0.05]` |

差距比先前估计的**更大**。但两者**不可直接比**：数据集、输出长度差异很大，短生成的前几个 token
本来就更难猜。pos-1 从 0.87 掉到 0.46 仍值得查。两种可能：(a) ADN vs FIA、逐层 RoPE vs fused
RoPE 的 fp 差异累积；(b) draft 管线某环节细微偏差。**不阻塞功能，列为后续调查项**；
要下结论应先在同数据集同输出长度下复测。

---

## 真机进展：整条 DSpark 链在 310P 跑通，token 一致性门选错了

**第三次真机**（`_07241640`）：**没有崩溃**。模型加载（含 drafter 4.42 GiB）、KV cache 分配、
baseline（3/3, 34.67 tok/s）、DSpark spec **全部跑完**。失败在我自己的 token 一致性断言：
prompt 0/1 完全一致，prompt 2 在第 32 个 token 处分岔。

**这条断言选错了。** greedy 投机解码只在**精确算术**下无损；真机上不是 bit-identical，因为
target 投机时一次 verify K+1 个 token（chunked-prefill 形状的 batch），非投机时一次 decode 1 个，
浮点累加顺序不同 → 边界 logit 的 argmax 翻转 → 从翻转点起序列分岔。仓库自己的 `test_dspark.py`
就是查 acceptance 而非 token 一致性，同一个原因。

**反直觉但决定性的一点**：draft 如果是坏的，输出反而会和 baseline **一致**（全被拒 → 纯 target
解码）；只有 draft 在工作、多 token 被接受，才会因 verify batch 形状不同而分岔。**所以 prompt 2
的分岔恰恰证明 draft 在正常工作。** 而且即便真有 bug，accepted token 永远是 target 的 argmax
（draft 只提议），分岔最多指向 verify/reject（上游共享），不指向我改的 ADN draft attention。

**改法**：正确性改判 acceptance——先算先打印，再断言 `num_drafts>0`、`0<accepted<max`、
**pos-0 acceptance >= 0.5**（DSpark 的 pos-0 是 anchor=target bonus token，draft 管线正确时几乎
必接受，塌了就是 draft 坏的签名，而这正是 token 一致性抓不到的），外加"至少一个 prompt 完全一致"
（systematically 坏的 verify 会污染所有 prompt）。token 分歧位置降级为记录。

**这不是放宽让它过**，是换用投机解码正确性的行业标准信号。复跑还能第一次看到 310P 的
acceptance 曲线——那才是"我的适配对不对"的真信号。

---

## 真机踩坑（二）：scope guard 检查了错误的架构名层级

**第二次真机**（`_07241620`）：prefill 段错误已消失，baseline 成功（3/3, 32.66 tok/s），
draft attention **成功路由到 ADN**（`forward_parallel_draft_adn`）——证明整条投机链已打通。
崩在 `validate_adn_scope`，且是一条干净可操作的报错（scope guard 本该如此）：
```
RuntimeError: draft architecture Qwen3DSparkModel is outside this scope
(['DFlashQwen3ForCausalLM', 'Qwen3DSparkForCausalLM'])
```

**根因**：我把白名单写成了 vLLM 的**类名**，而 `hf_config.architectures` 存的是 checkpoint
`config.json` 里的 **architecture 字符串（= vLLM registry 的 key）**。registry
（`vllm/model_executor/models/registry.py:595,597`）映射：
```
"DFlashDraftModel"  -> DFlashQwen3ForCausalLM
"Qwen3DSparkModel"  -> Qwen3DSparkForCausalLM
```
config 写的是 key，我匹配的是 value，永远不中。改成匹配 config key
`{"DFlashDraftModel", "Qwen3DSparkModel"}`。

**这次值得记的**：scope guard 起作用了——它没让错配置静默跑，而是清楚地把实际架构名打了出来，
一眼定位。防御式校验的价值正在于此：报错本身就是修复线索。

---

## 真机踩坑：parallel_drafting 把 seq_lens 变 device，310P prefill 段错误

**首次真机 E2E 崩溃**（`atb_195291`）：
```
[param.cpp:78] tensor.hostData is null
[self_attention_encoder_fuison_ops_runner_910a.cpp:218] build param from host tensor fail
```
segfault，不是 Python 异常。"910a" 只是 ATB 这个 SelfAttention 融合算子的 kernel 家族名，
310P/300I-DUO 共用这套 runner，**不代表跑到了 910 硬件**。

**根因**：baseline（无投机）成功（3/3, 30 tok/s），只有 DSpark 崩。基类
`AscendMetadata.build`（`attention_v1.py:299`）有一条 `parallel_drafting` 分支把 `seq_lens`
设成 **device 张量**（为 A2/A3 的 FIA 加的，那边 FIA 要 device）。DSpark 的
`parallel_drafting=True` 让**整个引擎**（含 target 的 prefill）都拿到 device seq_lens。而 310P 的
`forward_prefill_310` 把它直接当 host 传给 ATB SelfAttention encoder → `hostData is null` → 段错误。

有意思的是两个 310P 路径要求相反：`forward_paged_attention`（decode）显式把 seq_lens 搬到
**device**，`forward_prefill_310`（prefill）需要 **host**。非投机时基类给 CPU，prefill 正好；
投机翻成 device，prefill 就炸。**DSpark 在 310P 跑是第一次，这个潜伏不兼容第一次被触发。**

**修法**：`forward_prefill_310` 里把 seq_len 强制到 host（`.cpu()`），对称于 decode 的强制到
device——每条路各自把 seq_lens 钉到自己算子需要的位置，不依赖基类 builder 的猜测。附一条 CPU
单测：mock 一个 device seq_lens，断言传给 `_npu_flash_attention` 的是 host 张量。

**排查心得**：`EngineDeadError` / shm `RuntimeError: cancelled` 全是 worker 死后的包装，
根因在**独立的 atb_*.log**（segfault 不进 Python 栈）。以及先确认 baseline 成没成——它成了
就把范围从"310P prefill 普遍坏"缩到"DSpark 特有"，直接指向 parallel_drafting。

---

## 踩过的坑

### block table fixture 的不动点（`585d6d6aa`）

首次验证 `test_fixtures_are_not_degenerate` 失败。原因：fixture 用
`torch.arange(n).flip(0)` 造"乱序"physical page id，但**奇数长度的反转有不动点**——
`arange(3).flip(0) == [2, 1, 0]`，第 1 列仍映射到第 1 页。`seq_lens=[129]` 时整个 query block
恰好落在那一列，于是每个 slot 都等于它的 cache position，那条"非退化"断言正确地响了。

修正：整体 `+ max_blocks` 把物理页号移出逻辑索引区间，并加断言保证没有任何逻辑页映射到同号物理页。

值得记的是：**这条断言本身就是为了防止其余断言变成摆设而写的，它第一次运行就抓到了问题。**
如果没有它，一个完全忽略 block table 的实现也能让其余用例全绿。

### stub 属性名靠猜（同类错误已发生三次）

1. 计划里把 `_context_slot_mapping_buffers` 写成单数 `_context_slot_mapping_buffer`（review 抓到）；
2. Task 1.3b 的 DSpark stub 设了 `_per_group_block_table_buffers`，但
   `set_inputs_first_pass:212-214` 每次调用都会**从 `_per_group_block_tables` 重建它**，
   所以写进去的东西在被读之前就被丢弃了 → 5 条 `AttributeError`。

根因都是同一个：**按邻近命名推断属性名，而不是读代码确认**。尤其 DSpark 里同时存在
"源 dict"和"派生 dict"两套同前缀的属性时，stub 必须打在源头。

固定做法——写 stub 前先列出目标方法实际读的每个属性：

```bash
awk '/    def set_inputs_first_pass/,/    @torch.inference_mode/' \
    vllm_ascend/spec_decode/dspark_proposer.py | grep -oE "self\.[_a-zA-Z0-9]+" | sort -u
```

再和 stub 逐项比对。被方法**写入**的属性（如 `_dflash_num_context`）不需要 stub。

### 在共享路径顶部读 `_EXTRA_CTX` 引入了新前置要求

Task 3 最初把路由条件写成一个大 `and` 表达式放在 `forward_impl` 顶部，其中
`_EXTRA_CTX.is_draft_model` 是第一个条件。读它需要活跃的 forward context，于是
**所有 causal 路径也开始要求 forward context**——`test_forward_mtp_310` 因此失败
（它 mock 掉了 `forward_chunked_prefill_310`，原本根本走不到任何 `_EXTRA_CTX` 访问，
所以不像同文件另外两个 forward 测试那样 patch `get_forward_context`）。

生产环境里 attention 总在 `set_forward_context` 内跑，不会炸，但这仍是给 causal 路径
加了原本没有的要求。

修法不是改测试，而是把便宜的本地条件提到最前：

```python
if not attn_metadata.causal:      # 本地、绝大多数调用为 False
    ...  # 只有这里才读 _EXTRA_CTX
```

顺带也更快：省掉每层每步一次 contextvar 查找。

**教训**：短路求值的顺序不只是风格问题——它决定了哪些调用方需要满足哪些前置条件。
把需要全局状态的条件放在 `and` 链前面，等于给所有走这条路的人加了依赖。

### 回归命令开得太宽（流程问题）

我最初给的回归命令是 `pytest tests/ut/spec_decode/ tests/ut/_310p/`，把 A2（910B）平台的测试
卷了进来，而它们在 310P 机器上本来就跑不过，导致误报"疑似回归"。

正确做法：回归只跑 `tests/ut/_310p/`；A2 测试在 310P 机器上应以**已知失败基线**对待，
比对数字是否变化，而不是期待零失败。

---

## G0.5 首轮：post_target_replay 不复现，但**没有 control**，不算 Snapshot A

计划：[2026-07-25-qwen3-8b-dspark-310p-target-aclgraph.md](2026-07-25-qwen3-8b-dspark-310p-target-aclgraph.md)
代码：`47b0a8cc3`（= 故障基线 `f7c460f6e` + 环境开关保护的 sync seam）

### 已确认

- `tests/ut/_310p/` 全部通过，含新增 `test_debug_sync_310p.py`；
- 一次真机 instrumented run：

```text
VLLM_ASCEND_310P_SYNC_STAGE=post_target_replay
VLLM_USE_V2_MODEL_RUNNER=0
GRAPH_E2E_NUM_PROMPTS=1
-> test_qwen3_8b_parallel_draft_graph_310p.py 两个用例都 PASS
```

### 为什么这不能记成 Snapshot A 通过

计划 §6-G0.5 明确要求「每个边界至少重复两次，并保留无同步的 control」，并规定
「同步导致故障消失时只能记为 `NOT_REPRODUCED_UNDER_INSTRUMENTATION`，不能记成 PASS」。
本轮缺三项：**没有 control（sync OFF）**、边界只跑一次、只覆盖五个边界中的第一个。

缺 control 是致命的一项。现在有两种读法，观测上完全一样：

| 读法 | 含义 |
| --- | --- |
| A. 同步压住了故障 | 故障是时序相关的竞态，不是确定性坏地址 |
| B. 故障本来就不再复现 | 环境/算子 cache/镜像漂移，与本次插桩无关 |

在 control 跑出来之前，两者不可分，因此本轮状态记为
`NOT_REPRODUCED_UNDER_INSTRUMENTATION`，不进任何 PASS 计数。

### 如果 control 仍然崩（即读法 A 成立），意味着什么

replay **之后**加一个同步能压住故障，这条推理链是有分量的：

1. 它**削弱 Path B**。Path B 的故障机制是 replay 时用了固化的旧参数——replay 已经
   跑完了，事后同步救不回来。
2. 它**指向 §5.4 第 2 类未知量**：host buffer 的异步生命周期。replay 后同步会把
   下一轮的 metadata 构建推到本轮完全结束之后，正好消除「round N+1 覆盖 round N
   仍在消费的共享 pinned buffer」。
3. 它与「两次复现逐字节相同」**不矛盾**。逐字节相同说明故障一旦触发，状态是确定的；
   不说明触发本身是确定的。

这三条合起来，主嫌疑从 Path A/B 之争转向计划里早已单列的第三项——
**G2 case D（host qLens async-overwrite）**。但这一切都以 control 仍然崩为前提。

### 下一步

补齐 §6-G0.5 要求的 control 与重复，见下一节的测试序列。在 control 之前不改任何代码。

---

## G0.5 Snapshot A control：**07-25 的 AICore fault 两次都没有复现**

日志：`logs/20260727/snapshotA_S0_1_07271451.log`（PASSED）、
`logs/20260727/snapshotA_S0_2_07271451.log`（FAILED）
代码：`47b0a8cc3`，sync seam 未武装（S0 = control）

### 「第二次崩了」不是崩

两次运行都跑完了完整的 64 token 生成。fault 标记计数：

| 日志 | `aicore` / `Illegal instruction` 出现次数 |
| --- | --- |
| `logs/20260725/plog-133119_…`（07-25 单 prompt 崩溃） | 37 |
| `snapshotA_S0_1_07271451.log` | **0** |
| `snapshotA_S0_2_07271451.log` | **0** |

第二次的 FAILED 来自测试断言：

```text
assert exact >= 1, "no prompt matched the eager baseline exactly; ..."
AssertionError: assert 0 >= 1
```

### 关键观测：漂移的是 eager baseline，不是 graph

两次运行的 acceptance 指标**逐位相同**：

```text
num_drafts=22 total_accepted=43
acceptance_per_pos=[0.8181…, 0.5, 0.3181…, 0.1818…, 0.04545…, 0.04545…, 0.04545…]
```

draft/accept 模式完全一致 ⇒ graph 侧两次吐出同一串 token。但一次
`exact 1/1`，另一次 `0/1` 且 `prompt 0: diverges at index 6 (11 vs 13)`。
所以动的是**每轮各自重新生成的 eager baseline**：TP=4 greedy 解码跨进程不可复现，
HCCL 不保证 AllReduce 确定性，borderline argmax 会翻。

这独立证实了计划 §9.1 的判断，而且比预期更严重——参照系本身在漂。

### 处置

1. 删掉 `assert exact >= 1`，改为只记录（计划 §9.1 已明令它不能单独作 oracle）。
   在这个复现活动里它还有第二重危害：**红了无法与 aicore fault 区分**。
2. 加 `GRAPH_E2E_SKIP_BASELINE=1`，复现活动里跳过 eager baseline，单次从 ~4.3 min
   降到 ~2 min。
3. 本轮 Snapshot A 记为 `NOT_REPRODUCED`（2/2 未复现）。**不进 G1**：计划 §6-G0.5
   规定「Snapshot A 在同一 pinned 环境不能复现：先判环境/算子 cache 漂移」。

### 与 07-25 的差异候选（待排查）

- 07-25 的单 prompt 跑也是 capture `[8]`（HANDOFF §6.1d），所以**不是 capture size**；
- 本轮每格用了全新的 `VLLM_CACHE_ROOT`（cold compile cache），07-25 是热 cache；
- 07-25 之后是否动过 torch_npu / CANN / ATB / OPP / 镜像，需要按 §6-G0 的清单取证。

在确定 fault 是「低概率间歇」还是「已被环境漂移掩盖」之前，边界推进没有可定位的对象。

---

## 07-27 决定性结果：故障需要**多个 captured graph**

| capture sizes | prompts | 结果 |
| --- | --- | --- |
| `[8]` | 1 | PASS ×10 |
| `[32]` | 4 并发 serve | PASS |
| `[8, 16, 24, 32]` | 4 并发 serve | **FAULT** |

单图配置全过，多图配置崩。这一条把范围收得比之前任何一次都紧。

### 它同时暴露了计划 P0 的一个设计缺陷

计划 §5.1 把 P0 钉死为单个 `[8]` descriptor，理由正是「先排除 graph size 切换、
多 descriptor 共享 host metadata、event-id 资源膨胀」。现在的证据表明**故障恰好就住在
被排除掉的那一组条件里**——P0 配置在构造上无法复现它。§5.1 需要改：单 descriptor
是收敛后的验收配置，不能同时充当根因定位配置。

### 两个候选机制（都已在计划里点名）

- **H1 多 descriptor 共享 host metadata**（§5.4 / G2 case D）。
  `_query_lens_cpu_buffer` 是 builder 终生持有的**同一块 pinned base**，各 descriptor
  只是它不同长度的 view（`[:num_reqs]`）。单图时不可见；多图时 graph A 冻结的 host
  指针可能在另一个 num_reqs 的写入之后被读到。
- **H2 event-id / stream 资源**。310P 教程明确警告 TP>1 时可捕获图数量受硬件 event-id
  限制且随模型深度增长；这里是 36 层 × 4 图 × TP4。

### 环境不是变量

| | 07-24 | 07-27 |
| --- | --- | --- |
| torch_npu | `2.10.0.post2` | `2.10.0.post1.dev20260613` |
| CANN / OPP | `9.1.0-beta.1` / `20260509` | 同 |
| ATB | （无基线） | `9.1.0.B110` |

torch_npu 的更换发生在 **07-24→07-25 之间**（为支持 compressed mask），因此 07-25
崩溃与 07-27 复现用的是同一套栈。之前「环境漂移」的怀疑排除。

### 遗留异常

HANDOFF §6.1d 记录 07-25 单 prompt `[8]` 崩过，而 07-27 同配置 10 次全过。最省事的
解释是**故障是概率性的，多图显著提高概率**：若单图崩溃率约 10%，「崩 1 次后连过 10 次」
并不反常（0.9¹⁰ ≈ 35%）。暂记为未解释，不阻塞——多图配置已经是稳定复现体。

### 工具改动

`GRAPH_E2E_CAPTURE_SIZES` 把 capture 列表与 prompt 数解耦，用于「捕获 N 个图但只
replay 一个」这类实验；并对非 `1+K` 倍数的尺寸 fail-fast（`[28]` 会被 vLLM 静默改写成
`[8]`，见下）。

### 附带发现：`[28]` 陷阱

`vllm/config/compilation.py::adjust_cudagraph_sizes_for_spec_decode` 在
**MRV1 + decode_mode()==FULL + K>0** 三条同时成立时把 capture 列表向上取整到 `1+K`
的倍数，并丢弃超过 `max_cudagraph_capture_size` 的项；全被丢弃时兜底为 `[1+K]`。

```text
[28]            -> [8]      # 兜底分支，max 也降到 8，16/24/32 全部回落 eager
[28, 32]        -> [32]
[8, 16, 24, 32] -> 原样
```

PR #11765 的 `[28, 32]` 是对的——它是 **PIECEWISE**，`decode_mode()` 不等于 `FULL`，
改写不触发；且 28 = 4×K 是 **drafter** 的形状，PIECEWISE 下 drafter 也入图。
FULL_DECODE_ONLY 下 drafter 全程 eager，28 不对应任何形状。整个过程无 warning。
