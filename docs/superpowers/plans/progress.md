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
| Phase 0.2 ADN 算子基线 | ⬜ 待跑 `Ascend_Ops/tests/test_adn_fia.py` | 310P 真机 + ADN |
| Phase 0.4 ADN NZ 直读门禁 | 🟡 核心用例已过，fixture 修正待复跑 | 310P 真机 + ADN |
| Task 4 Qwen3-8B TP=2 eager E2E | ⬜ 未开始 | **需 310P 真机 + checkpoint** |
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
