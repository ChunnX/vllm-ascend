# DFlash/DSpark 310P 适配 — 项目交接

> 新会话从这里开始。详细过程见 [progress.md](progress.md)，原始设计见
> [2026-07-22-dflash-dspark-310p-qwen3-8b.md](2026-07-22-dflash-dspark-310p-qwen3-8b.md)。

| | |
| --- | --- |
| 分支 | `dflash_dspark_310p_adapt_20260723` |
| 基线 | vllm-ascend `f9a5d8df8`（= `e4c88fb0b` + 计划文档），vllm `752a3a504` (= tag v0.25.1) |
| 环境 | CANN OPP 9.1.0-beta.1、torch 2.10.0+cpu、torch_npu 2.10.0.post2 |

---

## 1. 项目目标

让 **Qwen3-8B + DSpark drafter（K=7）在 Ascend 310P（Atlas 300I DUO）上跑通投机解码**，
MRV1 + TP=4 + FP16 + eager。核心难点是 310P 没有 Triton、且没有 910 的 FIA 算子。

**当前状态：eager 全链路已跑通并验证正确；图模式捕获成功但 replay 崩在 aicore
（见 §6.1），根因未定位。**

---

## 2. 关键技术栈

- **vLLM Model Runner V1**（MRV1）— 必须 `VLLM_USE_V2_MODEL_RUNNER=0`
- **ADN 算子** `adn_fused_infer_attention`（来自 `Ascend_Ops` 仓库的 custom OPP + PTA wheel）
  — 提供 310P 上的 **non-causal paged attention**，是 draft attention 的唯一实现
- **ATB 算子** — 310P 的 causal attention（`_npu_flash_attention[_v3]`、
  `_npu_paged_attention[_splitfuse][_v2]`）。**310P 完全不用 FIA**
- torch_npu / CANN；**310P 无 Triton**

---

## 3. 三个仓库的分工

```
vllm-workspace/
├── vllm/           上游 vLLM（只读参考，v0.25.1）
├── vllm-ascend/    ★ 本次开发的仓库
└── Ascend_Ops/     ADN 算子源码 + 算子级测试（只读参考）
```

---

## 4. 本次新增/改动的文件

### 生产代码（8 个文件）

| 文件 | 说明 |
| --- | --- |
| `vllm_ascend/_310p/spec_decode/parallel_drafting_inputs.py` | **新增 136 行**。Triton kernel 的向量化 PyTorch 等价实现（310P 无 Triton）。含 `ADN_BLOCK_SIZE=128` 和 `resolve_310p_block_size()` |
| `vllm_ascend/_310p/attention/adn_fused_infer_attention.py` | **新增 309 行**。ADN adapter：惰性加载、scope 校验、逐步动态校验、forward |
| `vllm_ascend/_310p/attention/attention_v1.py` | `forward_impl` 加 non-causal 路由；`forward_prefill_310` 的 seq_len 强制到 host |
| `vllm_ascend/spec_decode/dflash_proposer.py` | 新增 `_expand_drafting_inputs()` dispatch seam 与 `_profile_rope_context()`（DSpark 继承二者） |
| `vllm_ascend/spec_decode/dspark_proposer.py` | per-group 循环改走 dispatch；单 group scope guard |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | 新增 `apply_context_rope()`：310P 逐层 RoPE + 接住返回值 |
| `vllm_ascend/patch/worker/__init__.py` | `patch_qwen3_dflash` 移出 `if not is_310p()` |
| `vllm_ascend/device/device_op.py` | ⚠️ **非我方改动**（用户在 `9454ca37d` 提交）：把三个 triton import 移进 `if HAS_TRITON`。**`else` 分支未给这三个名字赋值**，无 triton 时它们未定义 |

### 测试

| 文件 | 类型 |
| --- | --- |
| `tests/ut/_310p/spec_decode/test_parallel_drafting_inputs_310p.py` | CPU 单测（含 Triton kernel 的逐行 golden） |
| `tests/ut/_310p/test_qwen3_context_rope_310p.py` | CPU 单测（逐层 RoPE） |
| `tests/ut/_310p/attention/test_adn_fused_infer_attention_310p.py` | CPU 单测（ADN adapter，fake `adn_custom_ops`） |
| `tests/ut/_310p/attention/test_parallel_draft_routing_310p.py` | CPU 单测（路由） |
| `tests/ut/_310p/attention/test_attention_v1_310.py` | **改**：补 compressed-mask 路径覆盖 |
| `tests/e2e/_310p/spec_decode/smoke_parallel_drafting_inputs.py` | **手工**硬件门禁（NPU vs CPU 逐字段比对） |
| `tests/e2e/_310p/adn/smoke_adn_nz_readback.py` | **手工**硬件门禁（ADN 直读 vLLM NZ cache） |
| `tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_eager_310p.py` | E2E（eager，已通过） |
| `tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_graph_310p.py` | E2E（图模式，**待验证**） |

---

## 5. 已完成并验证

| 项 | 结果 |
| --- | --- |
| 无 Triton 输入展开 helper | CPU 单测通过 + 310P 真机算子门禁通过 |
| DFlash / DSpark dispatch seam | CPU 单测通过 |
| 五层逐层 context RoPE + patch gate | CPU 单测通过 |
| ADN adapter + 精确路由 | CPU 单测通过 |
| **ADN 直读 vLLM NZ cache（地基假设）** | **310P 真机通过**，mean_abs ~7e-5（判据 1e-4） |
| Ascend_Ops 算子基线 `tests/test_adn_fia.py` | 40/40 通过 |
| **eager E2E（DSpark + TP=4）** | **通过**：3/3 token 完全一致，acceptance `[0.80, 0.46, 0.38, 0.30, 0.11, 0.11, 0.05]` |
| `vllm serve` eager | 用户已验证可用 |
| 全量单测 `tests/ut/_310p/` | 164 passed, 2 skipped |

---

## 6. 正在开发 / 待验证

**target 入图 + drafter eager**：捕获成功，**replay 阶段真机崩了**（07-25 11:40）。

- 模式必须是 **`FULL_DECODE_ONLY`**（不是 PIECEWISE）——证据：PR #11765 = `41ff81e1a`
  自己的性能表两行都用 FULL_DECODE_ONLY
- `cudagraph_capture_sizes` 必须是 **`uniform_decode_query_len = 1 + K = 8`** 的整数倍，
  按 prompt 数推导（`[8, 16, 24]`）。**写 7 永远不会命中**

### 6.1 07-25 11:40 崩溃：已知事实

日志：`logs/20260725/test_e2e_parallel_draft_graph_310p_07251140.log` +
`logs/20260725/plog-80316_20260725034057875.log`

| 事实 | 出处 |
| --- | --- |
| 捕获成功，`[8,16,24]` 三个图 6 秒捕完，0.21 GiB | e2e log:106,114 |
| eager baseline 全过；崩的是第二个 `VllmRunner`（图模式） | e2e log:647-680 |
| 崩溃时刻 = 首次 replay 之后约 0.8 s | e2e log:134 vs plog:65 |
| **崩溃时 3 条 prompt 已完成 1 条，`num_running_reqs=2`** | e2e log:546 |
| 只有 TP1/TP2 报 aicore 异常，TP0/TP3 是被拖死的 | e2e log:211,332 |
| Python 栈停在 rejection sampler 的 `aclnnNonzeroV2`（**异步，栈不准**） | e2e log:248 |
| **真正出错的核是 `Add_41dad…_high_performance`，`blockDim=1`、`tilingKey=0`** | plog:5,3 |
| 报错语义：`Illegal instruction, usually caused by unaligned UUB addresses` | plog:28 |

**读法**：`Illegal instruction / unaligned UUB` 不是数值越界，是**这个 Add 核拿到的
tiling / 地址本身是坏的**——典型的"图里捕获的某个 op，其 tiling 或 workspace 缓冲在
replay 时已经被别人复用"。`NonzeroV2` 只是崩溃后第一个 sync 点，**不要去查 rejection
sampler**。

### 6.1b 已排除的假设（有直接证据）

用户在 **eager** 下用 `vllm serve` **显式开启 async-scheduling** 跑过，**正常**。

`gpu_model_runner.py:658`：`use_async_spec_decode = use_async_scheduling and num_spec_tokens > 0`
——只需这两个条件。所以那次 serve 里 `_prepare_inputs` 的异步簿记
（`update_num_computed_tokens_for_batch_change`，含嫌疑 Add）**确实执行过**，
且 serve 的连续批处理必然有请求先后结束 → batch 变化也覆盖了。

| 假设 | 状态 |
| --- | --- |
| async scheduling 本身有问题 | ❌ 排除 |
| 异步 spec decode 簿记代码本身有 bug | ❌ 排除（同一段代码在 eager serve 跑过） |
| 图 replay 与其他东西的交互 | ✅ 唯一剩下的 |

（此前是从 `vllm/config/vllm.py:992-1040` 的自动解析**推断** eager 也开着 async；
现在有直接证据。）

**因此 §6.4 的实验顺序调整为：先跑单 prompt**——既然簿记代码在 eager 下是好的，
问题必然涉及图，而单 prompt 是"图 replay + DSpark"的最小可验配置
（一张图 `[8]`、无尺寸切换、无 batch 变化、无 condense）。
过了再用 `ASYNC_SCHEDULING=0`（3 prompt）分 (a)/(b)；还崩则说明问题更根本，范围反而更好缩。

### 6.1c 补充证据（本次静态分析）

1. **args 有 poison**：`arg12 = 0xa5a5a5a5_03054048`，高 32 位是典型未初始化/已释放填充。
   配合 `tilingKey=0` 与 `unaligned UUB`，指向**该核拿到的 args/tiling 本身是脏的**，
   而非数值越界。`arg1`(输入1) 与 `arg3`(tiling) 仅差 32 字节，同一 args 区。
2. ~~嫌疑 Add 在图外，规模 = num_reqs = 2~~ **已被 §6.1d 推翻**：单 prompt 下
   args 9/10/11 仍是 `0x2,0x2,0x2`，与 `num_reqs` 无关。
3. **已有一道同类屏障**：`model_runner_310p.py:113-123` 在 `finished_req_ids` 非空时
   `torch.npu.current_stream().synchronize()`。**§6.1d 之后这条也不再是嫌疑**
   （单 prompt 根本不触发 condense，照样崩）。

### 6.1d 07-25 16:06 单 prompt 复现：**两次崩溃逐字节相同**

日志：`logs/20260725/plog-133119_20260725080027851.log`（`GRAPH_E2E_NUM_PROMPTS=1`）

两次运行不同进程、不同 device、不同 batch、不同 capture 集合，但故障状态**完全一致**：

| 字段 | 3 prompt 跑 | 1 prompt 跑 |
| --- | --- | --- |
| args 0-12 | `…d9a00, …a428, …e4200, …a448, …200000, 0x1,0,0,0, 0x2,0x2,0x2, 0xa5a5a5a503054048` | **完全相同** |
| `tilingKey` / `argsSize` / `blockDim` | 0 / 104 / 1 | **完全相同** |
| `pc start` / `current` / `para base` | `0x80012c18bc03f68` / `0x12c18bc03f80` / `0x12c20047a400` | **完全相同** |
| `task_id` | 19945 | **19945** |
| capture sizes | `[8,16,24]` | `[8]` |
| device / stream | 1 / 11 | 2 / 373 |

**这组事实一次性排掉一整族假设：**

| 假设 | 状态 |
| --- | --- |
| batch 3→2 / `condense()` / `finished_req_ids` | ❌ 单 prompt 无此事件 |
| 图尺寸切换（24→16）/ 多图共享 memory pool | ❌ 单 prompt 只有一张图 `[8]` |
| 任何竞态、时序、异步簿记 | ❌ **逐字节确定性复现，竞态不长这样** |
| args 形状随 `num_reqs` 变 | ❌ 1 和 3 个请求下都是 `0x2` |

剩下的只有一件事：**一个形状与 batch 无关、在确定性位置下发的核，拿到了从没被写过的
args**（`tilingKey=0` + `0xa5a5a5a5` poison = 该 args 缓冲整块没初始化）。

### 6.2 首要假设：`_npu_paged_attention_splitfuse_v2` 从没被捕获过

`forward_impl`（`_310p/attention/attention_v1.py:379-388`）分岔：

- `DecodeOnly` → `forward_paged_attention` → `_npu_paged_attention`
- **`SpecDecoding`** → `forward_chunked_prefill_310` → **`_npu_paged_attention_splitfuse_v2`**

仓库里现有的 310P 图模式覆盖，**全部走上面那条**：

| 测试 | 图 | 投机 |
| --- | --- | --- |
| `one_card/_310p/test_dense_model_310p.py::test_qwen3_dense_tp1_w8a8_aclgraph` | ✅ FULL_DECODE_ONLY | ❌ |
| `one_card/_310p/test_dense_model_310p.py::test_qwen3_5_dense_tp1_fp16_aclgraph` | ✅ FULL_DECODE_ONLY | ❌ |
| `one_card/_310p/test_spec_decode_mtp_310p.py::test_qwen3_5_mtp_tp1_eager` | ❌ `enforce_eager=True` | ✅ |
| **本仓库本次的 graph E2E** | ✅ | ✅ ← **唯一同时占两格的** |

也就是说 **splitfuse_v2 进 ACLGraph 这件事，此前没有任何测试做过**。它的 `seq_len` 是
主机 pinned 张量、tiling 在主机侧每次现算——正是设计文档
（`ACL_Graph.md`「Host-side attention parameter update for full graph replay」）说的
"即使图是静态的也需要运行时刷新参数"那类算子。而 310P 走独立契约、`attn_params` 为空、
`update_graph_params()` 空转（`acl_graph.py:316` 初始化为 `{size: []}`），
**replay 前没有任何参数刷新** → 与 poison args 的现象吻合。

### 6.2a 判决实验结果：**spec-off 图模式 PASS**（07-25）

`test_target_only_in_aclgraph_no_spec` **通过**。同一套 `COMMON`（TP=4 / fp16 /
block_size=128 / FULL_DECODE_ONLY / 同样的模型和机器），唯一变量是关掉投机。

**结论：310P 基础图路径本身是好的，故障是投机专属的。**

捕获路径的分岔已在代码里核实：
`model_runner_310p.py:638-645` → `_spec_dummy_capture=True` →
`model_runner_310p.py:196-197` 把 `attn_state` 强制成 `SpecDecoding` →
`forward_impl` 走 `forward_chunked_prefill_310` → **`_npu_paged_attention_splitfuse_v2`
进图**。关掉投机则是 `DecodeOnly` → `_npu_paged_attention` 进图。
**通过与崩溃的两次运行，差别正好就在这一个算子上。**

### 6.2b Add 的 json 查了：只给到 dtype，没给到调用点

`logs/20260725/Add_41dadce325b0f810d03359af2a38990b_high_performance.json`

```
simplifiedKey: Add/d=0,p=0/9,2/9,2/9,2      (9=int64, 2=ND)
inputs/outputs: int64, ND, shape [-2]       opMode: dynamic
coreType: AiCore   kernelList: 43 个 SoC 变体
```

这是 **CANN 算子库里通用的动态 shape int64 逐元素 Add 二进制**，进程里每一个 int64 加法
都共用它 —— 所以它**只能告诉我们 dtype 是 int64，认不出是哪一行 Python**。

不过 int64 这一条仍然有用：**排除了模型内部所有 fp16 加法**（residual 之类），
剩下的全是索引 / 元数据算术。候选集中在三处，都只在投机开启时执行：

- `sample/rejection_sampler.py:1041` `global_idx = start_indices.unsqueeze(1) + copy_indices`
  （紧挨着崩溃现场那个 NonZero）
- `_310p/spec_decode/parallel_drafting_inputs.py:121,125,127,136`（我们写的 helper，全 int64）
- `spec_decode/utils.py:28` `corrected = prev_computed + valid_counts`

**还差最后一步才能定死**：`tilingKey=0` 与 args 里的 `0xa5a5a5a5` poison 说明
tiling 缓冲整块没写过 —— 而 `exception_info` dump 里有出错时的**实际操作数字节**
（每个张量 dump 了 32 字节 = 4 个 int64），对上值就能认出调用点。这一步是 `cat`，不用跑：

```bash
ls -la /vllm-workspace/vllm-ascend/extra-info/data-dump/2/
xxd /vllm-workspace/vllm-ascend/extra-info/data-dump/2/exception_info.373.19945.20260725080027851 | head -40
```

### 6.3 ⚠️ `ASCEND_LAUNCH_BLOCKING=1` 在图模式下用不了

`vllm_ascend/platform.py:593-604`：只要 `cudagraph_mode != NONE` 且
`ASCEND_LAUNCH_BLOCKING=1`，`check_and_update_config()` 直接 `raise ValueError`。
设计文档 `docs/source/developer_guide/Design_Documents/ACL_Graph.md:71,122` 也写死了这条。
原因是 launch blocking 在每次下发后插 stream sync，而 capture 期间对捕获流做 sync 非法。

**所以拿不到"准确 Python 栈"这条路是堵死的**，只能从 plog / dump 侧反推。别再提这个建议。

### 6.4 下一步（`NUM_PROMPTS=1` 与 `ASYNC_SCHEDULING=0` 两条已作废，见 §6.1d）

**第 0 步不用跑 —— CANN 崩溃时已经把答案写到磁盘上了**（plog:142-143）：

```bash
cat /vllm-workspace/vllm-ascend/extra-info/data-dump/2/Add_41dadce325b0f810d03359af2a38990b_high_performance.json
ls -la /vllm-workspace/vllm-ascend/extra-info/data-dump/2/
```

那个 `.json` 里有这个核的输入 shape / dtype / 编译签名 —— 对着源码就能认出是哪个 Add，
不用再猜。`exception_info.373.19945.*` 里是出错时的实际张量数据。

~~第 1 步判决实验~~ **已完成，PASS，见 §6.2a。**

**下一步只剩两条，都还没做：**

1. `exception_info` 的操作数字节（§6.2b 的 `xxd`）→ 认出那一行 Python。**零成本，先做这个**
2. 拆开"splitfuse 进图"与"图外 eager 投机代码"：目前 spec-off 一次性去掉了
   splitfuse + drafter + rejection sampler + 投机簿记**四样**，还没分开。
   候选做法是用一个不走 splitfuse 的投机方法（310P 的 ngram 把
   `uniform_decode_query_len` 钉成 1 → `DecodeOnly` → paged attention，
   见 `model_runner_310p.py:106-110`），保留 rejection sampler 与簿记。
   **先确认那条组合在 310P 上本来是通的**，否则跑出来无法解释

**关于 `VLLM_LOGGING_LEVEL=DEBUG` 日志太多**：§6.1d 的确定性复现已经排掉了地址竞态那
一族，这条不再值得跑。真要跑就重定向后只看尾巴（断言失败会在最后）：
`… > /tmp/dbg.log 2>&1; tail -200 /tmp/dbg.log`。

**在第 0 步认出那个 Add 之前不要提交任何修复。**

### 6.5 `rejection_sampler.py:1042` 不是故障点

`aclnnNonzeroV2` 是 AiCPU 算子，输出 shape 要回主机，PTA 在它内部调
`aclrtSynchronizeStream`（plog:39 `UpdateOutputShapeFromExtInfo`）。它是**崩溃后第一个
同步点**，不是出错的地方——出错的是更早下发的 `Add`，task_id 19945（plog:5）。
在 1042 行上做任何改动只会把报错挪到下一个同步点。

另外 `Illegal instruction / unaligned UUB` 是核内地址计算越界，**不是**索引越界：
`target_argmax[global_idx[...]]` 就算 index 全错也不会让一个 `Add` 报这个。

---

## 7. 已知问题

### 7.1 acceptance 低于 A2/A3（效率，非正确性）

| | Acc Length | Acc Rate per Position |
| --- | --- | --- |
| A2/A3（PR #11765，gsm8k 100 req / 2048 token） | 6.30 | `[0.94, 0.87, 0.81, 0.74, 0.69, 0.63, 0.59]` |
| 本次 310P（3 短 prompt / 64 token） | ~2.2 | `[0.80, 0.46, 0.38, 0.30, 0.11, 0.11, 0.05]` |

**两者不可直接比**（数据集、输出长度差异大）。要下结论必须先在**同数据集同输出长度**复测。
pos-1 从 0.87 掉到 0.46 值得查，可能是 (a) ADN vs FIA / 逐层 vs fused RoPE 的 fp 差异累积，
或 (b) draft 管线细微偏差。**不阻塞功能。**

### 7.2 drafter 无法入图（需 Ascend_Ops 改算子）

- `custom_pta/csrc/registration.cpp:56-57`：`actual_seq_lengths_q/kv` 是 **`SymInt[]`**，
  图捕获时被固化成常量，而 KV 长度每步在变 → replay 会**静默算错**
- `adn_fused_infer_attention.cpp:26`：输出每次 `at::empty_symint` 新分配，无 `out=`
- A2/A3 的 FIA 有 tensor 形态可用，ADN 没有 → **vllm-ascend 侧无法绕过**

### 7.3 A2 测试在 310P 机器上有 9 条既有失败

`tests/ut/spec_decode/a2/test_eagle_proposer.py`，**与本期无关**（已回退验证）。
根因：无 Triton 的 `kernel[grid]`、`patch_idex_310` 的零参 `super()` 绑定错误。
**回归只跑 `tests/ut/_310p/`**，不要带 `tests/ut/spec_decode/`。

### 7.4 DFlash 端到端未验证

无 checkpoint。但其 q=9/skip-anchor 布局有 CPU 单测 + Phase 0 门禁的 DFlash q=9 用例覆盖。

---

## 8. 下一步

1. **定位 §6.1 的 Add**，按 §6.3 的三步跑。
   **不要只砍 capture sizes 不砍 prompt**，否则批次不匹配会静默回退 eager，测试通过但什么也没证明
   （`GRAPH_E2E_NUM_PROMPTS` 已经把两者绑在一起了）
2. **acceptance 复测**（可比条件下），再决定要不要深挖
3. **Task 5**：整理提交、写 PR。建议提交边界见计划文档 §5.4
4. （可选）向 Ascend_Ops 提 ADN 的 ABI 需求：lengths 的 tensor 形态 + `out=`

---

## 9. 约束与注意事项

### 硬约束

- **ADN 永不能被图捕获** — `forward_parallel_draft_adn` 里有逐次 guard，别删
- **non-causal 不得回退 causal splitfuse** — 会返回看似合理的错数值。所有未支持的
  non-causal 必须 `raise`
- **不得改 A2/A3 行为** — 所有 310P 限制只加在 `is_310p()` 分支或 `_310p/` 目录内。
  典型反例：把 block size 的 `!= 128` guard 写在 dispatch 外面会打挂 A2/A3
- **回归只跑 `tests/ut/_310p/`**（见 7.3）

### 容易踩的坑（都真实发生过）

| 坑 | 教训 |
| --- | --- |
| stub 属性名靠猜 | 写 stub 前先 `awk` 列出目标方法实际读的 `self.*`；注意"源 dict"vs"派生 dict" |
| 断言代替构造 | fixture 的性质要**构造保证**，不要"随机生成 + 事后断言"（block table 不动点踩过两次） |
| 短路顺序改变前置条件 | 把需要全局状态的条件放 `and` 链前面 = 给所有调用方加依赖（`_EXTRA_CTX` 打挂 causal 路径） |
| 读 `_EXTRA_CTX` 不检查 context | 必须 `is_forward_context_available()` 短路，否则无 forward context 的调用方会炸 |
| 拿测试阈值当性能基线 | `BASELINES` 是宽松阈值不是实测值，真实数字在 PR 描述里 |
| token 一致性当正确性门 | greedy 投机解码**不是** bit-identical（verify batch 形状不同）。**坏的 drafter 反而会让输出与 baseline 一致**（全被拒）。用 acceptance 判 |

### 环境相关

- `adn_custom_ops` 的包 `__init__` **顶层 import torchair** — 即使不用图模式也必须能导入
- 用户已升级 torch_npu + ATB 以启用 **compressed mask**（`_npu_flash_attention_v3` /
  `_npu_paged_attention_splitfuse_v2`）。旧的 `get_splitfuse_mask` 带 `.to("cpu")` 同步，
  **无法入图**；compressed 版用缓存固定 mask，这是图模式的前提
- 模型路径：`/opt/foundation_model/{Qwen3-8B, dspark_qwen3_8b_block7}`，
  可用 `QWEN3_8B_PATH` / `DSPARK_QWEN3_8B_PATH` 覆盖

### 开发流程

Mac 上**跑不了任何单测**（装不了 torch_npu，`tests/ut/base.py` → `vllm_ascend/utils.py:34` 依赖它）。
Mac 只能做语法 / 行宽 / import 检查，**所有验证在 310P 服务器手工执行**。
