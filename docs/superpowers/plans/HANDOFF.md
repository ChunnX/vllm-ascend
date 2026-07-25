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
2. **嫌疑 Add 在图外**：`model_runner_310p.py:316` 的调用点位于 `_prepare_inputs`（`:245`），
   在 model forward **之前**，不在捕获的图内。所以是"图 replay ↔ 图外主机代码"的交互，
   不是"图内 op 被破坏"。形状也对得上：Add 规模 = `num_reqs` = 2 = args 9/10/11 的 `0x2`。
3. **已有一道同类屏障，且我们正好踩在其触发条件上**：`model_runner_310p.py:113-123`
   在 `finished_req_ids` 非空时 `torch.npu.current_stream().synchronize()`，
   注释写明是为了「condense() 重写 block_table.np 前排干上一步的 ACL graph replay」。
   崩溃正好发生在 batch 3→2（有请求结束）。**它只覆盖 `block_table.np`**，
   而 `_prepare_inputs` 里那组 `num_computed_tokens` / `prev_positions` /
   `valid_sampled_token_count` 缓冲不在保护范围内 —— 这是修复方向的第一候选。

### 6.2 最可能的两条线（未验证，别直接改代码）

崩在"batch 3→2"这一步，同时踩中两件事，必须先分开：

- **(a) 图尺寸切换**：24 → 16，第一次换图。多个 graph size 共享一个 memory pool，
  某个 ATB op 捕获期分配的 workspace/tiling 被另一张图复用
- **(b) batch 变更的簿记**：`finished_req_ids` → `condense()` 搬 `block_table.np` 行 +
  async spec decode 的 `update_num_computed_tokens_for_batch_change()`
  （`vllm_ascend/spec_decode/utils.py:28` 正好是一个 `corrected = prev_computed + valid_counts`
  的小 Add）。`model_runner_310p.py:112` 已有一道 sync 屏障，但那是给同步调度写的

注意 310P 走的是**独立的图契约**（`model_runner_310p.py:66` 的 docstring）：不注册
`graph_params`，所以 A2/A3 的 `update_graph_params()` 在 310P 上是空转
（`attn_params` 为空 → early return）。也就是说 **replay 前没有任何 op 参数刷新**，
一切靠"输入缓冲地址不变"。这正是 (a) 成立的前提条件。

### 6.3 ⚠️ `ASCEND_LAUNCH_BLOCKING=1` 在图模式下用不了

`vllm_ascend/platform.py:593-604`：只要 `cudagraph_mode != NONE` 且
`ASCEND_LAUNCH_BLOCKING=1`，`check_and_update_config()` 直接 `raise ValueError`。
设计文档 `docs/source/developer_guide/Design_Documents/ACL_Graph.md:71,122` 也写死了这条。
原因是 launch blocking 在每次下发后插 stream sync，而 capture 期间对捕获流做 sync 非法。

**所以拿不到"准确 Python 栈"这条路是堵死的**，只能从 plog / dump 侧反推。别再提这个建议。

### 6.4 下一步实验（按序，每步一次跑，别并行改多个变量）

**第 0 步不用跑，服务器上现成的文件就能定位到是哪个 Add：**

- 完整（未过滤的）plog：`/root/ascend/log/debug/plog/` 里 pid 80316 那个文件，
  查 `stream_id=11` 上 `task_id` 19943 / 19944 / **19945** / 19946 附近的 kernel name
  → 出错的 Add 前后是哪几个算子，直接看出在图内还是图外
- CANN 已经自动写好的异常 dump（plog:25,27）：
  `extra-info/data-dump/1/exception_info.11.19945.20260725034057874`
  `extra-info/data-dump/1/Add_41dadce325b0f810d03359af2a38990b_high_performance_223000000_host.o`
  同目录/kernel_meta 里的同名 `.json` 有输入 shape 和 dtype → 对着源码认哪个 Add

`tests/.../test_qwen3_8b_parallel_draft_graph_310p.py` 已加两个 env 旋钮，
不用改代码就能 bisect（prompt 数会自动带着 capture sizes 一起缩）：

```bash
# 1) 打开 ACLGraphWrapper 的 replay 输入地址断言（acl_graph.py:101,243）——
#    正好验证 §6.2 的"缓冲地址在 capture 与 replay 之间变了"这一族假设
VLLM_LOGGING_LEVEL=DEBUG VLLM_USE_V2_MODEL_RUNNER=0 pytest -sv \
  tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_graph_310p.py
```

```bash
# 2) 单 prompt：无 batch 变化、无 condense、只有一张图 [8]
GRAPH_E2E_NUM_PROMPTS=1 VLLM_USE_V2_MODEL_RUNNER=0 pytest -sv \
  tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_graph_310p.py
```

```bash
# 3) 3 prompt 但关掉异步调度：过了 = (b)，还崩 = (a)
GRAPH_E2E_ASYNC_SCHEDULING=0 VLLM_USE_V2_MODEL_RUNNER=0 pytest -sv \
  tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_graph_310p.py
```

**在第 0 步认出那个 Add 之前不要提交任何修复**——现在能编出至少四个都自洽的故事。

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
