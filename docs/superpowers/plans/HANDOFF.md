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

**当前状态：eager 全链路已跑通并验证正确；图模式（target 入图）已实现待真机验证。**

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

**target 入图 + drafter eager**（代码已完成，未真机验证）：

- 模式必须是 **`FULL_DECODE_ONLY`**（不是 PIECEWISE）——证据：PR #11765 = `41ff81e1a`
  自己的性能表两行都用 FULL_DECODE_ONLY
- `cudagraph_capture_sizes` 必须是 **`uniform_decode_query_len = 1 + K = 8`** 的整数倍，
  按 prompt 数推导（`[8, 16, 24]`）。**写 7 永远不会命中**
- 待跑：`VLLM_USE_V2_MODEL_RUNNER=0 pytest -sv tests/e2e/pull_request/four_card/_310p/test_qwen3_8b_parallel_draft_graph_310p.py`

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

1. **跑图模式 E2E**。捕获失败时按此顺序退：减 prompt 数（capture sizes 会自动跟着变）→ 只留 1 条。
   **不要只砍 capture sizes 不砍 prompt**，否则批次不匹配会静默回退 eager，测试通过但什么也没证明
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
