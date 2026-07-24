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
| Task 1.3b DSpark per-group dispatch | 🟡 待验证 | CPU 单测 |
| Task 2 五层逐层 context RoPE + patch gate | ⬜ 未开始 | CPU 单测 |
| Task 3 ADN adapter + 精确路由 | ⬜ 未开始 | CPU 单测 |
| Phase 0 硬件门禁（ADN NZ readback / 输入展开 op smoke） | ⬜ 未开始 | **需 310P 真机** |
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

提交：待填
验证：**待服务器确认**，预期 20 passed（15 原有 + 5 新增）

DSpark 复用继承来的 `_expand_drafting_inputs`，不重复定义。它与 DFlash 的差异全部保留：
按 KV cache group 循环、per-group buffers、query 数为 K（非 K+1）、`sample_from_anchor=True`。

新增单 group scope guard（仅 310P 生效）：Qwen3-8B 的 5 层同规格会被 vLLM 并成 1 个 group，
多 group 机器是给 DeepSeek-V4 准备的，本期显式拒绝而不是静默只展开其中一个。

`kv_block_size` 那行**没有**加 128 断言——它跑在所有设备上，pin 死会打挂 A2/A3 的合法配置。
310P 的 pin 在 dispatch 内部的 `resolve_310p_block_size()` 里。

删掉了 `dspark_proposer.py` 已不再使用的 Triton import。

---

## 下一步

**Task 2：五层逐层 context RoPE + patch gate。**

两件事：

1. `patch/worker/__init__.py` 把 `patch_qwen3_dflash` 移出 `if not is_310p()`
   （`patch_qwen3_dspark` 已被删除，DSpark 靠继承同一个模型侧实现）；
2. `patch_qwen3_dflash.py` 的 context RoPE 改为 310P 逐层：当前是把
   `[L * num_ctx]` 一次性送进 RoPE，而 310P drafting 期的全局 cos/sin buffer 只有
   `max_num_batched_tokens` 大，5 层 drafter 一定溢出。逐层每次只送 `num_ctx`。
   同时必须接住 out-of-place 返回值（310P RoPE 返回新 tensor，不原地改）。

外加 §2.2 下半段的 `_profile_rope_context`：profile 路径绕过 `_run_merged_draft`，
drafting flag 从未打开，两个 forward（context KV 预写 + 紧随的 query forward）都会读到
target 遗留的 cos/sin slice。用 `try/finally` 把整个 `is_profile` 分支包起来。

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

### 回归命令开得太宽（流程问题）

我最初给的回归命令是 `pytest tests/ut/spec_decode/ tests/ut/_310p/`，把 A2（910B）平台的测试
卷了进来，而它们在 310P 机器上本来就跑不过，导致误报"疑似回归"。

正确做法：回归只跑 `tests/ut/_310p/`；A2 测试在 310P 机器上应以**已知失败基线**对待，
比对数字是否变化，而不是期待零失败。
