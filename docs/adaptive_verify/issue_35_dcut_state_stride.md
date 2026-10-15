# Issue #35：030 动态校验乱码的状态跨度修复

## 与 028_d_verify 的对比

`028_d_verify` 的 `_reshape_mamba_kv_cache` 按状态种类切分 raw buffer，
conv 和 SSM 各自是连续数组。DCUT 用逻辑行大小寻址，在该布局下成立。

`030_dspark_dev` 改为标准化混合缓存页：每个物理页内包含 conv、SSM 和 padding，
每种状态只保持行内连续，`stride(0)` 是整个物理页的跨度。
动态校验仍调用本仓库的 `npu_dcut_causal_conv1d` 和
`npu_dcut_recurrent_gated_delta_rule`。这些算子原本使用：

- conv：`cache_id * state_len * dim`；
- recurrent：`state_id * num_heads * value_dim * key_dim`。

两者均忽略物理行跨度，可能读取或覆盖错误的状态位置。
这解释了为何关闭动态校验的 FA4 路径正常，开启后 target 校验会出错；
本次修复没有修改已经验证的 FA4 页表映射。
用户随后在部署机器完成验证并确认功能正常，问题已关闭。

## 修复

适配层读取状态张量的 `stride(0)`，以元素数传入 ACLNN 属性和 tiling。
卷积的历史读取、回滚和写回，以及 recurrent 的初始状态读取、预取和逐 token 快照，
全部使用物理行跨度。910 和 950 recurrent 内核同步修改。

状态输入标记为 `IgnoreContiguous`，避免把原地更新的状态自动转换为临时连续副本。
仅支持行内连续的状态，适配层明确检查该约束。
未指定跨度时默认使用逻辑行大小，保留连续状态的默认行为。
Torch 算子调用参数保持不变；内部 ACLNN 属性和 tiling 已更新，需要重新构建配套二进制。

不复制整个状态池、不额外分配状态池、不增加设备到 CPU 的同步。
CMake 和其他编译工程文件未修改。

## 本地证据与限制

CPU 回归直接提取当前内核中的状态地址表达式，用 C++ 编译并检查地址是否落在目标物理行。
对未修复的 030 源码执行同一测试：连续布局通过，带跨度布局失败。
修复后两种布局均通过。

连同既有 FA4、动态校验和 DCUT CPU 参考测试，共 114 项测试及 16 项子测试通过。
新增 NPU 用例比较连续和带跨度状态的输出、原地状态更新及 padding 隔离，
覆盖 BF16/FP32 recurrent state、previous accepted 大于当前 query 长度、零长度请求和图回放。
随后在服务器的 `zsy-va-030-dev` 容器（CANN 9.2、ascend910b1 构建目标）
完成整包编译和 editable 安装，`pip install` 返回 0。
上机编译修复了 Arch35 未定义的 `DIM_3` 引用，以及 `EXEC_NPU_CMD` 不能接收
`stride(0)` 临时值的问题；本地 CPU 编译检查同时覆盖对应表达式和 Torch 适配头。
部署机器清理旧接口并重编译后，用户反馈服务功能正常。新增独立 NPU 用例未在本地
执行；不能将用户的服务验证表述为所有独立测试用例均已运行。

## 服务器验证

停止旧服务，切换到 `030_dspark_dev`，在容器的仓库中重新构建：

```bash
pip install -v -e . --no-deps --no-build-isolation
```

本次必须启用自定义算子编译（`COMPILE_CUSTOM_KERNELS` 默认是 `1`）。
仅切换 Python 源码或只重启旧二进制不会修复 DCUT 状态访问。
不要混用旧版扩展、旧版 `libcust_opapi.so` 与新版 DCUT 内核。

先运行算子回归：

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_dcut_state_stride.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_dcut_gdn_varlen.py
```

再沿用原启动脚本，保持 FA4、图模式、async scheduling、prefix caching，
开启 `enable_adaptive_verification=true`。同一提示词连续请求至少 10 次，
再测试不同提示词及少量并发，观察输出、接受率和动态预算日志。
最后关闭动态校验做一次对照。

若失败，请反馈编译或启动报错、当前 commit、前几轮请求输出，
以及 `[DSPARK-EAGER-AV/upstream]` 初始化和运行日志。

## 旧接口残留导致的启动段错误

新增状态行跨度属性后，旧 `autogen` ACLNN 接口可能继续被复用，导致新调用的
`stateRowStride` 被旧接口当成输出 Tensor 指针，在 `NnopbaseAddOutput/AppendShapeInfo`
处崩溃。备份并清理对应生成接口与 DCUT conv 发布缓存，重新生成、编译安装并完全重启
服务即可，无需修改 CMake 源码。完整命令见
[issue #35 操作步骤](https://github.com/singzhou/vllm-ascend/issues/35#issuecomment-6186313373)。

两项源码修复和编译兼容修正现已合入主开发分支；验证分支已删除，
[issue #35](https://github.com/singzhou/vllm-ascend/issues/35) 已关闭。
