# 910B4 TP=4 测试顺序

在 `vllm-ascend` 仓库根目录、已安装 NPU 依赖的 Python 环境中执行。
以下卡号仅以分配到 `0,1,2,3` 为例，请替换成自己的四张开发卡。不要并行跑测试。

## 1. CPU UT（已通过，可不重复）

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest --noconftest -o addopts='' -sv \
  tests/ut/spec_decode/test_eager_survival_verification.py \
  tests/ut/ops/test_dcut_cpu_reference.py
```

共 37 项，检查裁剪策略、请求映射、metadata 和独立数学参考。

## 2. 单卡 GDN 算子

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest --noconftest -o addopts='' -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_eager_gdn_varlen.py
```

共 5 项，调用已安装的 D-Cut Conv1D/recurrent，与独立 golden 比较。
当前已知 `[8,0,0,0,0,0,0,0]` 空行用例失败，其余四项曾通过；继续保留该失败，
不改算子、不跳过、不放宽误差。这个边界主要涉及固定请求轴的图 padding。

## 3. TP=4 模型对照

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_TEST_QWEN36_MODEL=/实际路径/Qwen3.6-27B
export VLLM_TEST_DSPARK_MODEL=/实际路径/DSpark
python3 -m pytest --noconftest -o addopts='' -sv \
  tests/e2e/nightly/single_node/spec_decode/test_qwen36_dspark_eager_survival.py
```

共 3 项，固定 TP=4，依次对照 fixed K=7 与阈值 0/0.4/1 的 greedy 输出。
每个模型子进程超时 1800 秒，完整日志路径会打印到终端。
模型路径未设置时测试会 skip，不能算通过。

正常验收应先通过算子测试，再运行模型。若现在独立运行模型，仅用于定位无空行
的 eager 路径；模型通过也不能关闭已知空行问题，更不代表 FULL 已验证。

## 注意事项

- 所有命令直接调用 pytest，不再经过 `tools/` 下的环境检查或批量脚本。
- `--noconftest` 只用于这里的自包含测试，不是完整仓库 UT 的运行方式。
- 不使用 `pytest-xdist` 或 `-n`，模型对照顺序占用同一组四张卡。
- 当前两个 D-Cut 算子源码已完整接入，与快照
  `645e05ac71960cc6bf01faca8aba7037dd752002` 一致；通用 Conv1D 已恢复 baseline。
- `git pull` 不更新 editable 安装中已有的算子二进制，当前先复用已编译版本。
- 回传 pytest 的失败堆栈及最后汇总；模型失败时附上对应子进程日志。
