# 910B4 TP=4 服务器测试清单

## 当前代码与已知结果

- 分支：`dspark_adaptive_eager_b5180b`，vLLM 0.28.0 / MRV2。
- 硬件：910B4，整机 8×32GB，只使用分配的四张开发卡。
- 两个 D-Cut 算子目录已经完整移入：recurrent 在 `a39b5f037`，Conv1D 在
  `e70733e13`。它们与 PR #15207 快照
  `645e05ac71960cc6bf01faca8aba7037dd752002` 逐字节一致，注册、meta 和构建列表
  也已接入。不再重复 cherry-pick 整个 PR；没有修改两个算子的 host/kernel。
- 通用 Conv1D 已恢复 `b5180b`。当前 speculative eager 路径使用两个 D-Cut 接口。
- 用户上机结果：D-Cut Conv1D 的 `[8,0,0,0,0,0,0,0]` 用例仍失败，其余四项通过。
  这不等于确认源码对应的二进制已被加载，更不等于 FULL 已验证。
- 本清单不构建、不安装、不修复算子；优先复用已安装版本，保留失败证据。

## 运行前

1. 激活原来能运行 Qwen3.6 DSpark 的 NPU Python/CANN 环境，在本分支仓库根目录执行。
2. 确认四张卡归你使用并已空闲，禁止和服务或其他测试并行抢占。
3. 设置模型路径，保留当前正确的 HCCL/CANN 配置；不要因为示例修改 SOC_VERSION。
4. `pip install -e .` 和切分支不会自动更新 `.so` / OPP 二进制。
   `git rev-parse HEAD` 只证明源码版本，不能证明算子版本。
5. 脚本独立运行 pytest，关闭外部插件自动加载和仓库 conftest；仅运行下列自包含测试。
   不使用 pytest-xdist。它不是完整仓库 UT/CI 的替代品。

```bash
git switch dspark_adaptive_eager_b5180b
git pull --ff-only
export VLLM_TEST_QWEN36_MODEL=/实际路径/Qwen3.6-27B
export VLLM_TEST_DSPARK_MODEL=/实际路径/DSpark
# 下面 IDs 必须换成你实际分配的四张卡，不要盲目使用 0,1,2,3。
bash tools/run_dspark_eager_validation.sh --stage all --devices IDs
```

`--python /path/to/python` 可指定已安装环境的解释器，默认 `python3`。
默认日志目录由 `mktemp` 创建；用 `--log-dir /path/to/new-directory` 指定一个
尚不存在的目录。脚本不会覆盖旧结果。

## 测试分层

| 阶段 | 文件 | 用例数 | 验证内容 |
| --- | --- | ---: | --- |
| CPU | `tests/ut/spec_decode/test_eager_survival_verification.py` | 29 | survival、预算、请求重排/slot 复用、准确边界、eager 配置、历史 accepted |
| CPU | `tests/ut/ops/test_dcut_cpu_reference.py` | 8 | 独立 golden 的手算、跨轮状态、分组 head、空行 |
| 单卡算子 | `tests/e2e/nightly/single_node/ops/singlecard_ops/test_eager_gdn_varlen.py` | 5 | 四组 Conv1D 布局＋一组 recurrent 多轮状态，真实 NPU 对独立 golden |
| TP=4 模型 | `tests/e2e/nightly/single_node/spec_decode/test_qwen36_dspark_eager_survival.py` | 3 | fixed K=7 与阈值 0/0.4/1 的 greedy token 对照 |

`tests/ut/helpers/dcut_reference.py` 是参考实现，不单独运行。
旧 `test_causal_conv1d_varlen_layout.py` 已随原算子修改撤回，不再执行。
这套测试尚不验证 confidence 权重校准、随机采样分布、长时压力或 FULL 图。

分别运行：

```bash
bash tools/run_dspark_eager_validation.sh --stage cpu
bash tools/run_dspark_eager_validation.sh --stage ops --devices IDs
bash tools/run_dspark_eager_validation.sh --stage model --devices IDs
```

- `cpu` 不使用 NPU；37 项通过只证明 CPU 合同与参考数学。
- `ops` 只使用指定四张卡中的第一张，5 项全部执行，不因首个断言失败就停止。
- `model` 使用全部四张卡、TP=4，各 threshold 和 fixed/adaptive 顺序创建进程。
  每个子进程超时 1800 秒；保留完整输出，不只保留最后几行报错。
- `all` 按 CPU → 环境检查 → 算子 → 模型执行。CPU、环境或算子失败即返回非零，
  不将已知失败伪装成整体验收通过。
- 当前空行用例已知失败，因此 `all` 预计会停在算子阶段。如果希望独立定位 eager
  模型，可显式运行 `--stage model`；其通过也不能关闭空行问题或作为 FULL 验收。

## 当前失败应如何理解

`[8,0,0,0,0,0,0,0]` 是八个请求槽位、八个 token、仅一个真实请求。
固定 GDN 请求轴的图 padding、batch 收缩、子批补齐时可能出现这种形状。
零 draft 的真实请求仍有一个 anchor，不等于零长度空行。

当前 eager 若只传真实请求，每行至少一个 anchor，则 T=B 只能是全体长度为 1。
因此该失败与普通无空行 eager 能否正确运行需要分开验证；后续 FULL 仍必须解决它。
共享 host 的 `T=B` 布局判断是源码层面的重点嫌疑，实际二进制版本尚待核实。
不要删掉这个用例、改成 xfail、放宽容差，或仅凭其它四项通过就宣称 GDN 全部正确。

## 结果与回传

脚本会打印日志目录，包含：

- `source.txt`：源码 SHA、工作区状态、Python、所选阶段和设备。
- `environment.log`：包版本、模块加载路径、扩展 `.so` 路径和 SHA256、两个接口
  schema、OPP 路径与设备型号。它是来源线索，不是底层 tiling/kernel 已匹配源码的证明。
- `summary.tsv`：每个已执行阶段的退出码；没有记录的阶段就是未运行。
- `cpu.log/xml`、`ops.log/xml`、`model.log/xml`：完整 pytest 输出和 JUnit 结果。
- `model-processes/`：各阈值对应的 `adaptive.log`、`fixed.log`，包含 worker 调试输出。

回传整个结果目录，并补充实际 CANN 版本和算子安装命令/构建时间。
若报缺少接口，先确认当前解释器加载了哪个注册扩展；若接口存在但数值失败，
保留源码和二进制信息再分析，不要立即重新编译覆盖当前证据。
