# DSpark 动态校验（adaptive verification）

面向 vLLM 0.30 MRV2 的 DSpark 动态校验。正式路径使用上游 cost-argmax 预算、
设备端 survival top-k 和 ragged full graph；按 confidence 裁剪每个请求的 draft 后缀。
裁剪会改变每步验证的 token 数，不改变 target 对输出 token 的最终判定。

## 开启方式

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
# 0：原生注意力；1：Omni FIA sink；2：FA4。
export VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=2
vllm serve /path/to/Qwen3.6-27B \
  --tensor-parallel-size 4 --max-num-seqs 16 --max-model-len 4096 \
  --enable-prefix-caching --async-scheduling \
  --speculative-config '{"method":"dspark","model":"/path/to/DSpark",
    "num_speculative_tokens":7,"enable_adaptive_verification":true}'
```

`enable_adaptive_verification=true` 选择 Ascend AV manager，图模式默认 `ragged`，
GDN 请求轴固定为 `max_num_seqs`，host query 边界保留上游的容量上界，真实边界取自设备。
正常部署不需要 eager 实验开关，也不要添加 `--enforce-eager`。
FA4 模式需要在 worker 环境安装提供 scheduler metadata API 的 `flash_attn_npu_4`；
Omni 模式需要 `omni_custom_ops`。

`num_speculative_tokens` 必须与 draft checkpoint 的 `block_size` 一致。
Ascend AV 适配覆盖 K=1..15、PP=PCP=DCP=1、无 LoRA/DBO 的配置；不支持的配置
默认告警并退回上游 manager。显式指定 `VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV=1`
会对不支持的配置报错；该变量名称为历史兼容保留，不表示默认运行 eager。

## 正式路径的组成

- `av_config.py`：配置准入、ragged 默认模式与请求轴约束。
- `adaptive_verification.py`：复用上游预算与重分配策略，适配 Ascend streams/events，
  异步双缓冲 confidence，修复非有限值并中性初始化新请求槽位。
- `av_log.py`：按窗口聚合预算、裁剪比例、图模式和 confidence 变化；调用方异步暂存
  设备计数，读取上一窗口，避免日志排空设备队列。
- runner、GDN builder、Mamba state：固定地址 metadata、inactive tail 清理、零长度
  请求行、跨轮 accepted-state 选择及混合批次回退。
- D-Cut conv/GDN：设备端变长 query 和物理状态行跨度寻址，支持 full graph replay。

保留以下对照开关，默认部署均无需设置：

| 变量 | 默认值及作用 |
| --- | --- |
| `VLLM_ASCEND_DSPARK_AV_GRAPH` | 空值等价 `ragged`；`uniform`/`none` 仅用于显式对照 |
| `VLLM_ASCEND_DSPARK_AV_ADAPT` | 1；0 退回上游 manager |
| `VLLM_ASCEND_DSPARK_GDN_FIXED_AXIS` | 0；ragged 模式始终固定，不受此值影响 |
| `VLLM_ASCEND_DSPARK_AV_TP_BROADCAST` | 0；1 每步广播 confidence，供排查 rank 分歧 |
| `VLLM_ASCEND_DSPARK_AV_KEEP_PIECEWISE` | 0；1 保留上游强制的 FULL_AND_PIECEWISE |
| `VLLM_ASCEND_DSPARK_EAGER_UPSTREAM_AV` | 0；1 对不支持的配置严格报错 |
| `VLLM_ASCEND_DSPARK_EAGER_AV_LOG_INTERVAL` | 50；聚合日志间隔 |

`graph=none` 的显式对照仍保留合成 cost curve，因为没有图 profile；其预算不能用于
性能结论。正式图模式使用实测 cost curve。历史 `DSPARK-EAGER-AV` 日志标签保持兼容。

已移除同步 survival-threshold 实验 manager、独立阈值策略、
`VLLM_ASCEND_DSPARK_EAGER_SURVIVAL_THRESHOLD`、
`VLLM_ASCEND_DSPARK_AV_CPU_UPPER_BOUND` 及只为该实验存在的精确 host 边界回读。
运行脚本应删除这两个已退休变量；整网验证改用下文的 full-graph 脚本。

## 图与状态的正确性契约

1. 每个真实 decode 请求至少保留 anchor，只能裁剪 draft 后缀。
2. 上轮 accepted-state 选择不能按本轮裁剪后长度截断，包括本轮零 draft 的请求。
3. padding 请求行保持固定轴，以零长度和无效状态索引成为 no-op；不能写真实状态/KV。
4. capture 与 replay 使用固定地址 buffer，批次收缩时清理 inactive tail；每个 KV 组
   分别清理其捕获缓冲区，即使共享批次计划也不能省略清理。
5. target uniform、target ragged 与 drafter descriptor 隔离。混合 prefill/decode
   不能误入纯 speculative ragged 图；图回退保留 Ascend PCP 分发和 DP 同步结果。
6. confidence 必须每次 draft replay 重算。图能捕获、输出合法，不代表 confidence
   没有冻结；日志的 `conf_moved` 用于检查该情况。
7. draft context 未使用 tail 的 slot mapping 必须无效，避免旧 hidden state 回写 KV。
8. FA4 的虚拟 page ID 必须反映真实物理 stride，K/V 不得重排破坏混合缓存块隔离。
9. DCUT 状态行通过真实 `stride(0)` 寻址，内部维度要求连续，不能按逻辑行大小猜地址。
10. TP rank 必须形成相同裁剪计划；默认 confidence head 输出已在组内归约。

## 编译与旧产物处理

```bash
pip install -v -e . --no-deps --no-build-isolation
```

D-Cut 自带 Host/Kernel 源码，不依赖 0.30 已移除的旧算子目录。
conv 编译单元独立定义 layout 策略并隔离 Host helper 命名空间，防止不同策略的 inline
函数发生链接符号冲突。通用 conv 算子的行为保持原状。

复用旧构建目录时，两种缓存残留须分别处理：

- `store_kv_block_metadata_obj` 缺失：清理旧缓存项，再重新安装；无需修改编译工程。

  ```bash
  cmake -U AICPU_CUST_OBJ_TARGETS -S csrc -B csrc/build
  ```

- `aclnnDcutCausalConv1dGetWorkspaceSize` 在 `NnopbaseAddOutput/AppendShapeInfo`
  段错误：旧自动生成接口缺少 `stateRowStride`，导致参数 ABI 错位。
  备份移走对应 `csrc/build/autogen/*dcut_causal_conv1d*`、
  `*dcut_recurrent_gated_delta_rule*` 文件，以及对应芯片目录下
  `bin/dcut_causal_conv1d` 和 `cache_actions/dcut_causal_conv1d` 发布产物，
  执行 `cmake --build csrc/build --target opbuild_gen_default -j8`，再重新安装。
  完整备份脚本和接口签名检查见
  [issue #35 操作步骤](https://github.com/singzhou/vllm-ascend/issues/35#issuecomment-6186313373)。

重新安装后必须完全退出旧服务进程并重新启动，避免继续使用旧动态库和函数指针。

## 验证

CPU 契约、物理地址和独立数学参考：

```bash
python -m pytest --noconftest -q \
  tests/ut/spec_decode/test_adaptive_verification.py \
  tests/ut/ops/test_dcut_cpu_reference.py \
  tests/ut/ops/test_dcut_state_stride.py \
  tests/ut/attention/test_fa4_kv_cache_layout.py
```

NPU 算子与 replay 对照（需要真实 NPU 和已重编译的算子）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest --noconftest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_dcut_gdn_varlen.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_dcut_state_stride.py
```

整网 full-graph 验证与吞吐测量分开执行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_TEST_QWEN36_MODEL=/path/to/Qwen3.6-27B
export VLLM_TEST_DSPARK_MODEL=/path/to/DSpark
python examples/dspark_adaptive_verify.py
python examples/dspark_adaptive_verify_throughput.py --concurrency 16 --repeats 5
```

正确性脚本在 fixed-K full graph 和配置驱动的 ragged AV full graph 间比较 token，
同时测量 baseline 自身的重复性。没有裁剪的数据不能证明裁剪路径正确，返回
`INCONCLUSIVE`；confidence 冻结也会失败。吞吐脚本单独报告 TPS、接受长度及实测
cost curve 的可达变化范围；接受率降低本身不是错误，也不能单凭 TPOT 判断收益。

此前服务器已完成 D-Cut 编译及变长算子验证；用户已确认 #34 连续请求无乱码且接受率
正常，并在 #35 清理旧接口、重编译后确认功能正常。历史验证不等于本次清理后的新一轮
NPU 整网测试；本次整理的本地验证范围见提交说明及适配介绍 Discussion。
