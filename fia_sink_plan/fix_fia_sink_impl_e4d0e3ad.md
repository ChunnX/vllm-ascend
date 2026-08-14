# FIA Sink 实现修复说明（基于 `e4d0e3ad`）

## 1. 修复结论

本次修复针对 `c357e30b` / `e4d0e3ad` 审视中确认的阻断问题。当前已验证、优先服务的
模型形态是：

- speculative method 为 `dspark`；
- draft attention 为非因果；
- GQA，`head_size=128`，GQA ratio 不超过 64；
- 无 sliding window、无 learnable sink；
- KV cache 使用分页 BBH 视图；
- 通过 device seq tensors + AICPU tiling 消除 `seq_lens.tolist()` 同步。

这里的 GQA / D=128 等只是当前业务场景和验证基线，**不是框架侧能力白名单**。底层
`omni-ops/.../docs/npu_fused_infer_attention_sink.md` 覆盖更广的 head topology、D 维度和输入组合；Python
接入层不应复制一份不完整且容易过时的算子约束。最终实现只限定 DSpark 非因果草稿路由，
具体 shape、dtype、layout 及组合是否合法交由 metadata op / FIA Sink 计算算子校验并报错。

默认开关仍为关闭：

```bash
VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=0
```

## 2. 代码修复

### 2.1 严格限制 DSpark 范围

只有下面条件同时成立才准备 sink 路径：

```python
VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK == 1
speculative_config.method == "dspark"
speculative_config.parallel_drafting is True
common_attn_metadata.causal is False
```

metadata builder 与 impl 使用同一个 `use_fia_sink` 路由结论，不再在 impl 侧二次拒绝后回落普通
FIA，因此不会出现 builder 已将 host-side 列表置为 `None`、impl 又拒绝 sink 的 split-brain。

`_enable_dspark_fia_sink()` 只加载并检查自定义算子是否注册，不再解析 attention layers，也不再
限制 GQA、`head_size=128` 或 GQA ratio。这样做是有意的：

- 算子能力文档支持的范围远大于当前 DSpark 验证配置，例如 Q\_S>1 时 D 可到 512（具体还受
  dtype、对齐、PageAttention 布局等综合约束）；
- 框架侧手写白名单会错误拒绝底层已经支持的新模型/新规格，并可能随算子演进而失真；
- 不支持的 head 拓扑 / D 维度 / dtype / layout 组合由
  `_npu_fused_infer_attention_sink_metadata` 或 `npu_fused_infer_attention_sink`
  返回原生校验错误，错误来源更准确。

保留的框架侧门禁只有：环境开关、`method="dspark"`、`parallel_drafting=True` 和非因果 draft
metadata；这些条件描述的是接入路径适用范围，而不是底层算子能力。

**例外**：`sliding_window` 与 `learnable sink` 仍由 impl 侧 `_use_fia_sink()` 显式拒绝并报错，
不能下放给算子——sink 路径把 `sparse_mode` 写死为 0（非因果全注意力）且不转发
`atten_mask` / `learnable_sink`，算子收不到这些信息，无从校验，直接走 sink 会静默算成全注意力。

### 2.2 启动期加载并检查自定义算子

sink 首次用于非因果 draft metadata 前会执行：

```python
import omni_custom_ops
```

并检查以下 torch ops 已注册：

- `custom::_npu_fused_infer_attention_sink_metadata`
- `custom::npu_fused_infer_attention_sink`

wheel、共享库或 CANN OPP 环境缺失时会在模型初始化/metadata 构建阶段给出明确错误，避免首次
attention 时才出现难以定位的 `torch.ops.custom` 属性错误。

### 2.3 修复 FULL aclgraph padding 的 TND 约束

DSpark 每个请求的 query token 数固定。sink 路径不再直接使用 producer 留下的 plateau
`query_start_loc`，而是根据静态 shape 在 device 上构造累计 Q 长度：

```text
num_tokens=32, num_reqs=4 -> actual_seq_qlen=[8, 16, 24, 32]
```

同时将 padding 请求的 KV 长度从 0 映射为 1：

```text
seq_lens=[19, 23, 0, 0] -> actual_seq_kvlen=[19, 23, 1, 1]
```

因此始终满足：

```text
actual_seq_qlen[-1] == query.shape[0]
len(actual_seq_qlen) == len(actual_seq_kvlen) == block_table.shape[0]
actual_seq_kvlen > 0
```

该过程只包含 device tensor 运算，不引入 device-to-host 同步。

### 2.4 metadata 每个 forward/signature 只计算一次

转换后的 Q/KV lengths 与 metadata tensor 缓存在当前 vLLM forward context 中，cache key 包含：

- seq_lens buffer 地址；
- token/request 数；
- query/KV head 数；
- head size 与 block size。

同一 attention signature 的首层执行一次 metadata op，后续层复用结果。aclgraph capture 时该
metadata producer 只被捕获一次；replay 时图会先重放 metadata producer，再执行所有 sink consumer。
不再使用 `e4d0e3ad` 中“每层独立 buffer + 每层无条件 metadata + copy_”的无效共享方案。

## 3. 测试补充

更新 `tests/ut/attention/a2/test_attention_v1.py`，覆盖：

1. 只有 `method="dspark"` 能命中，DFlash/draft_model 不命中；
2. FULL bucket padding 后累计 Q 长度覆盖完整 query；
3. padding KV 长度从 0 归一为 1；
4. 非整除的非 uniform query batch 明确失败；
5. 同一 forward/signature 的 metadata compute 只调用一次；
6. 缺少 `omni_custom_ops` 时给出明确错误；
7. builder 只检查并注册自定义算子，不因 GQA/MHA、head dimension 等模型形态提前拒绝。

## 4. 本地验证结果

已完成：

```text
PASS  Python AST / py_compile
PASS  git diff --check
```

本机没有安装 `pytest`、vLLM runtime、torch/torch_npu 和 NPU，因此未伪造运行结果。以下验证必须在
安装了匹配 omni-ops wheel/OPP 的 Ascend 容器中执行。

本机命令限制记录：

```text
python -m pytest ...  -> No module named pytest
ruff check ...        -> command not found: ruff
bash format.sh ci     -> pre-commit is not installed
```

## 5. NPU 验证门禁

### 5.1 单元测试

```bash
pytest -sv tests/ut/attention/a2/test_attention_v1.py -k fia_sink
```

### 5.2 单算子精度

按 `compile_omni_ops_fia_sink.md` 安装两个算子，执行其中单算子脚本，并与普通 FIA/朴素
non-causal GQA attention 比较，fp16 最大误差目标 `< 1e-2`。

### 5.3 DSpark eager

```bash
VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=1 \
pytest -sv tests/e2e/pull_request/one_card/spec_decode/test_dspark.py
```

先使用 eager/非 FULL 模式确认：

- 请求成功且输出非空；
- greedy token 与关闭 sink 时一致；
- accepted-count 序列与关闭 sink 时一致；
- profile 中不再出现 draft `seq_lens.tolist()` 同步。

### 5.4 FULL aclgraph

必须覆盖实际请求数小于 capture bucket 的场景，例如 `2/4/8` 个请求 replay `16` 请求图，确认：

- capture 和 replay 均成功；
- 日志出现 `Replaying aclgraph`；
- 无 FIA 参数错误 `561002`；
- padding 请求不会写 KV cache；
- metadata op 每个 step/signature 仅执行一次；
- 与关闭 sink 的 greedy token 和 accepted-count 序列一致。

### 5.5 性能判据

只有同时满足以下条件才可认为优化有效：

- device-to-host sequence-length 同步消失；
- AICPU metadata 派发由每层一次降为每步/signature 一次；
- 端到端 TPOT 或吞吐有稳定改善。

单个 profiler frame 变短不能替代端到端性能数据。

## 6. 已知边界

| 项目 | 状态 |
|---|---|
| DSpark GQA / non-causal / D=128 | 当前验证基线；已实现，待 NPU 实测 |
| eager | 已实现，待 NPU 实测 |
| FULL aclgraph padding | 已修复 device metadata，待 NPU replay 实测 |
| 其他 head topology / head dimension | Python 不设白名单，由底层算子校验；需按目标模型实测 |
| sliding window / learnable sink 等组合 | impl 侧 `_use_fia_sink()` 显式报错拒绝（`sparse_mode` 写死、无法下放）；不静默走 sink |
| DFlash / draft_model / P-EAGLE | 不受该开关影响 |
| EP / flashcomm1 | 非本次 draft attention 修改范围 |
| real-weight gate | 当前本机无 NPU/模型，尚未执行 |

在完成 5.2～5.4 前，本实现不能宣称已通过 Ascend 端到端验收。
