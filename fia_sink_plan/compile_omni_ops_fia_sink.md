# 交付件 1：编译 omni-ops 中的两个 Sink 算子，供单算子脚本调用 `npu_fused_infer_attention_sink`

## 目标

在 `/opt/zsy/omni-ops` 仓库中，编译并安装下面两个算子，使得在 NPU 宿主环境里，一个单算子 Python 脚本可以直接调用：

```python
torch.ops.custom._npu_fused_infer_attention_sink_metadata(...)  # AICPU，产出 meta_data
torch.ops.custom.npu_fused_infer_attention_sink(...)            # 主注意力算子，消费 meta_data
```

这两个算子分属两层，都需要编译：

| 层 | 算子 | 源码目录 | 产物 |
|---|---|---|---|
| AscendC 算子库（CANN opp 插件） | `AiInfraFusedInferAttentionSink`（含 AICORE kernel + host tiling） | `inference/ascendc/src/ops-transformer/attention/ai_infra_fused_infer_attention_sink` | 一个 `.run` 自解压包 |
| AscendC 算子库 | `AiInfraFusedInferAttentionSinkMetadata`（AICPU kernel） | `inference/ascendc/src/ops-transformer/attention/ai_infra_fused_infer_attention_sink_metadata` | 并入同一个 `.run` 包 |
| PyTorch 扩展（PTA） | `npu_fused_infer_attention_sink` / `npu_fused_infer_attention_sink_v2` | `inference/ascendc/torch_ops_extension/omni_custom_ops/ops_transformer/attention/fused_infer_attention_sink` | `omni_custom_ops` wheel |
| PyTorch 扩展（PTA） | `_npu_fused_infer_attention_sink_metadata` | `.../attention/fused_infer_attention_sink_metadata` | 并入同一个 wheel |

> 注意：`_npu_fused_infer_attention_sink_metadata` 的 CMake 里写了
> `add_modules_sources_aicpu(DEPENDENCIES ai_infra_fused_infer_attention_sink)`，
> 即 metadata 的 AICPU 依赖 sink 主算子的 AICPU 公共代码，所以这两个算子的 **AscendC 层必须一起编译**。

---

## 0. 前置条件

- 编译必须在 **NPU 宿主环境**（A2/A3/A5 + CANN 开发套件）进行，不能在本机的 macOS 上直接编译。
- 需要 CANN 工具链里的 `bisheng` 编译器（`build.sh` 会做 `which bisheng` 检查）。
- 需要与 CANN 匹配的 `torch_npu`、`torchair`（PTA 编译与运行期转换都依赖）。
- 参考仓库自带说明：`/opt/zsy/omni-ops/inference/ascendc/README.md`。

设置环境变量（按实际安装路径）：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
# 或者按 README 建议：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

芯片类型取值（`build.sh -c`）：`ascend910b`（A2）、`ascend910_93`（A3）、`ascend950`（A5）。
`config.ini` 里 metadata 算子标注了 `aicore_versions = ascend910b, ascend910_93`，这只是门禁识别用，不阻塞其他 soc 的编译（若不匹配可加 `--disable-check-compatible` 跳过版本校验）。

---

## 1. 编译 AscendC 算子包（`.run`）

```bash
cd /opt/zsy/omni-ops/inference/ascendc

# 只编译这两个算子（用分号分隔），-c 指定芯片
bash build.sh \
  -n 'ai_infra_fused_infer_attention_sink;ai_infra_fused_infer_attention_sink_metadata' \
  -c ascend910_93
```

- `-n` 值必须是算子目录名：`src/ops-transformer/attention/` 下的
  `ai_infra_fused_infer_attention_sink` 与 `ai_infra_fused_infer_attention_sink_metadata`。
- 编译成功后会在 `output/` 生成自解压包：
  `CANN-omni_custom_ops--linux.<arch>.run`。
- 若报版本校验失败，追加 `--disable-check-compatible`：
  ```bash
  bash build.sh -n '...;...' -c ascend910_93 --disable-check-compatible
  ```

## 2. 安装 `.run` 到 CANN opp

```bash
cd /opt/zsy/omni-ops/inference/ascendc/output
chmod +x CANN-omni_custom_ops--linux.<arch>.run
./CANN-omni_custom_ops--linux.<arch>.run --quiet \
  --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp

# 让 aclnn/图编译能发现自定义算子
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/omni_custom_transformer/bin/set_env.bash
```

安装后，自定义算子会出现在
`/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/omni_custom_transformer/` 下（含 op_api、op_host、op_kernel、op_kernel_aicpu 等）。

---

## 3. 编译并安装 PTA wheel（`omni_custom_ops`）

```bash
cd /opt/zsy/omni-ops/inference/ascendc/torch_ops_extension
bash build_and_install.sh
```

`build_and_install.sh` 实际执行：
```bash
python3 setup.py build bdist_wheel
pip3 install dist/omni_custom_ops-1.0-*.whl --force-reinstall
```

`setup.py` 用 `glob` 把 `omni_custom_ops/*/*/*/csrc/*.cpp` 全部编进一个扩展
`omni_custom_ops.custom_ops_lib`，所以 wheel 会带上所有算子的 torch 绑定，
其中包含我们需要注册的：

- `torch.ops.custom.npu_fused_infer_attention_sink`（`PrivateUse1` + `Meta` 实现）
- `torch.ops.custom.npu_fused_infer_attention_sink_v2`
- `torch.ops.custom._npu_fused_infer_attention_sink_metadata`

同时 `converter/` 下的 `@register_fx_node_ge_converter` 会在 `import` 时注册
图模式（torchair）的转换器，是后续 aclgraph 入图的前提。

---

## 4. 验证：单算子脚本

新建 `test_fia_sink.py`，覆盖 DSpark 草稿模型的实际形态：**GQA + 非因果(sparse_mode=0) + head_dim=128 + 分页 KV(BBH)**。

```python
import torch
import torch_npu
import omni_custom_ops   # noqa: F401  触发 custom ops 注册与 converter 注册

B = 2             # num_reqs
Q = 4             # num_query_per_req
T = B * Q         # 总 query token 数
Nq = 32           # num_query_heads
Nkv = 4           # num_kv_heads（GQA，Nq % Nkv == 0，比值 <= 64）
D = 128           # head_dim_qk == head_dim_v == 128
block_size = 128
num_blocks = 8

query = torch.randn(T, Nq, D, dtype=torch.float16, device="npu")
# 分页 KV，BBH 布局：(blocknum, blocksize, Nkv*D)
key = torch.randn(num_blocks, block_size, Nkv * D, dtype=torch.float16, device="npu")
value = torch.randn(num_blocks, block_size, Nkv * D, dtype=torch.float16, device="npu")
block_table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device="npu")

# TND 下 actual_seq_qlen 是累加值，单调非递减，元素个数 == batch
actual_seq_qlen = torch.tensor([Q, T], dtype=torch.int64, device="npu")
# 分页场景下 actual_seq_kvlen 是每个 batch 的真实长度（非累加）
actual_seq_kvlen = torch.tensor([128, 128], dtype=torch.int64, device="npu")

# 1) AICPU 生成 tiling 元数据（输入全部在 device 侧，无 host 同步）
meta = torch.ops.custom._npu_fused_infer_attention_sink_metadata(
    Nq, Nkv, D, D,
    actual_seq_lengths=actual_seq_qlen,
    actual_seq_lengths_kv=actual_seq_kvlen,
    batch_size=B,
    sparse_mode=0,               # 非因果
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="TND",
    input_layout_kv="BnBsH",     # (blocknum, blocksize, H) == BBH
    sink_num=0,
    block_size=block_size,
    aic_core_num=24,
    aiv_core_num=48,
)

# 2) 主算子，直接传 device 侧 seqlens
out, lse = torch.ops.custom.npu_fused_infer_attention_sink(
    query, key, value,
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_kvlen=actual_seq_kvlen,
    block_table=block_table,
    num_query_heads=Nq,
    num_key_value_heads=Nkv,
    softmax_scale=1.0 / (D ** 0.5),
    input_layout="TND",
    sparse_mode=0,               # 非因果，不传 atten_mask
    block_size=block_size,
    sink_number=0,               # 无 sink token
    meta_data=meta,
)

print("out:", out.shape, out.dtype)
print("lse:", lse.shape, lse.dtype)
```

预期：`out` 为 `(T, Nq, D)` 的 float16，`lse` 为 `(1,)`（`return_softmax_lse=False` 时）。

验证要点（对照 torch 朴素实现可选做精度比对）：
- 非因果：`sparse_mode=0` 且不传 `atten_mask`，等价全量 attention。
- GQA：`num_query_heads=32 / num_key_value_heads=4`。
- 分页：`block_table` + `block_size=128`，`actual_seq_kvlen` 为逐 batch 真实长度。

---

## 5. 常见问题

- **`which bisheng` 失败**：CANN 包未 `source setenv`，或装的不是完整开发套件。
- **版本校验失败**：加 `--disable-check-compatible`。
- **`import omni_custom_ops` 报 `custom_ops_lib` 找不到**：wheel 未安装或未 `--force-reinstall`；确认 `python3` 与 `torch_npu` 的 Python 版本一致。
- **运行时报算子未注册 / converter 未注册**：`.run` 未安装到 opp 且未 `source set_env.bash`；或先 `import omni_custom_ops`（触发 `TORCH_LIBRARY_IMPL` 注册）再调用。
- **AICPU 报错**：确认 `-n` 同时编译了 `..._sink` 与 `..._sink_metadata`（metadata 的 AICPU 依赖 sink 的公共代码）。

---

## 参考

- 仓库 README：`/opt/zsy/omni-ops/inference/ascendc/README.md`（第 254~308 行是编译/安装命令）
- 编译脚本：`/opt/zsy/omni-ops/inference/ascendc/build.sh`（`-n`/`-c`/`--disable-check-compatible`）
- PTA 编译：`/opt/zsy/omni-ops/inference/ascendc/torch_ops_extension/build_and_install.sh`
- 算子接口文档：
  - `.../ai_infra_fused_infer_attention_sink/docs/npu_fused_infer_attention_sink.md`
  - `.../ai_infra_fused_infer_attention_sink_metadata/docs/npu_fused_infer_attention_sink_metadata.md`
