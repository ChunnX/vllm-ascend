# SPDX-License-Identifier: Apache-2.0
"""Compile actual DCUT address expressions on CPU to check physical-page isolation.

This exercises the kernel formulas without Ascend headers; the corresponding
NPU tests cover tiling, state mutation and graph replay through the real ops.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def address_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the kernel address regression")
    directory = tmp_path_factory.mktemp("dcut_address")
    recurrent = ROOT / "csrc/attention/dcut_recurrent_gated_delta_rule/vendor/op_kernel"
    conv = ROOT / "csrc/moe/dcut_causal_conv1d/vendor/op_kernel"
    expressions = []
    for name in ("recurrent_gated_delta_rule.h", "arch35/recurrent_gated_delta_rule.h"):
        source = (recurrent / name).read_text()
        # Initial state load, prefetch of a later V tile, and per-token snapshot.
        expressions.extend(re.findall(r"(?:uint64_t )?nextStateOffset\s*=\s*([^;]+);", source))
        expressions.extend(re.findall(r"uint64_t curStateOutOffset\s*=\s*([^;]+);", source))
    assert len(expressions) == 6
    rec_count = len(expressions)
    for name in ("causal_conv1d.h", "causal_conv1d_fn_tasks.h"):
        source = (conv / name).read_text()
        expressions.extend(
            re.findall(r"const int64_t (?:stateBaseOffset|srcOffset[01]|dstOffset[01])\s*=\s*([^;]+);", source)
        )
    assert len(expressions) > rec_count
    host = (recurrent.parent / "op_host/arch35/recurrent_gated_delta_rule_tiling_arch35.cpp").read_text()
    dense_row_expression = re.search(r"const int64_t denseStateRow\s*=\s*([^;]+);", host).group(1)
    lines = [f"if (({dense_row_expression.replace('tilingData_.', 'recurrentTiling.')}) != 512) return 2;"]
    for index, expression in enumerate(expressions):
        # Bind just the runtime inputs; retain the actual kernel expressions.
        expression = expression.replace("StateRowStride()", "rowStride")
        lines.append(f'std::cout << {index} << " " << ({expression}) << "\\n";')
        if index == rec_count - 1:
            lines.append("rowStride = convStride; tiling.stateRowStride = rowStride;")
    program = (
        """#include <cstdint>
#include <cstdlib>
#include <iostream>
int main(int argc, char** argv) {
    uint64_t rowStride = std::strtoull(argv[1], nullptr, 10);
    uint64_t convStride = std::strtoull(argv[2], nullptr, 10);
    uint64_t stateOffset = 3, head_i = 1, realV_ = 16, realK_ = 16, NV_ = 2;
    uint64_t nextVOffset = 7, v_i = 5, stateTokenIdx = 0;
    struct Indices { int32_t GetValue(uint64_t) const { return 3; } } ssmStateIndicesGm_;
    struct Tiling { int64_t stateRowStride; int64_t stateLen = 7; } tiling;
    struct RecurrentTiling { uint32_t nv = 2, dv = 16, dk = 16; } recurrentTiling;
    auto* tilingData_ = &tiling;
    int64_t cacheIdx = 3, stateLen = 7, dim = 32, channelStart = 5, srcPos0 = 2, srcPos1 = 3;
"""
        + "\n".join(lines)
        + "\n}\n"
    )
    source = directory / "probe.cpp"
    source.write_text(program)
    executable = directory / "probe"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)], check=True, capture_output=True, text=True
    )
    return executable, rec_count


@pytest.mark.parametrize("padded", [False, True])
def test_actual_kernel_addresses_stay_in_selected_physical_row(address_probe, padded):
    executable, rec_count = address_probe
    recurrent_row, conv_row = 2 * 16 * 16, 7 * 32
    recurrent_stride = recurrent_row * 3 + 32 if padded else recurrent_row
    conv_stride = conv_row * 3 + 32 if padded else conv_row
    result = subprocess.run(
        [str(executable), str(recurrent_stride), str(conv_stride)], check=True, capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        index, offset = map(int, line.split())
        row_size, stride = (recurrent_row, recurrent_stride) if index < rec_count else (conv_row, conv_stride)
        assert 3 * stride <= offset < 3 * stride + row_size, (index, offset, stride)


def test_torch_adapters_pass_lvalue_arguments(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the adapter syntax regression")
    source = tmp_path / "adapters.cpp"
    adapters = [
        ROOT / "csrc/attention/dcut_recurrent_gated_delta_rule/dcut_recurrent_gated_delta_rule_torch_adpt.h",
        ROOT / "csrc/moe/dcut_causal_conv1d/dcut_causal_conv1d_torch_adpt.h",
    ]
    source.write_text(
        "#include <ATen/ATen.h>\n#include <tuple>\n"
        "template<class... Ts> auto ConvertTypes(Ts&... args) { return std::tie(args...); }\n"
        "#define EXEC_NPU_CMD(name, ...) static_cast<void>(ConvertTypes(__VA_ARGS__))\n"
        + "\n".join(f'#include "{adapter}"' for adapter in adapters)
    )
    includes = Path(torch.__file__).parent / "include"
    result = subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", f"-I{includes}", str(source)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
