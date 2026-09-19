# SPDX-License-Identifier: Apache-2.0
"""Compile real host dispatch and pure kernel window helpers without CANN.

This checks layout semantics, not NPU floating-point arithmetic or ABI/build.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def block(source, marker):
    start = source.index(marker)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_explicit_query_boundaries_override_decode_shape(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the source contract probe")
    host = (ROOT / "csrc/moe/causal_conv1d/op_host/causal_conv1d_tiling_validation.h").read_text()
    kernel = (ROOT / "csrc/moe/causal_conv1d/op_kernel/causal_conv1d.h").read_text()
    branch = block(host, "if (!qslAbsent && isDecodeMode && inputMode == 2)")
    helpers = "\n".join(
        block(kernel, marker)
        for marker in (
            "__aicore__ inline int32_t GetSeqTaskWindowMode",
            "__aicore__ inline SeqTaskWindow BuildSeqTaskWindowVarlen",
            "__aicore__ inline SeqTaskWindow BuildSeqTaskWindowDecode2D",
        )
    )
    source = (
        r"""
#include <cstdint>
#include <cassert>
#include <vector>
#define __aicore__
#define OP_CHECK_IF(cond, log, action) do { assert(!(cond)); } while(0)
enum {SEQ_TASK_WINDOW_MODE_VARLEN=0, SEQ_TASK_WINDOW_MODE_BATCH=1, SEQ_TASK_WINDOW_MODE_DECODE2D=2};
struct SeqTaskWindow { bool valid=false; int32_t start=0; int32_t len=0; };
struct Shape { int64_t t; int64_t GetDim(int i) { return i==0?t:64; } };
"""
        + helpers
        + r"""
int mode(int64_t tokens,int64_t rows,bool present=true) {
    bool qslAbsent=!present,isDecodeMode=true;
    int64_t inputMode=2,batch=tokens,dim=64,cuSeqlen=tokens,seqLen=1,qslSize=rows+1;
    Shape xShape{tokens};
"""
        + branch
        + r"""
    return GetSeqTaskWindowMode(inputMode);
}
void check(int tokens,std::vector<int32_t> qsl) {
    auto m=mode(tokens,qsl.size()-1);
    assert(m==SEQ_TASK_WINDOW_MODE_VARLEN);
    for(int i=0;i<qsl.size()-1;i++) {
        auto w=m==SEQ_TASK_WINDOW_MODE_VARLEN ? BuildSeqTaskWindowVarlen(qsl[i],qsl[i+1])
                                              : BuildSeqTaskWindowDecode2D(i);
        assert(w.start==qsl[i] && w.len==qsl[i+1]-qsl[i]);
        assert(w.valid==(qsl[i+1]>qsl[i]));
    }
}
int main() {
    check(8,{0,8});
    check(8,{0,1,2,3,4,5,6,7,8});
    check(8,{0,8,8,8,8,8,8,8,8});
    check(8,{0,3,6,6,6,6,6,6,6});
    check(9,{0,8,8,8,8,8,8,8,8});
    check(8,{0,8,8,8,8,8,8,8,8,8});
    assert(mode(8,8,false)==SEQ_TASK_WINDOW_MODE_DECODE2D);
}
"""
    )
    cpp, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    cpp.write_text(source)
    subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(executable)], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)
