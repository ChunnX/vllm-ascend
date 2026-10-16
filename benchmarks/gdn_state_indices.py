# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Compare graph-buffer state preparation; excludes model execution.

python benchmarks/gdn_state_indices.py --device 4 --groups 3
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.mamba.gdn_state_indices import fused_gdn_state_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    tables = [torch.arange(44, dtype=torch.int32, device="npu").view(1, 44) for _ in range(args.groups)]
    lengths = torch.tensor([130], dtype=torch.int32, device="npu")
    rows = torch.tensor([0], dtype=torch.int64, device="npu")
    outputs = [torch.empty((1, 8), dtype=torch.int32, device="npu") for _ in tables]

    def reference():
        for table, output in zip(tables, outputs):
            starts = (lengths - 1) // 64
            starts.clamp_(min=0)
            offsets = torch.arange(8, dtype=torch.int32, device="npu")
            indices = (starts.unsqueeze(1) + offsets).to(torch.int64)
            gathered = torch.gather(table, 1, indices)
            selected = torch.index_select(gathered, 0, rows)
            output[: rows.numel()].copy_(selected, non_blocking=True)
            output[rows.numel() :].fill_(-1)

    def fused():
        for table, output in zip(tables, outputs):
            fused_gdn_state_indices(table, lengths, rows, output, 64, -1)

    results = {}
    for name, fn in (("reference", reference), ("fused", fused)):
        for _ in range(10):
            fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            fn()
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1000)
        results[name] = {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}
    print(json.dumps({"groups": args.groups, "timing": "sync-inclusive wall time", "results": results}, indent=2))


if __name__ == "__main__":
    main()
