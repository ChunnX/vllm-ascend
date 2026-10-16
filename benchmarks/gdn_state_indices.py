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

from vllm_ascend.ops.triton.mamba.gdn_state_indices import fused_gdn_spec_metadata, fused_gdn_state_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--metadata", action="store_true", help="Compare state-only fusion plus padding with full fusion"
    )
    parser.add_argument("--graph-rows", type=int, default=1)
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

    if args.metadata:
        graph_rows = args.graph_rows
        assert graph_rows >= 1
        query_input = torch.tensor([0, 8], dtype=torch.int32, device="npu")
        accepted_input = torch.tensor([5], dtype=torch.int32, device="npu")
        mask_input = torch.tensor([True], dtype=torch.bool, device="npu")
        token_input = torch.arange(8, dtype=torch.int32, device="npu")
        buffers = [
            (
                torch.empty((graph_rows, 8), dtype=torch.int32, device="npu"),
                torch.empty(graph_rows + 1, dtype=torch.int32, device="npu"),
                torch.empty(graph_rows, dtype=torch.int32, device="npu"),
                torch.empty(graph_rows, dtype=torch.bool, device="npu"),
                torch.empty(graph_rows + 1, dtype=torch.int32, device="npu"),
                torch.empty(8, dtype=torch.int32, device="npu"),
            )
            for _ in tables
        ]

        def reference():
            for table, output in zip(tables, buffers):
                states, query, accepted, masks, actual_lengths, tokens = output
                fused_gdn_state_indices(table, lengths, rows, states, 64, -1)
                masks[:1].copy_(mask_input, non_blocking=True)
                masks[1:].fill_(False)
                tokens.copy_(token_input, non_blocking=True)
                query[:2].copy_(query_input, non_blocking=True)
                if graph_rows > 1:
                    query[2:].copy_(query[1:2].expand(graph_rows - 1), non_blocking=True)
                accepted[:1].copy_(accepted_input, non_blocking=True)
                accepted[1:].fill_(1)
                actual_lengths[:1].copy_(query[:1])
                torch.sub(query[1:], query[:-1], out=actual_lengths[1:])

        def fused():
            for table, output in zip(tables, buffers):
                fused_gdn_spec_metadata(table, lengths, query_input, accepted_input, output, 1, 8, 64, -1)

        reference()
        expected = [[x.clone() for x in output] for output in buffers]
        fused()
        for output, reference_output in zip(buffers, expected):
            for value, expected_value in zip(output, reference_output):
                torch.testing.assert_close(value.cpu(), expected_value.cpu(), rtol=0, atol=0)

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
    print(
        json.dumps(
            {
                "groups": args.groups,
                "metadata": args.metadata,
                "graph_rows": args.graph_rows,
                "timing": "sync-inclusive wall time",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
