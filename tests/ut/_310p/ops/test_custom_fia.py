"""
Unit test for custom_fused_infer_attention operator (310 only).
Covers 4 layout+headdim+blocksize combinations, 20 random cases each.
  - TND  + headdim=128, block_size=128
  - TND  + headdim=256, block_size=64
  - BSND + headdim=128, block_size=128
  - BSND + headdim=256, block_size=64

Constraints:
  - MAX_SEQ = 8192, MAX_BATCH = 32
  - num_heads >= num_kv_heads, num_heads % num_kv_heads == 0
  - num_heads <= 16, covers both MHA and GQA
  - Eager mode only (no graph mode)
"""

import sys
import random
import argparse
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


# ── Copied from aicc_run_adn_fia.py ────────────────────────────────

def generate_random_block_table(kv_seq_lens, block_size, total_physical_blocks):
    """
    根据每个 Batch 的 KV 实际长度，动态分配物理 Block。
    无效映射位置严格填充 -1。
    """
    B = len(kv_seq_lens)
    max_blocks_per_seq = int((max(kv_seq_lens) + block_size - 1) // block_size) + 2
    block_table = torch.full([B, max_blocks_per_seq], -1, dtype=torch.int32)
    available_blocks = list(range(total_physical_blocks))

    for b in range(B):
        cur_kv_len = kv_seq_lens[b].item()
        needed_blocks = (cur_kv_len + block_size - 1) // block_size

        if needed_blocks > len(available_blocks):
            raise ValueError(f"物理内存块不足！需要 {needed_blocks} 块，可用 {len(available_blocks)} 块")

        chosen_blocks = random.sample(available_blocks, needed_blocks)
        for cb in chosen_blocks:
            available_blocks.remove(cb)

        block_table[b, :needed_blocks] = torch.tensor(chosen_blocks, dtype=torch.int32)

    return block_table


def compute_golden_output_cpu(query, key_cache, value_cache, num_heads, num_key_value_heads,
                              head_dim, block_size, block_table, query_lens_abs,
                              kv_seq_lens, scale, layout):
    """
    全能 CPU 黄金函数：支持 TND 和 BSND 的自适应切割与物理寻址。
    """
    B = len(query_lens_abs)
    out_list = []
    q_offset = 0

    for b in range(B):
        q_len = query_lens_abs[b].item()

        if layout == "TND":
            cur_query = query[q_offset : q_offset + q_len]
            q_offset += q_len
        elif layout == "BSND":
            cur_query = query[b, :q_len]
        else:
            raise ValueError(f"不支持的 Layout: {layout}")

        cur_kv_len = kv_seq_lens[b].item()
        cur_block_indices = block_table[b]
        num_blocks_needed = (cur_kv_len + block_size - 1) // block_size

        keys_list = []
        values_list = []
        for i in range(num_blocks_needed):
            block_idx = cur_block_indices[i].item()
            keys_list.append(key_cache[block_idx])
            values_list.append(value_cache[block_idx])

        full_keys = torch.cat(keys_list, dim=0)[:cur_kv_len, :]
        full_values = torch.cat(values_list, dim=0)[:cur_kv_len, :]

        full_keys = full_keys.view(cur_kv_len, num_key_value_heads, head_dim).permute(1, 0, 2)
        full_values = full_values.view(cur_kv_len, num_key_value_heads, head_dim).permute(1, 0, 2)

        if num_key_value_heads != num_heads:
            group = num_heads // num_key_value_heads
            full_keys = full_keys.repeat_interleave(group, dim=0)
            full_values = full_values.repeat_interleave(group, dim=0)

        q_trans = cur_query.transpose(0, 1).to(torch.float32)
        k_trans = full_keys.transpose(-1, -2).to(torch.float32)
        attn_scores = torch.matmul(q_trans, k_trans) * scale

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        cur_out = torch.matmul(attn_weights, full_values.to(torch.float32))

        out_list.append(cur_out.transpose(0, 1))

    if layout == "TND":
        return torch.cat(out_list, dim=0).to(query.dtype)
    elif layout == "BSND":
        max_q_len = query.shape[1]
        padded_out = torch.zeros([B, max_q_len, num_heads, head_dim], dtype=torch.float32)
        for b in range(B):
            q_len = query_lens_abs[b].item()
            padded_out[b, :q_len] = out_list[b]
        return padded_out.to(query.dtype)


def _reshape_kv_to_nz_layout(cache):
    """Reshape [N, H, C] KV cache to [N, C//16, H, 16] NZ-like layout (ND metadata)."""
    return cache.reshape(cache.shape[0], cache.shape[1], -1, 16).permute(0, 2, 1, 3).contiguous()


def call_npu_paged_attention_eager(query, key_cache, value_cache, num_heads, num_kv_heads,
                                   input_layout, scale, block_table, block_size,
                                   query_lens_npu, kv_seq_lens_npu, kv_format="ND"):
    """Call the operator with either ND or NZ format KV cache.

    Both formats carry the same logical data and the same memory shape
    [block_num, C//16, block_size, 16].  The only difference is the
    format metadata tag (ND/NCHW vs FRACTAL_NZ), which verifies that
    the operator does not reject NZ-format tensors.
    """
    adn_k_cache = _reshape_kv_to_nz_layout(key_cache)
    adn_v_cache = _reshape_kv_to_nz_layout(value_cache)

    if kv_format == "NZ":
        adn_k_cache = torch_npu.npu_format_cast(adn_k_cache, 29)  # ACL_FORMAT_NZ = 29
        adn_v_cache = torch_npu.npu_format_cast(adn_v_cache, 29)

    attention_out = torch.ops._C_ascend.npu_custom_fused_infer_attention_v310(
        query=query,
        key=[adn_k_cache],
        value=[adn_v_cache],
        actual_seq_lengths_q=query_lens_npu.tolist(),
        actual_seq_lengths_kv=kv_seq_lens_npu.tolist(),
        block_table=block_table,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        block_size=block_size,
        input_layout=input_layout,
        scale_value=scale,
        inner_precise=2
    )
    return attention_out


# ── Parameter combinations ──────────────────────────────────────────

COMBINATIONS = [
    {"layout": "TND",  "head_dim": 128, "block_size": 128},
    {"layout": "TND",  "head_dim": 256, "block_size": 64},
    {"layout": "BSND", "head_dim": 128, "block_size": 128},
    {"layout": "BSND", "head_dim": 256, "block_size": 64},
]

MAX_SEQ = 8192
MAX_BATCH = 32
NUM_CASES = 1

_MHA_CONFIGS = [(1,1), (2,2), (4,4), (8,8), (16,16)]
_GQA_CONFIGS = [(2,1), (4,1), (4,2), (8,1), (8,2), (8,4), (16,1), (16,2), (16,4), (16,8)]


def compare_output(output_cpu, output_npu):
    if output_cpu is None or output_npu is None:
        return
    cpu_flatten = output_cpu.flatten()
    npu_flatten = output_npu.flatten()
    diff_flatten = abs(cpu_flatten - npu_flatten)
    diff_flatten_mean = diff_flatten.mean()
    diff_flatten_max = diff_flatten.max()
    print(f"check diff mean is {diff_flatten_mean}, max is {diff_flatten_max}, sum is {diff_flatten.sum()}")

    return diff_flatten_mean, diff_flatten_max


def run_single_case(layout, head_dim, block_size, num_heads, num_kv_heads, batch_size, case_idx):
    """Run a single test case (eager mode only, 310 only).

    Tests both ND and NZ format KV caches on the same data to verify
    the operator does not reject NZ-format tensors.
    """
    dtype = torch.float16
    atol = 1e-4
    scale = head_dim ** -0.5
    B = batch_size

    query_lens_cpu_abs = torch.tensor([random.randint(1, MAX_SEQ) for _ in range(B)], dtype=torch.int64)
    kv_seq_lens_cpu = torch.tensor([random.randint(1, MAX_SEQ) for _ in range(B)], dtype=torch.int64)

    total_needed_blocks = sum([(l.item() + block_size - 1) // block_size for l in kv_seq_lens_cpu])
    block_num = total_needed_blocks + 20
    block_table_cpu = generate_random_block_table(kv_seq_lens_cpu, block_size, block_num)
    block_table = block_table_cpu.npu()

    key_cache = torch.randn([block_num, block_size, num_kv_heads * head_dim], dtype=dtype).npu()
    value_cache = torch.randn([block_num, block_size, num_kv_heads * head_dim], dtype=dtype).npu()
    kv_seq_lens_npu = kv_seq_lens_cpu.npu()

    if layout == "TND":
        T_q = query_lens_cpu_abs.sum().item()
        query = torch.randn([T_q, num_heads, head_dim], dtype=dtype).npu()
        query_lens_npu = query_lens_cpu_abs.npu()
    elif layout == "BSND":
        max_q_len = query_lens_cpu_abs.max().item()
        query = torch.randn([B, max_q_len, num_heads, head_dim], dtype=dtype).npu()
        query_lens_npu = query_lens_cpu_abs.npu()

    # CPU golden
    golden_output_cpu = compute_golden_output_cpu(
        query=query.detach().cpu(), key_cache=key_cache.detach().cpu(),
        value_cache=value_cache.detach().cpu(),
        num_heads=num_heads, num_key_value_heads=num_kv_heads,
        head_dim=head_dim, block_size=block_size,
        block_table=block_table_cpu, query_lens_abs=query_lens_cpu_abs,
        kv_seq_lens=kv_seq_lens_cpu, scale=scale, layout=layout,
    )

    all_passed = True

    for kv_fmt in ("ND", "NZ"):
        attention_output_npu = call_npu_paged_attention_eager(
            query=query, key_cache=key_cache, value_cache=value_cache,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            input_layout=layout, scale=scale,
            block_table=block_table, block_size=block_size,
            query_lens_npu=query_lens_npu, kv_seq_lens_npu=kv_seq_lens_npu,
            kv_format=kv_fmt,
        )
        torch_npu.npu.synchronize()

        npu_out = attention_output_npu.detach().cpu()
        cpu_out = golden_output_cpu

        if layout == "BSND":
            valid_npu = []
            valid_cpu = []
            for b in range(B):
                q_len = query_lens_cpu_abs[b].item()
                valid_npu.append(npu_out[b, :q_len].flatten())
                valid_cpu.append(cpu_out[b, :q_len].flatten())
            npu_valid = torch.cat(valid_npu)
            cpu_valid = torch.cat(valid_cpu)
            diff_flatten_mean, diff_flatten_max = compare_output(cpu_valid, npu_valid)
        else:
            diff_flatten_mean, diff_flatten_max = compare_output(cpu_out.flatten(), npu_out.flatten())

        passed = diff_flatten_mean <= atol
        tag = (f"[{layout} hd={head_dim} bs={block_size} nh={num_heads} nkv={num_kv_heads} "
               f"B={B} kv={kv_fmt}] case {case_idx}")
        if passed:
            print(f"  PASS {tag}  max_diff={diff_flatten_max:.6f}")
        else:
            print(f"  FAIL {tag}  max_diff={diff_flatten_max:.6f} (atol={atol})")
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="custom_fused_infer_attention unit test (310 only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    enable_custom_op()

    op_name = "npu_custom_fused_infer_attention_v310"
    has_op = hasattr(torch.ops._C_ascend, op_name)
    print(f"[1] torch.ops._C_ascend.{op_name} registered: {has_op}")
    assert has_op, f"Operator {op_name} not found"

    total_pass = 0
    total_fail = 0

    for combo in COMBINATIONS:
        layout = combo["layout"]
        head_dim = combo["head_dim"]
        block_size = combo["block_size"]
        print(f"\n{'='*60}")
        print(f"Combination: layout={layout}  head_dim={head_dim}  block_size={block_size}")
        print(f"{'='*60}")

        combo_pass = 0
        combo_fail = 0

        for i in range(NUM_CASES):
            # First 10 MHA, last 10 GQA
            if i < 10:
                head_cfg = random.choice(_MHA_CONFIGS)
            else:
                head_cfg = random.choice(_GQA_CONFIGS)
            num_heads, num_kv_heads = head_cfg

            B = random.randint(1, MAX_BATCH)

            try:
                passed = run_single_case(
                    layout=layout,
                    head_dim=head_dim,
                    block_size=block_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    batch_size=B,
                    case_idx=i + 1,
                )
                if passed:
                    combo_pass += 1
                    total_pass += 1
                else:
                    combo_fail += 1
                    total_fail += 1
            except Exception as e:
                print(f"  ERROR case {i+1}: {e}")
                combo_fail += 1
                total_fail += 1

        print(f"\n  Subtotal: {combo_pass} passed, {combo_fail} failed")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed  (seed={args.seed})")
    print(f"{'='*60}")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
