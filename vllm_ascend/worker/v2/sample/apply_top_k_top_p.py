import torch
import torch_npu  # noqa: F401

from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch


def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    # NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
    # As a result, its workspace is not reserved during warmup, which may lead to an OOM
    # when the op is executed later.
    # TODO: Fix the workspace reservation issue in the future.
    if k is None and p is None:
        return logits
    # use pytorch ops
    return _apply_top_k_top_p_pytorch(logits, k, p)


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    if k is None and p is None:
        return logits
    if p is not None:
        p = p.to(device=logits.device, dtype=logits.dtype)
    # use cann ops
    return torch_npu.npu_top_k_top_p(logits, p, k)
