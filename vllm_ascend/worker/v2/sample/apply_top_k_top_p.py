import torch
import torch_npu  # noqa: F401


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    if k is None and p is None:
        return logits
    if p is not None:
        p = p.to(device=logits.device, dtype=logits.dtype)
    #use cann ops
    return torch_npu.npu_top_k_top_p(logits, p, k)