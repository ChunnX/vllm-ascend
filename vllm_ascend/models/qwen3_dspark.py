from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

from vllm_ascend.patch.worker.patch_draft_quarot import get_rotataion_matrix, get_rotation_path


# Process the first linear weight with rotation matrix, if the target model uses rotary quantization
def process_weight(linear_weight: torch.Tensor, rotation_weight: torch.Tensor):
    assert linear_weight.shape[1] % rotation_weight.shape[0] == 0, (
        f"Linear weight shape[1] must be a multiple of rotation weight shape[0],"
        f" but get {linear_weight.shape[1]=} and {rotation_weight.shape[0]=}"
    )
    if rotation_weight.dtype != torch.float32:
        rotation_weight = rotation_weight.to(torch.float32)
    hidden_size = rotation_weight.shape[0]
    ori_dtype = linear_weight.dtype
    processed_weight = torch.empty(linear_weight.shape, dtype=torch.float32)
    for start_pos in range(0, linear_weight.shape[1], hidden_size):
        linear_weight_chunked = linear_weight[:, start_pos : start_pos + hidden_size].to(torch.float32)
        processed_weight[:, start_pos : start_pos + hidden_size].copy_(
            torch.matmul(linear_weight_chunked, rotation_weight)
        )
    return processed_weight.to(ori_dtype)


class AscendQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    #: Whether the checkpoint actually carried a ``d2t`` vocabulary mapping.
    #:
    #: A draft declaring ``draft_vocab_size`` allocates ``draft_id_to_target_id``
    #: zero-filled, and the loader skips the parameter when the checkpoint has no
    #: ``d2t``. Its contents therefore cannot tell "nothing was loaded" apart from
    #: the legitimate mapping that keeps target ids ``0..K-1``, since both are all
    #: zeros -- training resolves the same ambiguity with ``t2d``, which serving
    #: drops. The weight stream can tell them apart, and this is the last place
    #: that sees it.
    has_draft_id_mapping: bool = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None

    def _note_draft_id_mapping(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
        """Record whether ``d2t`` is present, passing the stream through."""
        for name, loaded_weight in weights:
            if "d2t" in name:
                self.has_draft_id_mapping = True
            yield name, loaded_weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if self.rotation_path is not None:
            processed_weights: list[tuple[str, torch.Tensor]] = []
            rotation_weight = get_rotataion_matrix(self.rotation_path)
            for name, loaded_weight in weights:
                if "fc." in name:
                    loaded_weight = process_weight(loaded_weight, rotation_weight)
                processed_weights.append((name, loaded_weight))
            weights = processed_weights
        super().load_weights(self._note_draft_id_mapping(weights))
