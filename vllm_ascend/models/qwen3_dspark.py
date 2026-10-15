from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, process_eagle_weight

from vllm_ascend.models.llama_eagle3 import load_quarot_target_layer
from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
)

TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)


# Process the first linear weight with rotation matrix, if the target model uses rotary quantization
def process_weight(linear_weight: torch.Tensor, rotation_weight: torch.Tensor):
    assert linear_weight.shape[1] % rotation_weight.shape[0] == 0, (
        f"Linear weight shape[1] must be a multiple of rotation weight shape[0],"
        f" but get {linear_weight.shape[1]=} and {rotation_weight.shape[0]=}"
    )
    rotation_weight = rotation_weight.to(device=linear_weight.device, dtype=torch.float32)
    hidden_size = rotation_weight.shape[0]
    ori_dtype = linear_weight.dtype
    processed_weight = torch.empty(linear_weight.shape, dtype=torch.float32, device=linear_weight.device)
    for start_pos in range(0, linear_weight.shape[1], hidden_size):
        linear_weight_chunked = linear_weight[:, start_pos : start_pos + hidden_size].to(torch.float32)
        processed_weight[:, start_pos : start_pos + hidden_size].copy_(
            torch.matmul(linear_weight_chunked, rotation_weight)
        )
    return processed_weight.to(ori_dtype)


@torch.no_grad()
def align_draft_weights(model, projection, vllm_config):
    """Align draft inputs with the rotated target without modifying shared weights."""
    rotation_path = get_rotation_path(vllm_config)
    if rotation_path is not None:
        rotation = get_rotation_matrix(rotation_path).cpu()
        weight = projection.weight
        weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))
    else:
        if get_pp_group().world_size == 1:
            return
        # PP stages past the first cannot alias the target's stage-0
        # embedding; load this rank's vocab shard from the target
        # checkpoint unrotated so the draft owns its copy.
        rotation = None
        weight = projection.weight
    target_config = vllm_config.model_config.hf_text_config
    for owner, name, layer_cls, weight_names, own_flag in (
        (model.model, "embed_tokens", VocabParallelEmbedding, TARGET_EMBED_WEIGHT_NAMES, "has_own_embed_tokens"),
        (model, "lm_head", ParallelLMHead, TARGET_LM_HEAD_WEIGHT_NAMES, "has_own_lm_head"),
    ):
        if getattr(model, own_flag, False):
            continue
        if rotation is None and name != "embed_tokens":
            # The target lm_head lives on the last PP stage and can still
            # be aliased; only the embedding needs its own copy.
            continue
        with torch.device(weight.device):
            layer = layer_cls(target_config.vocab_size, target_config.hidden_size, params_dtype=weight.dtype)
        load_quarot_target_layer(layer, vllm_config.model_config.model, weight_names, rotation, f"draft {name}.weight")
        layer.quant_method.process_weights_after_loading(layer)
        setattr(owner, name, layer)
        setattr(model, own_flag, True)


class AscendQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config
        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        if not self.enable_confidence_head:
            raise RuntimeError("The DSpark confidence head is disabled.")
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def configure_target_aux_hidden_capture(self, target_model: torch.nn.Module) -> None:
        """Select draft auxiliary inputs, without changing target Eager/Graph mode."""
        set_capture_mode = getattr(target_model, "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is None:
            get_language_model = getattr(target_model, "get_language_model", None)
            if callable(get_language_model):
                set_capture_mode = getattr(get_language_model(), "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is not None:
            set_capture_mode(True)

    has_draft_id_mapping: bool = False
    has_own_lm_head_weights: bool = False

    def _note_checkpoint_contents(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Record optional checkpoint keys while preserving one-shot streams."""
        for name, loaded_weight in weights:
            if "d2t" in name:
                self.has_draft_id_mapping = True
            if "lm_head" in name:
                self.has_own_lm_head_weights = True
            yield name, loaded_weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights = list(self._note_checkpoint_contents(weights))
        if self.draft_id_to_target_id is not None and self.has_draft_id_mapping and not self.has_own_lm_head_weights:
            result = self._load_reduced_vocab_without_head(weights)
        else:
            result = super().load_weights(iter(weights))
        if self.draft_id_to_target_id is not None:
            # Keep the draft-sized head through sharing and QuaRot post-processing.
            # It is filled from the mapped target rows after target-space alignment.
            self.has_own_lm_head = True
        return result

    def _load_reduced_vocab_without_head(self, weights):
        """Load a mapped draft whose head will be derived after target loading.

        vLLM 0.30 rejects reduced checkpoints without a head. Keep its loading
        rules here, allowing only that head to be filled later by the speculator.
        """
        model_weights = {}
        includes_embed_tokens = False
        includes_confidence_head = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
            else:
                name = "model." + name
            includes_embed_tokens |= "embed_tokens" in name
            includes_confidence_head |= "confidence_head" in name
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)
        if self.config.vocab_size > self.target_vocab_size and not includes_embed_tokens:
            raise ValueError(
                "Qwen3 DSpark checkpoints with expanded input vocabularies must include embed_tokens weights."
            )
        skipped = {"mask_embedding": None, "lm_head": None}
        if not includes_embed_tokens:
            skipped["embed_tokens"] = None
        if self.model.confidence_head is None or not includes_confidence_head:
            self.model.confidence_head = None
            skipped["confidence_head"] = None
        AutoWeightsLoader(self).load_weights(model_weights.items(), mapper=WeightsMapper(orig_to_new_substr=skipped))
        self.model._build_fused_kv_buffers()

    def post_process(self, vllm_config: VllmConfig) -> None:
        align_draft_weights(self, self.model.fc, vllm_config)
