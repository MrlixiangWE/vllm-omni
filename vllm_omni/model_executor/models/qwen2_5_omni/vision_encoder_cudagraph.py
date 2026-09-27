# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen2.5-Omni image encoder contract for the upstream CUDA graph manager."""

import dataclasses
from typing import Any

from vllm.distributed import get_pp_group
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

# The thinker's vision tower is Qwen2.5-VL's, so buffer layout, window
# metadata, capture and replay are that model's protocol methods.
_VL = Qwen2_5_VLForConditionalGeneration


class Qwen2_5OmniVisionEncoderCudaGraphMixin:
    """Image-only protocol methods; video and audio retain embed_multimodal.

    The capability attribute is set on eligible model instances after tower
    construction. It is deliberately absent from this mixin's class: the
    runtime protocol checks attribute presence, not the value of a false flag.
    """

    # Read by Qwen2.5-VL's config; the thinker has no EVS pruning.
    is_multimodal_pruning_enabled = False

    def _enable_image_encoder_cudagraph(self) -> None:
        # Only the first pipeline rank runs the encoder. FP8 ViT attention
        # advances a host-side amax slot per call, which a graph would freeze.
        if (
            self.visual is not None
            and not self.multimodal_config.enable_mm_embeds
            and self.multimodal_config.mm_encoder_attn_dtype is None
            and self.multimodal_config.get_limit_per_prompt("image") > 0
            and get_pp_group().is_first_rank
            and self.visual.attn_backend
            in {
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.ROCM_AITER_FA,
                AttentionBackendEnum.TRITON_ATTN,
            }
        ):
            self.supports_encoder_cudagraph = True

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return dataclasses.replace(_VL.get_encoder_cudagraph_config(self), modalities=["image"])

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "image_grid_thw" not in mm_kwargs:
            raise ValueError("Qwen2.5-Omni encoder graphs only take images")
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def get_encoder_cudagraph_budget_range(self, vllm_config) -> tuple[int, int]:
        # Replay pays off while the eager tower is launch-bound; large images
        # padded to the next power-of-two budget run slower than eager, and
        # every budget takes graph memory from the KV cache. Auto-inferred
        # budgets therefore stop at 256 tokens; larger images run eagerly
        # unless encoder_cudagraph_token_budgets asks for more.
        maximum = min(vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.model_config.max_model_len, 256)
        return min(64, maximum), maximum

    _get_pixel_values_by_modality = _VL._get_pixel_values_by_modality
    _get_grid_thw_by_modality = _VL._get_grid_thw_by_modality
    get_encoder_cudagraph_item_specs = _VL.get_encoder_cudagraph_item_specs
    select_encoder_cudagraph_items = _VL.select_encoder_cudagraph_items
    prepare_encoder_cudagraph_capture_inputs = _VL.prepare_encoder_cudagraph_capture_inputs
    prepare_encoder_cudagraph_replay_buffers = _VL.prepare_encoder_cudagraph_replay_buffers
    encoder_cudagraph_forward = _VL.encoder_cudagraph_forward
    encoder_eager_forward = _VL.encoder_eager_forward
    postprocess_encoder_output = SupportsEncoderCudaGraph.postprocess_encoder_output
