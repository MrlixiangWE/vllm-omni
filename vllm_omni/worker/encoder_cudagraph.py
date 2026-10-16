# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Encoder CUDA graph manager that replays a group only when one graph holds it."""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

logger = init_logger(__name__)


class SingleReplayEncoderCudaGraphManager(EncoderCudaGraphManager):
    """Replay an encoder group only when one captured graph holds all of it.

    The upstream manager splits a group across several replays and gives each
    item above the largest budget its own eager call. For a launch-bound vision
    tower that is usually slower than one eager call over the whole group, so a
    group that does not fit one graph goes back to the runner, which then
    encodes it with a single ``embed_multimodal`` call. Models opt in by
    setting ``encoder_cudagraph_single_replay``.
    """

    def __init__(self, vllm_config, device: torch.device, dtype: torch.dtype, model):
        super().__init__(vllm_config, device, dtype, model)
        # A model may cap one path's budgets below the shared maximum, which
        # explicit encoder_cudagraph_token_budgets otherwise raise for every path.
        path_max_budget = getattr(model, "get_encoder_cudagraph_path_max_budget", None)
        if path_max_budget is not None:
            for path, budgets in self.path_token_budgets.items():
                cap = path_max_budget(path)
                if cap is None:
                    continue
                capped = [budget for budget in budgets if budget <= cap]
                if not any(capped):
                    raise ValueError(f"Encoder CUDA graph path {path!r} has no budget at or below its cap {cap}")
                self.path_token_budgets[path] = capped
            logger.info("Encoder CUDA graph budgets after per-path caps: %s", self.path_token_budgets)

    def execute(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor] | None:  # type: ignore[override]
        # Models may keep a group eager when a graph would not reproduce it.
        group_eligible = getattr(self.model, "encoder_cudagraph_group_eligible", None)
        if group_eligible is not None and not group_eligible(mm_kwargs):
            self.graph_misses += len(self._get_item_specs(mm_kwargs))
            return None
        if not self.use_dp:
            specs = self._get_item_specs(mm_kwargs)
            fits = len(specs) <= self.max_batch_size and all(
                sum(spec.get_path_output_tokens(path) for spec in specs) <= max(budgets)
                for path, budgets in self.path_token_budgets.items()
            )
            if not fits:
                self.graph_misses += len(specs)
                return None
        return super().execute(mm_kwargs)
