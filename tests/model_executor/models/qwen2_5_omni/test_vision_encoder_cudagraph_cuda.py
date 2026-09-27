# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Captured Qwen2.5-Omni image encoder graphs replay the eager tower on CUDA."""

import socket
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
]

DTYPE = torch.bfloat16
# Four input patches per output token; budgets are in output tokens.
BUDGETS = [8, 64]


def _tol(reference):
    # bf16 relative tolerance, with the absolute floor scaled to the output.
    return {"rtol": 1.6e-2, "atol": 1.6e-2 * reference.abs().max().item()}


def _vision_config():
    # 32-pixel windows over 8-pixel patches: two merged tokens per window side.
    return SimpleNamespace(
        patch_size=8,
        temporal_patch_size=2,
        in_channels=3,
        depth=2,
        hidden_size=128,
        num_heads=2,
        out_hidden_size=64,
        window_size=32,
        spatial_merge_size=2,
        fullatt_block_indexes=[1],
        intermediate_size=256,
        hidden_act="silu",
    )


def _engine_config():
    mm = SimpleNamespace(
        enable_mm_embeds=False,
        get_limit_per_prompt=lambda modality: 4,
        mm_encoder_tp_mode="weights",
        mm_encoder_attn_dtype=None,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=mm, max_model_len=256),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        compilation_config=SimpleNamespace(
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=BUDGETS,
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


@pytest.fixture(scope="module")
def _parallel_state():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state

    if parallel_state.model_parallel_is_initialized():
        yield
        return
    previous_device = torch.accelerator.current_device_index()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.accelerator.set_device_index(0)
    parallel_state.init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl"
    )
    with set_current_vllm_config(VllmConfig()):
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
        torch.accelerator.set_device_index(previous_device)


@pytest.fixture(scope="module", params=[AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN])
def encoder(request, _parallel_state):
    import vllm.model_executor.models.vision as vision_module
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    override = MultiModalConfig(mm_encoder_attn_backend=request.param)
    with (
        pytest.MonkeyPatch.context() as patch,
        set_current_vllm_config(VllmConfig()),
        set_default_torch_dtype(DTYPE),
        torch.device("cuda"),
    ):
        patch.setattr(vision_module, "get_multimodal_config", lambda: override)
        visual = Qwen2_5_VisionTransformer(vision_config=_vision_config(), prefix="visual")
    assert visual.attn_backend == request.param
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in visual.named_parameters():
            if "norm" in name:
                param.fill_(1.0)
            elif param.dim() > 1:
                fan_in = param[0].numel()
                param.copy_(torch.randn(param.shape, generator=generator) / fan_in**0.5)
            else:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.02)
    model = object.__new__(Qwen2_5OmniThinkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.visual = visual
    model.vllm_config = _engine_config()
    model.multimodal_config = model.vllm_config.model_config.multimodal_config
    model._enable_image_encoder_cudagraph()
    return model


def _images(grids, seed):
    generator = torch.Generator().manual_seed(seed)
    patches = sum(t * h * w for t, h, w in grids)
    pixels = torch.randn(patches, 3 * 2 * 8 * 8, generator=generator)
    return {"image_grid_thw": torch.tensor(grids), "pixel_values": pixels.to("cuda", DTYPE)}


def _forward(model, pixels, metadata):
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context

    # The tower on caller-supplied metadata, inside a forward context like the thinker.
    with torch.inference_mode(), set_forward_context(None, VllmConfig()):
        return model.visual(pixels, None, encoder_metadata=metadata)


def _eager(model, inputs):
    from vllm.config import VllmConfig

    # The thinker's own image path, which the runner calls without graphs.
    engine_config, model.vllm_config = model.vllm_config, VllmConfig()
    try:
        with torch.inference_mode():
            return list(model.embed_multimodal(**inputs))
    finally:
        model.vllm_config = engine_config


def _assert_detectable(model, inputs, wrong_metadata):
    # A replay built from wrong metadata must fall outside the tolerance.
    right = torch.cat(_eager(model, inputs))
    wrong = _forward(model, inputs["pixel_values"], wrong_metadata)
    assert not torch.allclose(wrong, right, **_tol(right))


def test_tolerance_detects_wrong_window_metadata(encoder):
    visual = encoder.visual
    inputs = _images([[1, 8, 12]], seed=1)
    grids = inputs["image_grid_thw"].tolist()
    metadata = visual.prepare_encoder_metadata(grids)
    shifted = dict(metadata, window_index=metadata["window_index"].roll(1))
    _assert_detectable(encoder, inputs, shifted)
    rotated = dict(metadata, rotary_pos_emb_cos=metadata["rotary_pos_emb_cos"].roll(1, dims=0))
    _assert_detectable(encoder, inputs, rotated)
    whole = metadata["cu_window_seqlens"][[0, -1]]
    merged = dict(metadata, cu_window_seqlens=whole, max_seqlen_window=metadata["max_seqlen_full"])
    _assert_detectable(encoder, inputs, merged)


def test_captured_graphs_match_eager_and_keep_previous_outputs(encoder):
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    assert supports_encoder_cudagraph(encoder)
    manager = EncoderCudaGraphManager(encoder.vllm_config, torch.device("cuda"), DTYPE, encoder)
    try:
        # The runner captures inside graph_capture(), which supplies the
        # non-default capture stream.
        with torch.inference_mode(), graph_capture(device=torch.device("cuda")):
            set_cudagraph_capturing_enabled(True)
            try:
                manager.capture(graph_pool=torch.cuda.graph_pool_handle())
            finally:
                set_cudagraph_capturing_enabled(False)
        assert manager.get_cumulative_stats()["num_budgets"] == len(BUDGETS)
        # An image filling the smallest budget; a grid whose windows do not
        # tile it (partial windows, padded replay); one 256-patch image filling
        # the largest budget, longer than one attention tile; two images the
        # manager packs into one replay in reverse order.
        cases = [[[1, 4, 8]], [[1, 6, 10]], [[1, 16, 16]], [[1, 8, 8], [1, 4, 8]]]
        retained = []
        for seed, grids in enumerate(cases):
            inputs = _images(grids, seed)
            expected = _eager(encoder, inputs)
            with torch.inference_mode():
                actual = manager.execute(inputs)
            assert [tuple(x.shape) for x in actual] == [tuple(x.shape) for x in expected]
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output, reference, **_tol(reference))
                retained.append((output, output.clone()))
        stats = manager.get_cumulative_stats()
        # Hits and misses count images, not replays.
        assert stats["graph_hits"] == 5 and stats["graph_misses"] == 0, stats
        for output, saved in retained:
            torch.testing.assert_close(output, saved, rtol=0, atol=0)

        # Larger than every budget: served by the eager tower.
        inputs = _images([[1, 16, 32]], seed=9)
        with torch.inference_mode():
            actual = manager.execute(inputs)
        expected = _eager(encoder, inputs)[0]
        torch.testing.assert_close(actual[0], expected, **_tol(expected))
        assert manager.get_cumulative_stats()["graph_misses"] == 1
    finally:
        manager.clear()
