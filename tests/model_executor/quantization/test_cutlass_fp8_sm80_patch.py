# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the CUTLASS FP8 support check installed by ``vllm_omni.patch``.

On vLLM 0.30.0 ``CutlassFP8ScaledMMLinearKernel.is_supported`` only checks for
CUDA, so on SM80 it is chosen ahead of Marlin and the first FP8 GEMM fails with
``cutlass_scaled_mm_sm80_epilogue``. The patch adds the ``cutlass_fp8_supported()``
check from vLLM #55884. The device is simulated, so the result does not depend on
the GPU (or its absence) of the machine running the tests.
"""

import inspect

import pytest
import torch
from vllm.model_executor.kernels.linear import (
    CutlassFP8ScaledMMLinearKernel,
    MarlinFP8ScaledMMLinearKernel,
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils import w8a8_utils
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum, current_platform

import vllm_omni.patch as omni_patch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _simulate_cuda(monkeypatch, capability: int) -> None:
    """Make every FP8 kernel's support check see a CUDA device of ``capability``."""
    import vllm.model_executor.kernels.linear.scaled_mm.cutlass as cutlass_mod
    import vllm.model_executor.kernels.linear.scaled_mm.flashinfer as flashinfer_mod
    from vllm.platforms.interface import DeviceCapability

    cutlass_fp8 = capability >= 89
    monkeypatch.delenv("VLLM_DISABLED_KERNELS", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_platform, "supports_fp8", lambda: cutlass_fp8)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(capability // 10, capability % 10),
    )
    monkeypatch.setattr(current_platform, "has_device_capability", lambda c, device_id=0: capability >= c)
    monkeypatch.setattr(current_platform, "is_device_capability_family", lambda c, device_id=0: False)
    monkeypatch.setattr(flashinfer_mod, "has_flashinfer", lambda: False, raising=False)
    monkeypatch.setattr(w8a8_utils, "cutlass_fp8_supported", lambda: cutlass_fp8)
    # Once upstream's own check is installed it resolves the name in the cutlass module.
    monkeypatch.setattr(cutlass_mod, "cutlass_fp8_supported", lambda: cutlass_fp8, raising=False)


def test_cutlass_fp8_rejects_devices_without_cutlass_fp8(monkeypatch):
    _simulate_cuda(monkeypatch, capability=80)
    supported, reason = CutlassFP8ScaledMMLinearKernel.is_supported(80)
    assert not supported
    assert reason


def test_cutlass_fp8_kept_on_devices_with_cutlass_fp8(monkeypatch):
    _simulate_cuda(monkeypatch, capability=90)
    assert CutlassFP8ScaledMMLinearKernel.is_supported(90) == (True, None)


@pytest.mark.parametrize(
    ("capability", "activation_key", "expected"),
    [
        # Fp8LinearMethod picks per-tensor activations when CUTLASS FP8 is unavailable.
        (80, kFp8DynamicTensorSym, MarlinFP8ScaledMMLinearKernel),
        (90, kFp8DynamicTokenSym, CutlassFP8ScaledMMLinearKernel),
    ],
    ids=["sm80_falls_back_to_marlin", "sm90_keeps_cutlass"],
)
def test_fp8_linear_kernel_selection(monkeypatch, capability, activation_key, expected):
    _simulate_cuda(monkeypatch, capability)
    kernel = init_fp8_linear_kernel(
        activation_quant_key=activation_key,
        weight_quant_key=kFp8StaticTensorSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(256, 256),
    )
    assert type(kernel) is expected


def test_forced_cutlass_falls_back_on_sm80(monkeypatch):
    _simulate_cuda(monkeypatch, capability=80)
    kernel = init_fp8_linear_kernel(
        activation_quant_key=kFp8DynamicTensorSym,
        weight_quant_key=kFp8StaticTensorSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(256, 256),
        force_kernel=CutlassFP8ScaledMMLinearKernel,
    )
    assert type(kernel) is MarlinFP8ScaledMMLinearKernel


def test_patch_install_is_idempotent():
    before = inspect.getattr_static(CutlassFP8ScaledMMLinearKernel, "is_supported")
    omni_patch._patch_cutlass_fp8_is_supported()
    assert inspect.getattr_static(CutlassFP8ScaledMMLinearKernel, "is_supported") is before


def test_patch_skips_an_upstream_check(monkeypatch):
    def upstream_is_supported(cls, compute_capability=None):
        if not w8a8_utils.cutlass_fp8_supported():
            return False, "CUTLASS FP8 kernels not available"
        return True, None

    upstream = classmethod(upstream_is_supported)
    monkeypatch.setattr(CutlassFP8ScaledMMLinearKernel, "is_supported", upstream)
    omni_patch._patch_cutlass_fp8_is_supported()
    assert inspect.getattr_static(CutlassFP8ScaledMMLinearKernel, "is_supported") is upstream


def test_patch_raises_when_is_supported_is_not_a_classmethod(monkeypatch):
    monkeypatch.setattr(
        CutlassFP8ScaledMMLinearKernel, "is_supported", staticmethod(lambda compute_capability=None: (True, None))
    )
    with pytest.raises(RuntimeError, match="no longer a classmethod"):
        omni_patch._patch_cutlass_fp8_is_supported()
