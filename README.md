# MammothModa2 attention comparison

Actual checkpoint outputs from the fixed-input runtime comparison for [#7094](https://github.com/vllm-project/vllm-omni/pull/7094): runtime `050b518` versus the same runtime with the full attention change at `4101cf6`. The model is MammothModa2-Preview, BF16, 1024×1024, CFG 4, seed 42, 50 steps.

The overview displays the baseline, shared-auto output and absolute RGB difference multiplied by 8. The detail panel shows the highest-error 128-pixel grid tile, enlarged 2× with nearest-neighbor sampling. Full original images, raw decoded tensors and reproduction scripts accompany the PR evidence release. These panels are visual diagnostics; they do not establish an agreed image-quality threshold.
