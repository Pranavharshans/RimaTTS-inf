# Chatterbox inference experiments

This ledger records every performance experiment on the regular English
`ResembleAI/chatterbox` model. Turbo results are intentionally out of scope.

## Benchmark contract

- Hardware: Vast.ai instance `44639182`, NVIDIA RTX 3090 24 GB.
- Workloads: fixed short, medium, and long prompts from
  `benchmarks/benchmark_regular.py`.
- Sampling: upstream defaults (`temperature=0.8`, `cfg_weight=0.5`,
  `repetition_penalty=1.2`, `min_p=0.05`, `top_p=1.0`).
- Quality controls: fixed seeds, watermark enabled, saved speech-token tensors,
  and saved WAV output for every case.
- Timing: two warmups followed by five measured runs per case. CUDA is
  synchronized at measurement boundaries.
- Acceptance: an optimization is retained only when it improves repeated
  measurements without changing generated speech tokens for the fixed seeds.

## Environment

- Upstream source: `resemble-ai/chatterbox` commit `65b1843`.
- PyTorch: `2.12.0+cu130`.
- CUDA device capability: `8.6`.
- Model checkpoint: `ResembleAI/chatterbox` regular English checkpoint.

## Experiments

Results will be appended after each benchmark run. Raw samples, audio, and token
references are stored under `benchmark_results/`.
