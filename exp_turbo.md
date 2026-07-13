# Chatterbox Turbo inference experiments

This ledger records every performance experiment on
`ResembleAI/chatterbox-turbo`. Regular English Chatterbox results remain in
[`exp.md`](./exp.md).

## Benchmark contract

- Hardware target: a dedicated NVIDIA RTX 3090 24 GB.
- Workloads: fixed short, medium, and long prompts from
  `benchmarks/benchmark_turbo.py`.
- Sampling: upstream Turbo defaults (`temperature=0.8`, `top_k=1000`,
  `top_p=0.95`, `repetition_penalty=1.2`), checkpoint built-in voice,
  S3Gen mean-flow decode unchanged, and watermark enabled.
- Timing: two warmups followed by five measured runs per case, with CUDA
  synchronized at every measurement boundary.
- Metrics: end-to-end latency, T3 time to first sampled token, T3 latency,
  S3Gen latency, watermark latency, audio duration, RTF, generated tokens,
  T3 tokens/s, and peak CUDA memory.
- Quality gate: fixed seeds, deterministic timed runs, exact speech-token
  tensors, and exact float-waveform tensors. An optimization is retained only
  if all three cases exactly match the untouched baseline.
- TTFT definition: time from entry into `model.t3.inference_turbo()` until the
  first sampled speech token materializes. This is not streamed time to first
  audio; Turbo currently returns a complete waveform.

## Environment

The exact GPU, software, and checkpoint revisions will be recorded with the
untouched baseline after a healthy RTX 3090 worker is attached.

## Experiments

No Turbo experiment has been accepted or rejected yet. The current Vast worker
lost its CUDA UVM attachment before the baseline run; no invalid CPU or failed
CUDA measurements are recorded as model results.
