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
- torchaudio: `2.11.0+cu130`.
- CUDA device capability: `8.6`.
- Model checkpoint: `ResembleAI/chatterbox` regular English revision
  `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`.

## Experiments

### EXP-000: untouched upstream baseline

- Source commit: `65b1843`.
- Benchmark harness commit: `11f5443`.
- Change: none to model or inference code.
- Runs: two warmups and five measured runs per prompt.
- Result: baseline accepted as the comparison point.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2694.75 +/- 26.22 | 42.21 +/- 1.12 | 2009.22 +/- 20.48 | 656.90 +/- 7.57 | 2.200 | 1.2249 +/- 0.0119 | 56 | 27.87 +/- 0.28 | 3165.4 |
| Medium | 6617.03 +/- 123.35 | 52.09 +/- 18.71 | 5876.48 +/- 113.55 | 689.59 +/- 17.81 | 6.440 | 1.0275 +/- 0.0192 | 162 | 27.58 +/- 0.52 | 3245.8 |
| Long | 24876.00 +/- 1126.91 | 58.82 +/- 2.37 | 23701.25 +/- 1114.99 | 1044.13 +/- 10.39 | 19.640 | 1.2666 +/- 0.0574 | 492 | 20.80 +/- 1.03 | 3500.2 |

Fixed-seed speech-token references:

- Short: `22c2f704d30e1070065f4331ccdc77fca479fa5453511c7785be8604c17ca76c`
- Medium: `63b826922acf7cb3b23cecabf6b1000512b8bb5e3461bb04100a89b33a136f60`
- Long: `b4a4420291d6207626df144d57c8cff2e7caf2efbe37cd49ed17d65813b9f1c6`

Observations:

- The regular model is slower than real time for all three workloads on this
  RTX 3090 under the upstream implementation.
- T3 dominates total latency: 74.6% of short, 88.8% of medium, and 95.3% of
  long end-to-end time.
- Decode throughput falls from about 28 tokens/s to 21 tokens/s on the long
  case. This points to growing dynamic KV-cache cost as the first profiling
  target.
- Medium TTFT contains one 85.49 ms outlier; its median is 44.25 ms. The other
  four samples are between 41.97 and 45.20 ms.

Raw samples, audio, and token references are stored on the benchmark machine
under `benchmark_results/baseline/`.
