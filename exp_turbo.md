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

- Vast.ai instance: `44661021` on machine `19748`.
- GPU: NVIDIA GeForce RTX 3090 24 GB, driver `560.35.03`, 370 W limit.
- Host: Intel Core i5-10500; the rented GPU is exposed over PCIe 3.0 x1.
- Python: `3.12.13`.
- PyTorch: `2.12.0+cu126`.
- torchaudio: `2.11.0+cu126`.
- CUDA runtime: `12.6`; cuDNN `9.10.2`.
- Device capability: `8.6`.
- Upstream source: `resemble-ai/chatterbox` commit `65b1843`.
- Turbo checkpoint: `ResembleAI/chatterbox-turbo` revision
  `749d1c1a46eb10492095d68fbcf55691ccf137cd`.

## Experiments

### EXP-T000: untouched Turbo baseline

- Benchmark harness commit: `c5faec7`.
- Change: none to `inference_turbo()`, Turbo checkpoint loading, S3Gen,
  sampling, conditioning, or watermarking. The shared regular-model T3 path has
  prior optimizations, but the Turbo method remains byte-for-byte upstream.
- Runs: two warmups and five measured runs per prompt.
- Result: accepted as the exact-output comparison point.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Watermark ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 749.64 +/- 8.18 | 22.11 +/- 0.13 | 638.59 +/- 9.31 | 93.99 +/- 1.90 | 16.07 +/- 1.09 | 2.720 | 0.2756 +/- 0.0030 | 65 | 101.80 +/- 1.46 | 2824.9 |
| Medium | 1887.27 +/- 17.66 | 23.18 +/- 2.22 | 1730.28 +/- 14.40 | 111.76 +/- 0.73 | 43.55 +/- 3.40 | 7.360 | 0.2564 +/- 0.0024 | 181 | 104.61 +/- 0.87 | 2900.5 |
| Long | 6384.64 +/- 31.01 | 22.87 +/- 0.03 | 5947.03 +/- 27.65 | 268.56 +/- 0.68 | 165.25 +/- 21.82 | 25.080 | 0.2546 +/- 0.0012 | 624 | 104.93 +/- 0.49 | 3419.0 |

Fixed-seed speech-token references:

- Short: `694ab5f7af59523030379696e24a182126d2ff1f4be092269a8fc4cdfe42ec62`
- Medium: `f299ea2212ea1e024950116a93b6d4bb7c6a5abe8fa77ed0f9365d0a957f3df0`
- Long: `e22a7ab72ba83a76c57a376deb555f9d5d59155705124d80caca95f4548ec265`

Exact float-waveform references after the unchanged watermark:

- Short: `1ecced6f9c212839f9172d45f69ab6f18ec2256861b8ca3a1d8c68ed704e1034`
- Medium: `03989119848d716d76da23852dd223b2f7bb7bf9ac8588537e4507688e3e7287`
- Long: `739b09408460a41e481aa72bee70b776afb424117a4c89a21ebce7d1289a26b7`

All five timed runs per case produced identical token and float-waveform
hashes. T3 accounts for 85.2% of short latency, 91.7% of medium latency, and
93.1% of long latency. Decode throughput is nearly flat at 102-105 tokens/s,
so the first optimization target is per-token T3 dispatch and transformer
work, not S3Gen. The raw artifacts remain on the GPU under
`benchmark_results/turbo_exp_t000/`.

The previous instance `44639182` developed a CUDA UVM `EIO` before a Turbo
baseline could run. That infrastructure failure is documented here but is not
counted as an experiment or model result.

### EXP-T001: T3 kernel and KV-cache profile

- Profiler commit: `85f1fef`.
- Workload: two warmups followed by a PyTorch CPU/CUDA trace of the short T3
  path only; S3Gen, vocoder, and watermark were not profiled.
- Quality check: 65 generated tokens with hash
  `694ab5f7af59523030379696e24a182126d2ff1f4be092269a8fc4cdfe42ec62`,
  exactly matching EXP-T000.
- Result: diagnostic accepted; model behavior did not change.

Architecture and cache findings:

- All 298 T3 parameter tensors are FP32 and occupy 1630.33 MiB.
- Transformers selects its `sdpa` implementation. The actual decode kernel is
  PyTorch's FP32 CUTLASS memory-efficient attention
  (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`), not Flash Attention.
- All SDP backend switches are enabled, but native Flash Attention is not
  eligible for this FP32 path.
- The cache is a 24-layer FP32 `DynamicCache`. First-layer K and V are each
  contiguous `[1, 16, sequence, 64]` tensors.
- Prefill creates a cache length of 386. The final short-run cache length is
  451 after 66 transformer calls.

Main traced operators:

| Operator group | Calls | CUDA total ms | Self CUDA ms | CPU total ms | Cumulative CUDA allocation |
|---|---:|---:|---:|---:|---:|
| Projection/MLP (`aten::addmm`) | 6,403 | 123.35 | 123.28 | 131.65 | 388.18 MiB |
| Efficient attention forward | 1,584 | 119.38 | 119.38 | 39.03 | 43.08 MiB |
| Dynamic-cache and loop concatenation (`aten::cat`) | 3,237 | 15.45 | 15.45 | 47.51 | 5.15 GiB |
| Top-p processing | 66 | 3.79 | diagnostic annotation | 18.86 | 9.18 MiB released net |
| Top-k processing | 66 | 3.68 | diagnostic annotation | 8.84 | 1.19 MiB released net |
| Speech logits head | 66 | 2.27 | 2.27 | 3.90 | included in linear ops |
| Multinomial sampling | 66 | 1.98 | child kernels | 13.60 | 1.97 MiB released net |
| Repetition penalty | 66 | 0.71 | diagnostic annotation | 10.65 | 0.16 MiB released net |
| Speech embedding | 66 | 0.15 | 0.15 | 3.42 | negligible |

The trace also contains 130 `aten::all` calls, 197 scalar reads, and 131
device-to-host scalar copies. These come from the per-token all-invalid-logits
and EOS checks. Profiler wall time is intentionally excluded from latency
comparisons because instrumentation reduces the visible loop rate from about
105 to 63 tokens/s.

The dominant work is split almost evenly between FP32 projection/MLP matmuls
and FP32 attention. First test TF32 matmuls, which can accelerate projections
without changing tensor storage or cache precision. Then isolate cache/loop
allocation and compilation; BF16/Flash paths are not eligible for retention
unless they pass the exact token and waveform gate.
