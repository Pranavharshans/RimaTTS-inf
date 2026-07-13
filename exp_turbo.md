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

### EXP-T002: T3-only TF32 matmul precision

- Benchmark-mode commit: `d00eaab`.
- Change: set FP32 matmul precision to `high` only while T3 runs, then restore
  `highest` before S3Gen. Weights, activations, KV cache, sampling, S3Gen, and
  watermark remain FP32 and unchanged.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected for performance; exact-output quality gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 758.24 +/- 14.03 | 19.64 +/- 0.09 | 646.48 +/- 10.80 | 95.01 +/- 2.24 | 2.720 | 0.2788 +/- 0.0052 | 65 | 100.57 +/- 1.69 | 2824.9 |
| Medium | 1919.40 +/- 9.52 | 19.91 +/- 0.12 | 1765.28 +/- 5.54 | 111.92 +/- 1.34 | 7.360 | 0.2608 +/- 0.0013 | 181 | 102.53 +/- 0.32 | 2900.5 |
| Long | 6468.48 +/- 93.58 | 30.95 +/- 14.43 | 6025.69 +/- 89.03 | 268.00 +/- 0.54 | 25.080 | 0.2579 +/- 0.0037 | 624 | 103.57 +/- 1.56 | 3419.0 |

Every speech-token tensor and final float waveform exactly matches EXP-T000;
maximum absolute audio difference is `0.0` for all cases. TF32 improves the
stable short and medium TTFT values, but complete T3 time regresses from
638.59/1730.28/5947.03 ms to 646.48/1765.28/6025.69 ms. The batch-one decode
is dominated by narrow GEMV-like operations that do not receive the regular
model's TF32 benefit, so this mode is not retained.

### EXP-T003: fixed generated-token buffer and hidden progress

- Implementation commit: `7040c31`.
- Change: opt-in fixed-size storage for sampled token IDs instead of rebuilding
  them with a list-wide `torch.cat` each step; disable T3's `tqdm` rendering.
  Transformer execution, KV cache, sampling processors, S3Gen, and watermark
  are unchanged. Upstream behavior remains the public default.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected for performance; exact-output quality gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 760.45 +/- 5.90 | 21.93 +/- 0.08 | 650.51 +/- 4.64 | 95.12 +/- 1.13 | 2.720 | 0.2796 +/- 0.0022 | 65 | 99.93 +/- 0.71 | 2824.9 |
| Medium | 1942.51 +/- 39.70 | 22.12 +/- 0.13 | 1781.33 +/- 18.43 | 123.23 +/- 21.96 | 7.360 | 0.2639 +/- 0.0054 | 181 | 101.62 +/- 1.04 | 2900.5 |
| Long | 6398.62 +/- 161.44 | 22.76 +/- 0.12 | 5950.07 +/- 165.07 | 268.77 +/- 0.52 | 25.080 | 0.2551 +/- 0.0064 | 624 | 104.94 +/- 2.81 | 3419.1 |

Every token and float-waveform tensor exactly matches EXP-T000 with maximum
absolute audio difference `0.0`. The optimization removes only about 65 of the
3,237 traced concatenations; 3,120 come from K/V growth inside the 24
transformer layers. Short and medium regress, and long is statistically flat.
The opt-in implementation is retained only for reproducibility and is not part
of the recommended path.

### EXP-T004: remove redundant decode synchronizations

- Implementation commit: `633b52f`.
- Change: opt-in path skips the per-token all-`-inf` guard for valid model
  logits and uses a single scalar EOS read instead of an equality kernel plus
  reduction. T3 weights, FP32 DynamicCache, transformer execution, sampling
  processors, S3Gen, and watermark are unchanged. Progress rendering is hidden.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected for performance; exact-output quality gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 758.69 +/- 5.98 | 22.17 +/- 0.05 | 646.06 +/- 6.08 | 95.98 +/- 0.50 | 2.720 | 0.2789 +/- 0.0022 | 65 | 100.62 +/- 0.95 | 2824.9 |
| Medium | 1931.97 +/- 26.62 | 24.00 +/- 4.16 | 1777.32 +/- 28.05 | 113.57 +/- 1.50 | 7.360 | 0.2625 +/- 0.0036 | 181 | 101.86 +/- 1.59 | 2900.5 |
| Long | 6498.42 +/- 17.51 | 27.25 +/- 10.08 | 6068.44 +/- 34.52 | 267.87 +/- 0.59 | 25.080 | 0.2591 +/- 0.0007 | 624 | 102.83 +/- 0.59 | 3419.0 |

Every token and float-waveform tensor exactly matches EXP-T000 with maximum
absolute audio difference `0.0`. T3 latency is 646.06/1777.32/6068.44 ms versus
the untouched baseline's 638.59/1730.28/5947.03 ms. The remaining EOS scalar
read still synchronizes each iteration; removing the earlier guard changes when
the host waits but does not shorten the dependent GPU work. This mode is not
part of the recommended path.

### EXP-T005: FP32 preallocated dynamic-length KV cache

- Implementation commit: `b210046`.
- Change: replace per-token DynamicCache concatenation with fixed-capacity FP32
  K/V allocations, in-place writes, and populated-prefix views. Attention still
  uses the Transformers SDPA path and receives no padded future positions.
  Sampling, S3Gen, and watermark are unchanged. Progress rendering is hidden.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected for performance; exact-output quality gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 770.98 +/- 4.91 | 22.00 +/- 0.04 | 661.68 +/- 4.62 | 94.91 +/- 0.45 | 2.720 | 0.2834 +/- 0.0018 | 65 | 98.24 +/- 0.68 | 3012.3 |
| Medium | 1958.29 +/- 6.70 | 21.95 +/- 0.09 | 1808.35 +/- 7.21 | 112.15 +/- 1.31 | 7.360 | 0.2661 +/- 0.0009 | 181 | 100.09 +/- 0.40 | 3016.4 |
| Long | 6700.90 +/- 44.89 | 22.58 +/- 0.08 | 6243.53 +/- 35.98 | 267.84 +/- 0.80 | 25.080 | 0.2672 +/- 0.0018 | 624 | 99.95 +/- 0.58 | 3419.0 |

Every token and float-waveform tensor exactly matches EXP-T000 with maximum
absolute audio difference `0.0`. T3 latency is 661.68/1808.35/6243.53 ms versus
the untouched baseline's 638.59/1730.28/5947.03 ms. A populated prefix of a
max-length `[B,H,T,D]` allocation keeps the maximum-length stride between heads;
the resulting non-contiguous attention input costs more than the avoided K/V
concatenations save. Preallocation also raises short-case peak allocation from
2824.9 to 3012.3 MiB. This mode is not part of the recommended path.

### EXP-T006: full-graph FP32 model-specific GPT-2 decode

- Implementation commits: `c4a0dce`, `aa46c2e`.
- Change: keep Transformers prefill unchanged, then copy its FP32 K/V values
  into fixed buffers and run the 24 GPT-2 decode blocks through a model-specific
  `torch.compile(fullgraph=True, dynamic=True)` callable. Positional embedding,
  layer norms, QKV projection, SDPA, output projection, MLP, residual ordering,
  logits processing, S3Gen, and watermark are preserved. CUDA graphs and TF32
  are disabled. Progress rendering is hidden.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected for performance; exact-output quality gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2530.25 +/- 6.31 | 22.13 +/- 0.12 | 2418.56 +/- 7.60 | 96.33 +/- 1.62 | 2.720 | 0.9302 +/- 0.0023 | 65 | 26.88 +/- 0.08 | 3754.5 |
| Medium | 6977.96 +/- 11.98 | 22.25 +/- 0.13 | 6822.89 +/- 8.62 | 114.73 +/- 0.78 | 7.360 | 0.9481 +/- 0.0016 | 181 | 26.53 +/- 0.03 | 3756.3 |
| Long | 24924.85 +/- 24.08 | 27.71 +/- 10.90 | 24457.15 +/- 39.07 | 273.13 +/- 0.34 | 25.080 | 0.9938 +/- 0.0010 | 624 | 25.51 +/- 0.04 | 3802.2 |

Every token and float-waveform tensor exactly matches EXP-T000 with maximum
absolute audio difference `0.0`. The stable measured runs prove compile startup
is excluded, but dynamic-prefix SDPA inside the monolithic compiled graph runs
far less efficiently than the native eager CUTLASS path. Short T3 latency rises
from 638.59 to 2418.56 ms and peak allocation rises from 2824.9 to 3754.5 MiB.
This mode is not part of the recommended path.

### EXP-T007: eager BF16 AR attention and KV cache

- Implementation commits: `c4a0dce`, `aa46c2e`.
- Change: use the model-specific decode path without `torch.compile`; keep all
  weights, QKV/MLP projections, logits, conditioning, S3Gen, and watermark in
  FP32, while casting only AR self-attention Q/K/V and its fixed cache to BF16.
  Progress rendering is hidden.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected for performance; exact-output quality gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 812.89 +/- 5.17 | 22.17 +/- 0.02 | 704.51 +/- 4.46 | 93.83 +/- 0.59 | 2.720 | 0.2989 +/- 0.0019 | 65 | 92.27 +/- 0.59 | 3016.9 |
| Medium | 2078.51 +/- 12.45 | 22.23 +/- 0.58 | 1925.96 +/- 7.67 | 115.97 +/- 2.67 | 7.360 | 0.2824 +/- 0.0017 | 181 | 93.98 +/- 0.37 | 3092.5 |
| Long | 7079.21 +/- 38.15 | 22.72 +/- 0.02 | 6641.70 +/- 37.86 | 273.11 +/- 2.38 | 25.080 | 0.2823 +/- 0.0015 | 624 | 93.95 +/- 0.54 | 3611.0 |

Every token and float-waveform tensor exactly matches EXP-T000 with maximum
absolute audio difference `0.0`. T3 latency is 704.51/1925.96/6641.70 ms versus
the untouched baseline's 638.59/1730.28/5947.03 ms. The fixed cache cuts element
size in half but retains a padded head stride, and the Q/K/V casts plus inefficient
attention layout outweigh the saved bandwidth. This mode is not part of the
recommended path.

### EXP-T008: full-graph BF16 AR attention and KV cache

- Implementation commits: `c4a0dce`, `aa46c2e`.
- Change: combine the model-specific full-graph decoder from EXP-T006 with the
  selective BF16 AR self-attention/cache configuration from EXP-T007. Weights,
  QKV/MLP projections, logits, conditioning, S3Gen, and watermark remain FP32.
  CUDA graphs and TF32 are disabled. Progress rendering is hidden.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected for performance; exact-output quality gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2390.52 +/- 8.02 | 22.42 +/- 0.15 | 2277.82 +/- 8.26 | 94.08 +/- 1.31 | 2.720 | 0.8789 +/- 0.0029 | 65 | 28.54 +/- 0.10 | 3278.5 |
| Medium | 6560.71 +/- 14.50 | 22.36 +/- 0.12 | 6404.37 +/- 13.29 | 114.79 +/- 2.45 | 7.360 | 0.8914 +/- 0.0020 | 181 | 28.26 +/- 0.06 | 3280.3 |
| Long | 22873.99 +/- 27.80 | 27.80 +/- 10.97 | 22390.87 +/- 11.44 | 273.03 +/- 0.26 | 25.080 | 0.9120 +/- 0.0011 | 624 | 27.87 +/- 0.01 | 3610.5 |

Every token and float-waveform tensor exactly matches EXP-T000 with maximum
absolute audio difference `0.0`. Selective BF16 improves the rejected compiled
FP32 path slightly, but it remains about four times slower than the untouched
native decoder. The compiler still lowers dynamic-prefix attention inefficiently,
so this mode is not part of the recommended path.
