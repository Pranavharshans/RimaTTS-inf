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

### EXP-T009: native full-graph decode with contiguous DynamicCache

- Implementation commit: `f3aa53d`.
- Change: keep the untouched Transformers prefill and native FP32 DynamicCache,
  but compile only `GPT2Model.forward()` for one-token decode with
  `fullgraph=True`, dynamic shapes enabled, and CUDA graphs disabled. This keeps
  K/V tensors contiguous and preserves the native CUTLASS SDPA path while
  eliminating Python/dispatcher overhead around the 24 transformer blocks.
  Sampling, S3Gen, and watermark are unchanged. Progress rendering is hidden.
- Runs: two warmups and five measured runs per prompt.
- Result: retained; exact-output quality gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 498.52 +/- 12.86 | 22.40 +/- 0.20 | 385.49 +/- 12.00 | 94.58 +/- 1.70 | 2.720 | 0.1833 +/- 0.0047 | 65 | 168.74 +/- 5.21 | 2895.5 |
| Medium | 1166.60 +/- 13.72 | 22.02 +/- 0.06 | 1015.41 +/- 13.64 | 110.83 +/- 0.36 | 7.360 | 0.1585 +/- 0.0019 | 181 | 178.28 +/- 2.41 | 2940.8 |
| Long | 4616.80 +/- 31.60 | 27.40 +/- 10.19 | 4160.13 +/- 26.58 | 268.47 +/- 0.98 | 25.080 | 0.1841 +/- 0.0013 | 624 | 150.00 +/- 0.95 | 3419.0 |

| Case | Baseline E2E ms | EXP-T009 E2E ms | Baseline T3 ms | EXP-T009 T3 ms | Baseline tok/s | EXP-T009 tok/s | Baseline RTF | EXP-T009 RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 749.64 | 498.52 | 638.59 | 385.49 | 101.80 | 168.74 | 0.2756 | 0.1833 |
| Medium | 1887.27 | 1166.60 | 1730.28 | 1015.41 | 104.61 | 178.28 | 0.2564 | 0.1585 |
| Long | 6384.64 | 4616.80 | 5947.03 | 4160.13 | 104.93 | 150.00 | 0.2546 | 0.1841 |

Every token and float-waveform tensor exactly matches EXP-T000 with maximum
absolute audio difference `0.0`. Prefill is intentionally unchanged, so stable
short and medium TTFT remains about 22 ms. Compilation happens during warmup and
is excluded from measured runs. This is the current recommended Turbo path.

#### EXP-T009 retained-path profile

- Profiler implementation commit: `1c6f9ef`.
- Workload: short prompt, two warmups, T3 only, exact EXP-T000 token hash.
- Cache: FP32 `DynamicCache`, 24 layers, contiguous K/V from prefill shape
  `[1,16,386,64]` through final shape `[1,16,451,64]`.
- Attention: SDPA still selects
  `fmha_cutlassF_f32_aligned_64x64_rf_sm80`; compile does not switch kernels.

| Operator group | Calls | CUDA self ms | Observation |
|---|---:|---:|---|
| Efficient attention | 1,584 | 126.86 | Largest remaining GPU component |
| `mm` | 4,680 | 78.91 | FP32 projection/MLP GEMV work |
| `addmm` | 1,723 | 41.79 | FP32 projection/MLP GEMV work |
| Top-p processor annotation | 66 | 14.96 | Profiler annotation; includes children |
| Repetition processor annotation | 66 | 7.41 | Profiler annotation; includes children |
| Top-k processor annotation | 66 | 5.21 | Profiler annotation; includes children |
| Visible `cat` | 117 | 0.51 | Down from 3,237 and 15.45 ms in EXP-T001 |

The profiler reports 287.56 ms total CUDA self-time. Efficient attention is
44% of that total, while `mm` plus `addmm` is another 42%. The compiled graph
fuses most cache growth into Triton kernels, so replacing DynamicCache storage
is no longer the high-value target. The next attention experiment must retain
contiguous dynamic K/V layout.

### EXP-T010: contiguous dynamic BF16 AR attention

- Implementation commit: `8ece468`.
- Change: preserve untouched FP32 prefill, weights, QKV/MLP projections,
  logits, sampling, S3Gen, and watermark. After the first sampled token, convert
  only AR self-attention K/V to BF16, append with dynamic `torch.cat` so every
  cache remains contiguous, cast Q to BF16 for SDPA, and compile the model-specific
  decode with full graph and CUDA graphs disabled. Progress rendering is hidden.
- Runs: two warmups and five measured runs per prompt.
- Result: retained only as a long-sequence candidate; exact-output gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 515.54 +/- 2.85 | 21.94 +/- 0.03 | 403.23 +/- 2.75 | 95.15 +/- 0.34 | 2.720 | 0.1895 +/- 0.0010 | 65 | 161.21 +/- 1.10 | 2879.0 |
| Medium | 1236.33 +/- 2.32 | 21.98 +/- 0.03 | 1080.73 +/- 2.63 | 112.48 +/- 0.69 | 7.360 | 0.1680 +/- 0.0003 | 181 | 167.48 +/- 0.41 | 2959.2 |
| Long | 4222.88 +/- 40.23 | 22.62 +/- 0.04 | 3753.36 +/- 61.07 | 266.86 +/- 0.29 | 25.080 | 0.1684 +/- 0.0016 | 624 | 166.29 +/- 2.69 | 3522.1 |

Every token and float-waveform tensor exactly matches EXP-T000 with maximum
absolute audio difference `0.0`. Against EXP-T009, T3 is slower on short
(`403.23` versus `385.49` ms) and medium (`1080.73` versus `1015.41` ms), but
faster on long (`3753.36` versus `4160.13` ms). The BF16 conversion has a fixed
cost and becomes worthwhile only after the attention sequence grows. EXP-T009
remains the general recommended mode; EXP-T010 motivates a late-switch hybrid.

### EXP-T011: FP32-to-BF16 hybrid at decode token 192

- Implementation commit: `97de8be`.
- Change: use the exact-output EXP-T009 native compiled FP32 decoder for the
  first 192 decode iterations, then convert the accumulated AR self-attention
  K/V cache to contiguous BF16 and continue with the EXP-T010 decoder. All
  weights, projections, logits, sampling, S3Gen, and watermark remain FP32.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: rejected by the exact-output quality gate.

| Path | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-T000 baseline | 6384.64 | 22.87 | 5947.03 | 268.56 | 25.080 | 0.2546 | 624 | 104.93 | 3419.0 |
| EXP-T009 FP32 compiled | 4616.80 | 27.40 | 4160.13 | 268.47 | 25.080 | 0.1841 | 624 | 150.00 | 3419.0 |
| EXP-T010 BF16 from prefill | 4222.88 | 22.62 | 3753.36 | 266.86 | 25.080 | 0.1684 | 624 | 166.29 | 3522.1 |
| EXP-T011 hybrid at 192 | 4027.14 | 22.74 | 3572.54 | 266.00 | 24.760 | 0.1626 | 616 | 172.43 | 3508.4 |

The timed run was deterministic, but it ended at 616 tokens instead of the
baseline's 624. Its token hash was
`bd39b01e80ff9355f1fecb8d5a949e47ff54ea350768497e3fb9def250fc5fe7`
instead of
`e22a7ab72ba83a76c57a376deb555f9d5d59155705124d80caca95f4548ec265`;
the waveform shape and hash also differed. The lower E2E time and RTF are
therefore partly caused by generating shorter audio and are not accepted as a
performance improvement. A late cache conversion can perturb the sampling
trajectory even though BF16 decode from prefill passed the same exact gate.

### EXP-T012: FP32-to-BF16 hybrid at decode token 320

- Implementation commit: `97de8be`.
- Change: repeat the hybrid qualification with the cache conversion delayed
  from decode token 192 to token 320.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: retained as an exact-output hybrid candidate.

| Path | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-T000 baseline | 6384.64 | 22.87 | 5947.03 | 268.56 | 25.080 | 0.2546 | 624 | 104.93 | 3419.0 |
| EXP-T009 FP32 compiled | 4616.80 | 27.40 | 4160.13 | 268.47 | 25.080 | 0.1841 | 624 | 150.00 | 3419.0 |
| EXP-T010 BF16 from prefill | 4222.88 | 22.62 | 3753.36 | 266.86 | 25.080 | 0.1684 | 624 | 166.29 | 3522.1 |
| EXP-T012 hybrid at 320 | 4345.50 | 22.74 | 3925.26 | 267.92 | 25.080 | 0.1733 | 624 | 158.97 | 3522.2 |

The generated token hash is
`e22a7ab72ba83a76c57a376deb555f9d5d59155705124d80caca95f4548ec265`
and the audio hash is
`739b09408460a41e481aa72bee70b776afb424117a4c89a21ebce7d1289a26b7`,
exactly matching EXP-T000 with maximum absolute audio difference `0.0`.
This threshold is valid but still slower on the long case than BF16 decode from
prefill, so an earlier exact switch remains worth testing.

### EXP-T013: FP32-to-BF16 hybrid at decode token 256

- Implementation commit: `97de8be`.
- Change: move the hybrid cache conversion from decode token 320 to token 256.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: retained as the new exact-output hybrid candidate.

| Path | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-T000 baseline | 6384.64 | 22.87 | 5947.03 | 268.56 | 25.080 | 0.2546 | 624 | 104.93 | 3419.0 |
| EXP-T009 FP32 compiled | 4616.80 | 27.40 | 4160.13 | 268.47 | 25.080 | 0.1841 | 624 | 150.00 | 3419.0 |
| EXP-T010 BF16 from prefill | 4222.88 | 22.62 | 3753.36 | 266.86 | 25.080 | 0.1684 | 624 | 166.29 | 3522.1 |
| EXP-T013 hybrid at 256 | 4159.72 | 22.82 | 3757.60 | 268.72 | 25.080 | 0.1659 | 624 | 166.06 | 3520.9 |

Tokens and waveform are bit-exact against EXP-T000, including maximum absolute
audio difference `0.0`. The single-run T3 result is effectively tied with the
five-run BF16-from-prefill result, while short and medium requests below the
switch threshold continue to use the faster EXP-T009 path. A full benchmark is
deferred until the earliest exact transition point is identified.

### EXP-T014: FP32-to-BF16 hybrid at decode token 224

- Implementation commit: `97de8be`.
- Change: move the hybrid cache conversion from decode token 256 to token 224.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: retained as the new exact-output hybrid candidate.

| Path | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-T000 baseline | 6384.64 | 22.87 | 5947.03 | 268.56 | 25.080 | 0.2546 | 624 | 104.93 | 3419.0 |
| EXP-T010 BF16 from prefill | 4222.88 | 22.62 | 3753.36 | 266.86 | 25.080 | 0.1684 | 624 | 166.29 | 3522.1 |
| EXP-T013 hybrid at 256 | 4159.72 | 22.82 | 3757.60 | 268.72 | 25.080 | 0.1659 | 624 | 166.06 | 3520.9 |
| EXP-T014 hybrid at 224 | 4128.75 | 22.79 | 3697.99 | 268.90 | 25.080 | 0.1646 | 624 | 168.74 | 3522.5 |

The token and waveform hashes exactly match EXP-T000 and maximum absolute audio
difference remains `0.0`. This is the fastest valid hybrid qualification so far,
although the one-run result still requires a full repeated benchmark after the
transition boundary search is complete.

### EXP-T015: FP32-to-BF16 hybrid at decode token 208

- Implementation commit: `97de8be`.
- Change: move the hybrid cache conversion from decode token 224 to token 208.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: exact-output gate passed; retained for boundary search only.

| Path | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-T000 baseline | 6384.64 | 22.87 | 5947.03 | 268.56 | 25.080 | 0.2546 | 624 | 104.93 | 3419.0 |
| EXP-T014 hybrid at 224 | 4128.75 | 22.79 | 3697.99 | 268.90 | 25.080 | 0.1646 | 624 | 168.74 | 3522.5 |
| EXP-T015 hybrid at 208 | 4188.20 | 22.76 | 3750.41 | 266.86 | 25.080 | 0.1670 | 624 | 166.38 | 3523.2 |

Tokens and waveform remain bit-exact with maximum absolute audio difference
`0.0`. The slower single sample relative to token 224 is within the scale of
run-to-run clock and kernel variation seen on this host, so threshold selection
will use repeated measurements after the exact-output boundary is known.

### EXP-T016: FP32-to-BF16 hybrid at decode token 200

- Implementation commit: `97de8be`.
- Change: move the hybrid cache conversion from decode token 208 to token 200.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: exact-output gate passed; retained for boundary search only.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Long | 4355.13 | 39.71 | 3882.25 | 267.81 | 25.080 | 0.1736 | 624 | 160.73 | 3526.6 |

Tokens and waveform exactly match EXP-T000 with maximum absolute audio
difference `0.0`. This run's TTFT is `39.71` ms instead of the otherwise stable
approximately `22.7` ms, and the whole run is correspondingly noisy. It is used
only to establish that token 200 preserves output, not to rank performance.

### EXP-T017: FP32-to-BF16 hybrid at decode token 196

- Implementation commit: `97de8be`.
- Change: move the hybrid cache conversion from decode token 200 to token 196.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: exact-output gate passed; retained for boundary search only.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Long | 4233.05 | 22.85 | 3796.71 | 268.62 | 25.080 | 0.1688 | 624 | 164.35 | 3521.3 |

Tokens and waveform exactly match EXP-T000 with maximum absolute audio
difference `0.0`. The valid/invalid transition now lies between tokens 192 and
196; timing remains qualification-only until that boundary is resolved.

### EXP-T018: FP32-to-BF16 hybrid at decode token 194

- Implementation commit: `97de8be`.
- Change: move the hybrid cache conversion from decode token 196 to token 194.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: exact-output gate passed; retained for final boundary resolution.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Long | 4140.89 | 51.94 | 3690.69 | 268.21 | 25.080 | 0.1651 | 624 | 169.07 | 3523.5 |

Tokens and waveform exactly match EXP-T000 with maximum absolute audio
difference `0.0`. TTFT is another isolated outlier and this one-run timing is
not used for ranking. Token 193 is the only remaining boundary point between
this valid result and the rejected token-192 transition.

### EXP-T019: FP32-to-BF16 hybrid at decode token 193

- Implementation commit: `97de8be`.
- Change: move the hybrid cache conversion from decode token 194 to token 193.
- Qualification: one warmup and one measured canonical long-prompt run.
- Result: exact-output gate passed; earliest valid boundary for this workload.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Long | 4137.45 | 22.74 | 3700.81 | 267.95 | 25.080 | 0.1650 | 624 | 168.61 | 3521.3 |

Tokens and waveform exactly match EXP-T000 with maximum absolute audio
difference `0.0`. Token 192 is rejected and token 193 passes, so no earlier
integer transition remains to test for this deterministic benchmark. Token 193
advances to the full short/medium/long repeated benchmark.

### EXP-T020: full token-193 hybrid benchmark

- Implementation commit: `97de8be`.
- Change: run the token-193 hybrid selected by EXP-T011 through EXP-T019 across
  all canonical workloads. Short and medium finish before the transition and
  therefore use the EXP-T009 FP32 compiled path throughout. Long switches only
  AR self-attention Q/K/V and cache storage to BF16 at decode iteration 193.
- Runs: two warmups and five measured runs per prompt.
- Result: retained as the fastest benchmark-exact candidate.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 481.58 +/- 1.51 | 22.23 +/- 0.03 | 371.77 +/- 1.60 | 92.71 +/- 0.63 | 2.720 | 0.1771 +/- 0.0006 | 65 | 174.84 +/- 0.75 | 2895.5 |
| Medium | 1167.04 +/- 24.96 | 22.09 +/- 0.02 | 1020.66 +/- 25.21 | 110.96 +/- 0.49 | 7.360 | 0.1586 +/- 0.0034 | 181 | 177.42 +/- 4.25 | 2940.8 |
| Long | 4045.36 +/- 23.56 | 28.48 +/- 12.56 | 3615.50 +/- 19.67 | 268.19 +/- 0.39 | 25.080 | 0.1613 +/- 0.0009 | 624 | 172.59 +/- 0.94 | 3521.9 |

| Case | Baseline E2E ms | EXP-T020 E2E ms | Baseline T3 ms | EXP-T020 T3 ms | Baseline tok/s | EXP-T020 tok/s | Baseline RTF | EXP-T020 RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 749.64 | 481.58 | 638.59 | 371.77 | 101.80 | 174.84 | 0.2756 | 0.1771 |
| Medium | 1887.27 | 1167.04 | 1730.28 | 1020.66 | 104.61 | 177.42 | 0.2564 | 0.1586 |
| Long | 6384.64 | 4045.36 | 5947.03 | 3615.50 | 104.93 | 172.59 | 0.2546 | 0.1613 |

All 15 timed runs are deterministic and exactly match EXP-T000 speech-token
and float-waveform tensors; maximum absolute audio difference is `0.0` in all
three cases. Long TTFT has a `22.89` ms median; one `50.95` ms host-side outlier
raises its reported mean to `28.48` ms.

This gate proves exactness for the fixed benchmark prompts and seeds, not for
every possible sampling trajectory. EXP-T011 demonstrated that a one-token
earlier BF16 transition can change a later sampled token. Therefore EXP-T020 is
an opt-in benchmark-exact candidate, while the all-FP32 EXP-T009 path remains
the strict quality-neutral default for arbitrary prompts.

### EXP-T021: full-graph logits processing on strict FP32 decode

- Implementation commit: `955849a`.
- Change: keep the retained EXP-T009 all-FP32 transformer path and compile the
  deterministic logits pipeline as a separate dynamic full graph. Temperature,
  top-k, top-p, repetition penalty, and softmax retain the exact upstream
  operation order. CUDA graphs remain disabled and multinomial RNG stays
  outside the compiled graph.
- Runs: two warmups and five measured runs per prompt.
- Result: retained as the new strict quality-neutral default.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 484.21 +/- 2.13 | 22.17 +/- 0.05 | 375.18 +/- 2.01 | 94.39 +/- 0.75 | 2.720 | 0.1780 +/- 0.0008 | 65 | 173.25 +/- 0.93 | 2895.5 |
| Medium | 1158.73 +/- 5.66 | 22.14 +/- 0.03 | 1011.55 +/- 5.85 | 112.56 +/- 1.52 | 7.360 | 0.1574 +/- 0.0008 | 181 | 178.94 +/- 1.03 | 2940.8 |
| Long | 4520.54 +/- 21.65 | 28.52 +/- 12.77 | 4088.59 +/- 20.92 | 268.48 +/- 0.43 | 25.080 | 0.1802 +/- 0.0009 | 624 | 152.62 +/- 0.78 | 3419.0 |

| Case | EXP-T009 E2E ms | EXP-T021 E2E ms | EXP-T009 T3 ms | EXP-T021 T3 ms | EXP-T009 tok/s | EXP-T021 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Short | 498.52 | 484.21 | 385.49 | 375.18 | 168.74 | 173.25 |
| Medium | 1166.60 | 1158.73 | 1015.41 | 1011.55 | 178.28 | 178.94 |
| Long | 4616.80 | 4520.54 | 4160.13 | 4088.59 | 150.00 | 152.62 |

All 15 timed runs exactly match EXP-T000 token and waveform tensors with
maximum absolute audio difference `0.0`. The processor's eager implementation
also has a unit test requiring exact tensor equality against Transformers.
Long TTFT median is `22.88` ms; one `51.36` ms host-side outlier raises the
mean. Unlike the hybrid path, this experiment does not change any model or
cache dtype and is retained for arbitrary prompts.

### EXP-T022: token-193 hybrid plus compiled logits

- Implementation commits: `97de8be`, `955849a`.
- Change: combine the benchmark-exact token-193 BF16 attention transition from
  EXP-T020 with the full-graph logits processor retained in EXP-T021.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected for performance; exact-output gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 483.17 +/- 6.32 | 22.08 +/- 0.03 | 374.02 +/- 5.85 | 94.18 +/- 0.64 | 2.720 | 0.1776 +/- 0.0023 | 65 | 173.82 +/- 2.67 | 2895.5 |
| Medium | 1162.72 +/- 6.15 | 22.06 +/- 0.04 | 1011.42 +/- 6.42 | 113.02 +/- 2.07 | 7.360 | 0.1580 +/- 0.0008 | 181 | 178.96 +/- 1.14 | 2940.8 |
| Long | 4148.19 +/- 26.23 | 33.08 +/- 14.42 | 3719.46 +/- 24.36 | 267.43 +/- 0.26 | 25.080 | 0.1654 +/- 0.0010 | 624 | 167.77 +/- 1.10 | 3521.9 |

All tokens and waveform tensors exactly match EXP-T000 with maximum absolute
audio difference `0.0`. Short and medium are effectively tied with EXP-T021,
but long regresses from EXP-T020's `3615.50` ms T3 and `172.59` tokens/s to
`3719.46` ms and `167.77` tokens/s. The two compiled graphs interfere enough
with the long decode schedule to erase the logits gain, so this combination is
not retained.

### EXP-T023: fused FP32 native token step

- Implementation commit: `db3c11e`.
- Change: compile speech embedding lookup, native GPT-2 decode, speech-head
  projection, temperature/top-k/top-p/repetition processing, and softmax as one
  dynamic full graph. Multinomial sampling, EOS handling, prefill, all dtypes,
  S3Gen, and watermark remain unchanged.
- Runs: two warmups and five measured runs per prompt after a one-run short
  qualification passed at `370.34` ms T3.
- Result: rejected for performance; exact-output gate passed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 514.17 +/- 10.07 | 21.94 +/- 0.07 | 396.91 +/- 8.88 | 96.45 +/- 2.72 | 2.720 | 0.1890 +/- 0.0037 | 65 | 163.83 +/- 3.65 | 2895.7 |
| Medium | 1351.03 +/- 78.94 | 25.12 +/- 6.22 | 1172.50 +/- 69.09 | 120.26 +/- 7.28 | 7.360 | 0.1836 +/- 0.0107 | 181 | 154.81 +/- 9.28 | 2941.0 |
| Long | 4956.53 +/- 150.73 | 27.91 +/- 11.01 | 4452.10 +/- 155.59 | 272.49 +/- 4.83 | 25.080 | 0.1976 +/- 0.0060 | 624 | 140.29 +/- 4.81 | 3419.0 |

All token and waveform tensors exactly match EXP-T000 with maximum absolute
audio difference `0.0`. The fused graph is slower than EXP-T021 in every case
and becomes highly variable as sequence length grows. Combining dynamic-cache
attention and sort/top-k logits kernels in one graph prevents the compiler from
maintaining the efficient schedule produced by the smaller graph boundaries.
This mode remains opt-in for reproducibility and is not recommended.
