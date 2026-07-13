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

### EXP-001: T3 kernel and KV-cache profile

- Profiler commits: `be758a1`, corrected sampling in `5cdc4de`.
- Workload: warmed short prompt, T3 only, 56 generated speech tokens.
- Quality check: token hash
  `22c2f704d30e1070065f4331ccdc77fca479fa5453511c7785be8604c17ca76c`
  exactly matches EXP-000.
- Result: diagnostic accepted; no model behavior changed.

Runtime findings:

- All 292 T3 parameter tensors and both buffers are FP32, occupying 2030.96 MiB.
- The cache is a 30-layer FP32 `DynamicCache`. First-layer K and V each have
  shape `[2, 16, sequence, 64]`, are contiguous, and grow by one position on
  every decode step.
- Although all SDP backends are enabled, decode selects
  `aten::_scaled_dot_product_efficient_attention` with the FP32 CUTLASS FMHA
  kernel. It does not select Flash Attention.
- The short profile contains 11,817 matrix multiplications, 1,682 attention
  calls, and 6,894 `aten::cat` calls.

| Operator group | Calls | CUDA total ms | Self CUDA ms | CPU total ms | Share of measured self CUDA |
|---|---:|---:|---:|---:|---:|
| Matrix multiplication (`aten::mm`) | 11,817 | 230.76 | 230.70 | 640.37 | 60.1% |
| Efficient attention forward | 1,682 | 42.55 | 42.55 | 82.55 | 11.1% |
| Dynamic-cache concatenation (`aten::cat`) | 6,894 | 22.99 | 22.99 | 192.91 | 6.0% |
| Sampling (`aten::multinomial`) | 56 | 1.85 | 0.00 | 23.83 | 0.5% total CUDA |
| Top-k internals used by min-p processing | 56 | 1.58 | 1.58 | 1.87 | 0.4% |

The memory-enabled capture attributed 2.38 GiB of cumulative CUDA allocation to
`aten::cat` even for the short case. Profiler execution time is intentionally not
used as a latency benchmark because operator tracing adds heavy CPU overhead.

The first diagnostic capture accidentally used T3's internal `top_p=0.95`
default. Its kernel and cache findings were the same, but its token hash was
excluded. The corrected capture explicitly uses the public API's `top_p=1.0`
and all other EXP-000 sampling values.

Next action: test BF16 T3 execution first. This targets the dominant FP32
matmuls, halves KV bandwidth, and should make SDPA eligible for the native Flash
Attention kernel. Static KV allocation remains the next independent target.

### EXP-002: cast the complete T3 path to BF16

- Benchmark-mode commit: `d6fcd30`.
- Change: cast T3 parameters and T3 conditioning tensors to BF16 after model
  loading. S3Gen, vocoder, public sampling values, and watermark stayed FP32 and
  unchanged.
- Runs: two warmups and five measured runs per prompt.
- Result: rejected as the default runtime path.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2673.07 +/- 65.64 | 48.05 +/- 7.21 | 1975.89 +/- 61.85 | 670.58 +/- 5.49 | 1.880 | 1.4218 +/- 0.0349 | 48 | 24.31 +/- 0.73 | 2140.9 |
| Medium | 7461.18 +/- 67.41 | 46.96 +/- 1.31 | 6700.33 +/- 59.17 | 712.02 +/- 14.13 | 6.440 | 1.1586 +/- 0.0105 | 162 | 24.18 +/- 0.21 | 2196.4 |
| Long | 21199.14 +/- 465.01 | 44.50 +/- 1.58 | 20069.99 +/- 468.41 | 1015.20 +/- 1.34 | 19.800 | 1.0707 +/- 0.0235 | 496 | 24.72 +/- 0.58 | 2490.8 |

Fixed-seed speech-token hashes:

- Short: `f7986ddd67e759b84c0ab25db0280769690753a6fb83c714cbe20c0d98eb8d72`
- Medium: `d37a9b8a0ab4f4f247705b4aa5b86ee9e4e91e648e4cf0a47aa7071a5fdaff4d`
- Long: `bdeab4b72e8afb9adaa3364975636542d70c073101bd35df6f59e0bd05d0709e`

BF16 reduced peak allocated memory by roughly 1 GiB and improved long-sequence
throughput, where KV bandwidth is expensive. It was slower for the identical
162-token medium workload: T3 increased from 5876.48 ms at 27.58 tokens/s to
6700.33 ms at 24.18 tokens/s. Short throughput also fell from 27.87 to 24.31
tokens/s. All three token hashes changed, and the short output stopped at 48
tokens instead of 56. The long-only speedup therefore does not justify a default
precision change without a separate perceptual-quality qualification.

### EXP-003: T3-only TF32 matmul precision

- Benchmark-mode commit: `f1bbd61`.
- Change: set `torch.set_float32_matmul_precision("high")` only while T3 runs,
  then restore `highest` before S3Gen. Parameters, activations, KV cache, S3Gen,
  sampling, and watermark remain FP32.
- Runs: two warmups and five measured runs per prompt.
- Result: accepted as an exact-token long-sequence candidate, rejected as an
  unconditional default because short and medium latency regressed.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2837.34 +/- 46.30 | 42.93 +/- 1.08 | 2135.90 +/- 40.88 | 673.74 +/- 12.25 | 2.200 | 1.2897 +/- 0.0210 | 56 | 26.23 +/- 0.50 | 3165.4 |
| Medium | 6842.07 +/- 128.54 | 51.88 +/- 16.72 | 6104.00 +/- 113.78 | 685.21 +/- 19.80 | 6.440 | 1.0624 +/- 0.0200 | 162 | 26.55 +/- 0.50 | 3245.8 |
| Long | 19379.95 +/- 197.45 | 45.10 +/- 1.85 | 18239.34 +/- 183.63 | 1015.36 +/- 1.65 | 19.640 | 0.9868 +/- 0.0101 | 492 | 26.98 +/- 0.27 | 3500.2 |

Every fixed-seed token hash exactly matches EXP-000:

- Short: `22c2f704d30e1070065f4331ccdc77fca479fa5453511c7785be8604c17ca76c`
- Medium: `63b826922acf7cb3b23cecabf6b1000512b8bb5e3461bb04100a89b33a136f60`
- Long: `b4a4420291d6207626df144d57c8cff2e7caf2efbe37cd49ed17d65813b9f1c6`

TF32 reduced long T3 time from 23701.25 ms to 18239.34 ms while preserving
the complete sampled sequence. Short T3 increased from 2009.22 ms to 2135.90
ms, and medium increased from 5876.48 ms to 6104.00 ms. The next experiment
will retain `highest` precision for early decode and switch to TF32 only after
the sequence is long enough for its faster kernel path to amortize overhead.

### EXP-004: adaptive TF32 after 192 tokens, forward-wrapper prototype

- Benchmark prototype commit: `942634b`.
- Change: keep `highest` FP32 matmuls for prefill and the first 192 generated
  tokens, then use TF32 for later T3 transformer forwards. Restore `highest`
  before S3Gen.
- Runs: two warmups and five measured runs per prompt.
- Result: exact-token policy accepted; Python forward-wrapper implementation
  superseded because it adds measurable short-request overhead.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2769.90 +/- 37.34 | 44.12 +/- 1.89 | 2071.19 +/- 39.06 | 671.15 +/- 16.08 | 2.200 | 1.2590 +/- 0.0170 | 56 | 27.05 +/- 0.51 | 3165.4 |
| Medium | 6694.25 +/- 67.89 | 44.53 +/- 0.75 | 5954.21 +/- 61.01 | 690.52 +/- 14.67 | 6.440 | 1.0395 +/- 0.0105 | 162 | 27.21 +/- 0.28 | 3245.8 |
| Long | 20102.63 +/- 16.99 | 46.72 +/- 1.04 | 18942.96 +/- 20.74 | 1017.67 +/- 3.47 | 19.640 | 1.0236 +/- 0.0009 | 492 | 25.97 +/- 0.03 | 3500.2 |

All three fixed-seed token hashes exactly match EXP-000. The 192-token policy
keeps short and medium on the original arithmetic path and reduces long T3 time
from 23701.25 ms to 18942.96 ms. However, wrapping every transformer forward in
an additional Python function increased short T3 time from 2009.22 ms to
2071.19 ms. EXP-005 moves the one-time switch directly into the existing token
loop to remove that prototype overhead.

### EXP-005: adaptive TF32 after 192 tokens, direct T3 loop

- Implementation commit: `855eb19`.
- Change: add an optional `tf32_after_tokens` policy directly to
  `T3.inference()`. The loop switches matmul precision once, after token 192,
  and restores the caller's original precision in `finally`.
- Runs: two warmups and five measured runs per prompt.
- Result: retained. All fixed-seed token hashes exactly match EXP-000.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2701.77 +/- 132.86 | 40.93 +/- 1.58 | 2027.07 +/- 119.40 | 649.48 +/- 29.39 | 2.200 | 1.2281 +/- 0.0604 | 56 | 27.70 +/- 1.56 | 3165.4 |
| Medium | 6491.25 +/- 53.04 | 46.51 +/- 11.59 | 5765.98 +/- 54.80 | 671.95 +/- 4.41 | 6.440 | 1.0080 +/- 0.0082 | 162 | 28.10 +/- 0.27 | 3245.8 |
| Long | 18658.39 +/- 296.71 | 45.76 +/- 0.56 | 17512.40 +/- 283.32 | 1011.87 +/- 1.07 | 19.640 | 0.9500 +/- 0.0151 | 492 | 28.10 +/- 0.45 | 3500.2 |

Exact-token references:

- Short: `22c2f704d30e1070065f4331ccdc77fca479fa5453511c7785be8604c17ca76c`
- Medium: `63b826922acf7cb3b23cecabf6b1000512b8bb5e3461bb04100a89b33a136f60`
- Long: `b4a4420291d6207626df144d57c8cff2e7caf2efbe37cd49ed17d65813b9f1c6`

Compared with EXP-000, long T3 time fell from 23701.25 ms to 17512.40 ms
and long end-to-end time fell from 24876.00 ms to 18658.39 ms. Medium T3
also fell from 5876.48 ms to 5765.98 ms. Short is statistically close to the
baseline and follows the same arithmetic path because it never reaches the
threshold. This is retained as the current best exact-token implementation.

### EXP-006: remove T3 loop allocation and unused outputs

- Implementation commit: `a62edc4`.
- Change: reuse `T3HuggingfaceBackend`, request only the transformer's final
  hidden state, preallocate generated-token storage, and create the CFG scalar
  once. EXP-005's adaptive TF32 policy remains enabled after token 192.
- Runs: two warmups and five measured runs per prompt.
- Result: retained. All fixed-seed token hashes exactly match EXP-000.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2571.13 +/- 17.86 | 40.17 +/- 1.48 | 1888.26 +/- 9.96 | 651.85 +/- 15.25 | 2.200 | 1.1687 +/- 0.0081 | 56 | 29.66 +/- 0.16 | 3164.9 |
| Medium | 6169.17 +/- 57.57 | 39.74 +/- 1.27 | 5469.44 +/- 61.93 | 650.51 +/- 7.40 | 6.440 | 0.9579 +/- 0.0089 | 162 | 29.62 +/- 0.33 | 3245.2 |
| Long | 18515.39 +/- 600.11 | 45.52 +/- 1.02 | 17367.73 +/- 602.83 | 1014.96 +/- 2.92 | 19.640 | 0.9427 +/- 0.0306 | 492 | 28.36 +/- 0.98 | 3500.2 |

Exact-token references remain:

- Short: `22c2f704d30e1070065f4331ccdc77fca479fa5453511c7785be8604c17ca76c`
- Medium: `63b826922acf7cb3b23cecabf6b1000512b8bb5e3461bb04100a89b33a136f60`
- Long: `b4a4420291d6207626df144d57c8cff2e7caf2efbe37cd49ed17d65813b9f1c6`

Compared with EXP-005, T3 fell from 2027.07 to 1888.26 ms on short and
from 5765.98 to 5469.44 ms on medium. Compared with the untouched baseline,
long T3 is 6333.52 ms lower. Medium and long are now faster than real time;
short remains above real time because the fixed 652 ms S3Gen cost is a larger
share of its 2.2-second output.

### EXP-007: preallocated dynamic-prefix FP32 KV cache

- Experimental implementation commit: `405f747`.
- Change: replace per-token dynamic K/V concatenation with fixed-capacity
  backing tensors and return only each cache's populated prefix. EXP-006 loop
  cleanup and adaptive TF32 remain enabled.
- Runs: one short smoke run, then two warmups and five measured runs per prompt.
- Result: rejected and removed after logging. All token hashes match EXP-000,
  but every workload is slower and memory use increases.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 2642.48 +/- 41.70 | 40.98 +/- 0.74 | 1962.28 +/- 25.57 | 653.47 +/- 17.26 | 2.200 | 1.2011 +/- 0.0190 | 56 | 28.54 +/- 0.37 | 3614.8 |
| Medium | 6502.72 +/- 71.64 | 42.50 +/- 2.15 | 5775.35 +/- 70.54 | 677.38 +/- 14.16 | 6.440 | 1.0097 +/- 0.0111 | 162 | 28.05 +/- 0.34 | 3648.6 |
| Long | 18896.82 +/- 162.30 | 45.64 +/- 0.54 | 17750.54 +/- 160.51 | 1014.39 +/- 1.67 | 19.640 | 0.9622 +/- 0.0083 | 492 | 27.72 +/- 0.25 | 3756.1 |

The smoke run produced the exact short hash but only 26.29 tokens/s. The full
run warmed to 28.54 tokens/s, still below EXP-006's 29.66. Long throughput also
fell from 28.36 to 27.72 tokens/s. Although the cache eliminates `aten::cat`,
its prefix is a non-contiguous view with full-capacity strides; slower attention
access outweighs the removed copies. Peak allocation increases by 450 MiB on
short and 256 MiB on long. A useful future fixed cache therefore needs a paged
attention kernel designed for its layout, not an eager SDPA view.

### EXP-008: Hugging Face/Liger fused RMSNorm kernel

- Code change: none. Runtime-only dependency experiment.
- First attempt: `kernels==0.16.0` failed during Transformers import because its
  `LayerRepository` API now requires a revision or version, while Transformers
  5.2's built-in RMSNorm mapping does not provide one.
- Corrected runtime: `kernels==0.10.5`, matching Transformers 5.2's declared
  `kernels>=0.10.2,<0.11` range.
- Workload: one short warmup and one timed short run, using EXP-006 plus adaptive
  TF32 after token 192.
- Result: rejected after the smoke benchmark; the regression is too large to
  justify a full three-case run. The runtime package was removed afterward.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short smoke | 3555.30 | 53.91 | 2696.63 | 833.39 | 2.200 | 1.6160 | 56 | 20.77 | 3164.9 |

The token hash is the exact EXP-000 short reference,
`22c2f704d30e1070065f4331ccdc77fca479fa5453511c7785be8604c17ca76c`.
Compared with EXP-006, T3 increased from 1888.26 ms to 2696.63 ms. The fused
kernel's launch path does not amortize for batch-2, one-token decoder rows on
this RTX 3090, so no dependency or model change was retained.

### EXP-009: compiled T3 decode with eager RoPE

- Implementation commits: `3c9394f`, `b5a3d0f`.
- Change: compile the repeated transformer decode forward with
  `torch.compile(dynamic=True, fullgraph=False, mode="default")`. Keep RoPE
  eager because compiling the rotary embedding buffer fails in TorchInductor
  with an `UntypedStorage` weak-reference error. EXP-006's loop cleanup and
  adaptive TF32 after token 192 remain enabled.
- Runs: two compile/warmup runs and five measured runs per prompt. The first
  process-level compile took approximately two minutes and is excluded from
  all measurements.
- Result: retained as the fastest exact-token implementation so far. All
  fixed-seed token hashes exactly match EXP-000.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1701.40 +/- 44.36 | 42.26 +/- 4.74 | 1008.04 +/- 31.68 | 665.62 +/- 17.88 | 2.200 | 0.7734 +/- 0.0202 | 56 | 55.60 +/- 1.73 | 3218.9 |
| Medium | 3611.79 +/- 81.17 | 41.20 +/- 1.05 | 2863.81 +/- 68.51 | 689.76 +/- 19.81 | 6.440 | 0.5608 +/- 0.0126 | 162 | 56.59 +/- 1.36 | 3376.2 |
| Long | 10837.73 +/- 142.40 | 45.44 +/- 0.56 | 9706.42 +/- 140.98 | 1018.26 +/- 1.62 | 19.640 | 0.5518 +/- 0.0073 | 492 | 50.70 +/- 0.74 | 3849.0 |

Exact-token references remain:

- Short: `22c2f704d30e1070065f4331ccdc77fca479fa5453511c7785be8604c17ca76c`
- Medium: `63b826922acf7cb3b23cecabf6b1000512b8bb5e3461bb04100a89b33a136f60`
- Long: `b4a4420291d6207626df144d57c8cff2e7caf2efbe37cd49ed17d65813b9f1c6`

Compared with the untouched EXP-000 baseline, short T3 falls from 2009.22 ms
to 1008.04 ms, medium T3 from 5876.48 ms to 2863.81 ms, and long T3 from
23701.25 ms to 9706.42 ms. End-to-end time falls from 2694.75 ms to 1701.40 ms
on short, 6617.03 ms to 3611.79 ms on medium, and 24876.00 ms to 10837.73 ms on
long. TTFT remains approximately 41-45 ms because compilation primarily
accelerates repeated decode. Peak allocation rises by 53.5 MiB on short,
131.0 MiB on medium, and 348.8 MiB on long relative to EXP-006.

This path has no checkpoint, sampling, decoder-step, precision, or watermark
change. Exact sampled speech-token identity across all prompts is the primary
quality gate. The deployment tradeoff is startup compilation latency and a
persistent TorchInductor cache, not generated-audio quality.

### EXP-010: `reduce-overhead` CUDA-graph capture, initial attempt

- Compile-mode support commit: `4486d4a`.
- Change: replace EXP-009's `mode="default"` with
  `mode="reduce-overhead"`, which enables TorchInductor CUDA graphs. No model,
  precision, cache, or sampling change.
- Workload: short smoke benchmark with compiler performance hints enabled.
- Result: failed during the first warmup; no latency result is reported.

TorchInductor began CUDA-graph execution but raised:

```text
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run
```

The reported source is Transformers' dynamic cache update, where each layer
assigns `torch.cat([self.values, value_states], dim=-2)`. The concatenated K/V
tensors are graph outputs and become inputs to the next decode invocation, but
the graph's static output storage is reused before that lifetime is understood.
The compiler specifically recommends calling
`torch.compiler.cudagraph_mark_step_begin()` before each model invocation or
cloning outputs outside the compiled region. The next attempt will test the
step marker first because it adds no arithmetic or KV copy.

#### EXP-010b: explicit CUDA-graph step markers

- Implementation commit: `d6df466`.
- Change: call `torch.compiler.cudagraph_mark_step_begin()` immediately before
  every compiled decode invocation in `reduce-overhead` mode.
- Result: rejected. The first warmup fails with the same overwritten graph
  output error at the dynamic K/V cache update, so the marker does not express
  the cross-step cache lifetime sufficiently for this model. No benchmark
  metrics or output tokens were produced.

The remaining compiler-prescribed path is to clone the K/V graph outputs after
each invocation and before they are reused. That adds a full cache copy per
token, so it will be evaluated as a smoke benchmark before any full run.

#### EXP-010c: clone dynamic K/V outputs outside the CUDA graph

- Implementation commit: `8b1b486`.
- Change: after each `reduce-overhead` decode, clone every initialized dynamic
  cache layer's K and V tensors before the next graph invocation. Keep the step
  boundary from EXP-010b after logits sampling.
- Runs: two warmups and two measured short runs.
- Result: successful exact-token short smoke; full evaluation pending.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short smoke | 1296.10 +/- 129.71 | 49.91 +/- 7.29 | 506.42 +/- 50.51 | 762.41 +/- 76.00 | 2.200 | 0.5891 +/- 0.0590 | 56 | 111.13 +/- 11.08 | 3164.4 |

The two T3 samples were 542.14 ms at 103.29 tokens/s and 470.70 ms at
118.97 tokens/s. The token hash exactly matches EXP-000:
`22c2f704d30e1070065f4331ccdc77fca479fa5453511c7785be8604c17ca76c`.

TorchInductor warned that dynamic shapes require a separate CUDA graph for each
distinct input size and reported at least nine shapes during warmup. The next
gate is therefore not only throughput: medium and long runs must establish
capture startup cost, graph count, and memory practicality before this path can
replace EXP-009.

#### EXP-010d: full CUDA-graph benchmark

- Configuration: EXP-010c unchanged, with two warmups and five measured runs
  for every prompt.
- Result: retained as the current fastest implementation. Every speech-token
  hash and final WAV file is exactly identical to EXP-000.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1217.62 +/- 26.92 | 45.02 +/- 3.11 | 472.70 +/- 14.70 | 714.51 +/- 17.89 | 2.200 | 0.5535 +/- 0.0122 | 56 | 118.56 +/- 3.61 | 3164.4 |
| Medium | 2203.27 +/- 36.21 | 45.13 +/- 3.51 | 1416.71 +/- 30.92 | 730.24 +/- 19.98 | 6.440 | 0.3421 +/- 0.0056 | 162 | 114.39 +/- 2.44 | 3243.8 |
| Long | 6736.75 +/- 58.28 | 46.98 +/- 2.73 | 5589.07 +/- 49.30 | 1021.73 +/- 4.73 | 19.640 | 0.3430 +/- 0.0030 | 492 | 88.03 +/- 0.77 | 3503.0 |

Compared with EXP-000, T3 time falls from 2009.22 ms to 472.70 ms on short,
5876.48 ms to 1416.71 ms on medium, and 23701.25 ms to 5589.07 ms on long.
End-to-end time falls from 2694.75 ms to 1217.62 ms, 6617.03 ms to 2203.27 ms,
and 24876.00 ms to 6736.75 ms respectively. TTFT remains in the same practical
range at 45.02-46.98 ms because the captured graph accelerates repeated decode,
not conditioning and prefill.

Final WAV equivalence:

| Case | Baseline and EXP-010 SHA-256 | Equal samples | Max absolute difference |
|---|---|---:|---:|
| Short | `6a8dcd0eda8cf4b22c6c6bc798b52c2352fc509b2df360b39ead9c0cd83f0073` | Yes | 0.0 |
| Medium | `2499b840f7f352b9139f492fb2edeffbc7d7e14a6994c3fa94e0925c6f9459f4` | Yes | 0.0 |
| Long | `547146b2b8784bec379f7e72c2f28168a51638cc6437fe1cf2d3b187dc283b7c` | Yes | 0.0 |

CUDA graphs are process-memory state: the first request for each unseen cache
length records a new graph. The benchmark excludes compilation and capture in
its two warmups. NVIDIA process memory reached approximately 6.4 GiB while the
long graph set was being captured; measured PyTorch peak allocation remained
3.16-3.50 GiB. Production integration therefore needs an explicit warmup API
and must document the startup/capture tradeoff.

### EXP-011: all-TF32 CUDA graphs

- Code change: none; use EXP-010 with T3 matmul precision set to `high` from
  prefill through the entire decode instead of switching after token 192.
- Runs: two warmups and two measured short runs.
- Result: successful exact-token smoke; full evaluation pending.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short smoke | 1127.91 +/- 11.59 | 39.44 +/- 0.19 | 448.00 +/- 12.32 | 653.97 +/- 1.06 | 2.200 | 0.5127 +/- 0.0053 | 56 | 125.05 +/- 3.44 | 3164.4 |

The two T3 samples were 456.71 ms at 122.62 tokens/s and 439.29 ms at
127.48 tokens/s. The token hash exactly matches EXP-000. Unlike eager EXP-003,
CUDA-graph replay removes enough launch overhead for the tensor-core matmul
path to improve short decode. Medium and long fixtures remain required before
this precision policy can replace EXP-010's adaptive threshold.

#### EXP-011 full benchmark

- Runs: two warmups and five measured runs per prompt.
- Result: retained as the new fastest exact-audio configuration.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1153.42 +/- 9.11 | 40.53 +/- 0.81 | 448.17 +/- 7.76 | 672.49 +/- 2.11 | 2.200 | 0.5243 +/- 0.0041 | 56 | 124.98 +/- 2.12 | 3164.4 |
| Medium | 2096.30 +/- 24.77 | 45.92 +/- 6.78 | 1357.96 +/- 23.55 | 684.53 +/- 6.55 | 6.440 | 0.3255 +/- 0.0038 | 162 | 119.32 +/- 2.05 | 3243.8 |
| Long | 6620.37 +/- 47.17 | 43.34 +/- 1.55 | 5488.09 +/- 39.32 | 1023.99 +/- 3.46 | 19.640 | 0.3371 +/- 0.0024 | 492 | 89.65 +/- 0.64 | 3503.0 |

All speech-token and WAV SHA-256 hashes exactly match EXP-000 and EXP-010.
Compared with EXP-010, all-TF32 lowers T3 from 472.70 ms to 448.17 ms on
short, 1416.71 ms to 1357.96 ms on medium, and 5589.07 ms to 5488.09 ms on
long. End-to-end means fall from 1217.62 ms to 1153.42 ms, 2203.27 ms to
2096.30 ms, and 6736.75 ms to 6620.37 ms respectively.

NVIDIA process memory reached approximately 8.0 GiB after capturing all prompt
shape sets in this run. The per-run PyTorch peak allocation remains unchanged
at 3.16-3.50 GiB. The all-TF32 policy also removes the mid-generation precision
switch, so it is simpler to expose as the optimized public configuration.

### EXP-012: public optimized API validation

- Implementation commit: `6261a3b`.
- Change: expose compile mode, matmul precision, adaptive TF32, and progress
  control through `ChatterboxTTS.generate()`. Refactor the benchmark so timing
  wrappers only measure calls and all optimization settings enter through this
  public method.
- Runs: two warmups and two measured short runs using the EXP-011 settings.
- Result: retained; public API wiring validated with the exact EXP-000 token
  hash.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short smoke | 1177.25 +/- 13.68 | 41.16 +/- 0.18 | 466.39 +/- 16.97 | 683.30 +/- 2.55 | 2.200 | 0.5351 +/- 0.0062 | 56 | 120.15 +/- 4.37 | 3164.4 |

The two T3 samples were 478.40 ms at 117.06 tokens/s and 454.39 ms at
123.24 tokens/s. They are within the run-to-run range of EXP-011 and prove that
the published API executes the same exact-output CUDA-graph path instead of a
benchmark-only monkeypatch.

### EXP-013: optimized CUDA-graph kernel profile

- Profiler commit: `9fcfe64`.
- Workload: T3-only short prompt after two EXP-011 warmups, with PyTorch CPU and
  CUDA operator tracing enabled.
- Quality check: 56 tokens with the exact EXP-000 hash.
- Result: diagnostic accepted; profiler execution time is not a latency result.

| Operator/group | Calls | CUDA total ms | CPU total ms |
|---|---:|---:|---:|
| Compiled transformer graph | 55 | 280.33 | 179.05 |
| K/V and graph copies (`aten::copy_`) | 7,337 | 21.26 | 134.91 |
| K/V graph-output clones (`aten::clone`) | 3,412 | 10.38 | 103.68 |
| CUDA graph launches | 220 | 9.23 | 70.81 |
| Eager matmul outside graph | 213 | 7.31 | 13.57 |
| Multinomial sampling | 56 | 1.75 | 22.69 |
| Eager attention outside graph/prefill | 32 | 0.58 | 2.38 |

Inside graph replay, CUTLASS tensor-core matmuls consume approximately
180.49 ms and FP32 memory-efficient attention consumes 41.28 ms. The graph is
already using generated Triton fusion kernels for normalization, activation,
mask, and cache-related elementwise sequences.

The actionable overhead outside the graph is cache ownership: cloning 30 K and
30 V tensors after each decode creates roughly 3,300 clone dispatches for a
56-token request. The next experiment will copy those identically shaped
tensors into one stacked contiguous allocation per step and assign contiguous
views back to the cache, preserving values while collapsing dispatch count.

### EXP-014: batched stacked K/V ownership copy

- Implementation commit: `9d180a8`.
- Change: replace 60 independent K/V `clone()` operations per generated token
  with one `torch.stack()` allocation and one `unbind()` into contiguous layer
  views. The copy still occurs outside the compiled graph and preserves the
  graph-output lifetime fix from EXP-010.
- Runs: two warmups and five measured short runs.
- Result: successful exact-audio short benchmark; full evaluation pending.

| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1109.68 +/- 25.16 | 39.25 +/- 2.40 | 440.88 +/- 16.13 | 642.82 +/- 20.28 | 2.200 | 0.5044 +/- 0.0114 | 56 | 127.15 +/- 4.49 | 3218.7 |

The speech-token hash and final WAV hash exactly match EXP-000. Compared with
EXP-011, short T3 falls from 448.17 ms to 440.88 ms and throughput rises from
124.98 to 127.15 tokens/s. Peak allocation rises from 3164.4 MiB to
3218.7 MiB because the stacked ownership buffer is live as the next graph
input. Medium and long runs are required to determine whether the dispatch
reduction continues to outweigh the larger contiguous copy.
