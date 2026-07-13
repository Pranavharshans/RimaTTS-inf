# RTX 5090 benchmark

Date: 2026-07-13

Hardware: Vast.ai instance `44697129`, NVIDIA GeForce RTX 5090, 32,607 MiB
VRAM, driver `580.159.03`, CUDA `13.0`, PyTorch `2.12.0+cu130`, capability
`12.0`.

All cases used two warmups and five timed runs. Prompts, seeds, sampling,
watermarking, checkpoints, and output length were held constant between each
baseline and optimized pair. Timings are means across the five runs.

## Regular model

Baseline: FP32 T3, highest matmul precision, eager decode.

Optimized: FP32 T3, high matmul precision, `torch.compile` with
`reduce-overhead` CUDA graphs, existing K/V graph-output lifetime clones.

| Config | Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | RTF | Tokens | Tok/s | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Short | 1636.07 | 91.84 | 1183.26 | 401.29 | 0.7437 | 56 | 47.33 | 3188.8 |
| Baseline | Medium | 3800.21 | 86.74 | 3275.58 | 415.53 | 0.5901 | 162 | 49.46 | 3268.6 |
| Baseline | Long | 10450.44 | 25.90 | 9752.17 | 405.22 | 0.5321 | 492 | 50.45 | 3524.1 |
| Optimized | Short | 858.07 | 69.44 | 373.99 | 419.52 | 0.3900 | 56 | 150.86 | 3187.1 |
| Optimized | Medium | 1500.42 | 71.82 | 979.38 | 418.91 | 0.2330 | 162 | 165.59 | 3266.0 |
| Optimized | Long | 4405.04 | 25.99 | 3761.27 | 409.37 | 0.2243 | 492 | 133.59 | 3525.9 |

## Turbo model

Baseline: native FP32 decode and eager logits processing.

Optimized: native FP32 one-token decode graph plus compiled logits processing;
CUDA graphs remain disabled for the Turbo native path.

| Config | Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | RTF | Tokens | Tok/s | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Short | 1000.71 | 61.00 | 835.93 | 112.61 | 0.3679 | 65 | 77.82 | 2849.2 |
| Baseline | Medium | 2402.46 | 45.85 | 2149.27 | 113.22 | 0.3264 | 181 | 84.23 | 2924.0 |
| Baseline | Long | 7695.28 | 33.95 | 7226.93 | 124.84 | 0.3068 | 624 | 86.34 | 3442.9 |
| Optimized | Short | 678.76 | 76.96 | 509.21 | 109.94 | 0.2495 | 65 | 128.19 | 2916.4 |
| Optimized | Medium | 1483.18 | 14.50 | 1213.83 | 112.55 | 0.2015 | 181 | 149.12 | 2965.6 |
| Optimized | Long | 4753.23 | 25.30 | 4282.28 | 126.70 | 0.1895 | 624 | 145.85 | 3442.7 |

## Quality checks

Regular optimized token files matched the regular baseline byte-for-byte:

| Case | SHA-256 |
|---|---|
| Short | `3dc73f58265f07f2f3cb4b6ad82a0155a8b449256333d4b9a1a638b7bed42201` |
| Medium | `d547150269c142a9fd78c00da406f09fbd1f733728482bc150762fa520db1ed2` |
| Long | `9401b5808ff0a3d06c889a30ce2fa429f9a5fc05f4d02e8ba1a342eee0f49192` |

Turbo's exact-output gate also passed for all three prompts. Optimized Turbo
token and waveform hashes equal the baseline hashes; no sampling, checkpoint,
decoder-step, or watermark change was made.

Raw reports are archived under `benchmark_results/5090/`.

## 3090 reference

| Model | Case | E2E ms | T3 ms | RTF | Tok/s |
|---|---|---:|---:|---:|---:|
| Regular EXP-018 | Short | 1122.73 | 434.91 | 0.5103 | 128.80 |
| Regular EXP-018 | Medium | 2070.48 | 1343.21 | 0.3215 | 120.62 |
| Regular EXP-018 | Long | 6583.55 | 5451.26 | 0.3352 | 90.27 |
| Turbo EXP-T021 | Short | 484.21 | 375.18 | 0.1780 | 173.25 |
| Turbo EXP-T021 | Medium | 1158.73 | 1011.55 | 0.1574 | 178.94 |
| Turbo EXP-T021 | Long | 4520.54 | 4088.59 | 0.1802 | 152.62 |
