#!/usr/bin/env python3
"""Reproducible latency benchmark for the regular English Chatterbox model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from chatterbox.tts import ChatterboxTTS
import chatterbox.models.t3.t3 as t3_module


PROMPTS = {
    "short": "Hello, this is a short latency test.",
    "medium": (
        "Real-time speech generation should begin quickly, remain clear, and preserve "
        "the speaker's natural rhythm throughout the entire sentence."
    ),
    "long": (
        "Reliable conversational speech synthesis requires more than a fast headline number. "
        "The system must respond quickly, preserve the intended voice, maintain stable pacing, "
        "and continue producing natural audio as the input becomes longer. This benchmark uses "
        "a fixed prompt and sampling seed so that every optimization can be compared against the "
        "same workload without quietly changing model behavior or reducing output quality."
    ),
}


@dataclass
class Sample:
    case: str
    run: int
    seed: int
    e2e_ms: float
    ttft_ms: float
    t3_ms: float
    s3gen_ms: float
    watermark_ms: float
    other_ms: float
    audio_seconds: float
    rtf: float
    generated_tokens: int
    tokens_per_second: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    token_sha256: str


def synchronize() -> None:
    torch.cuda.synchronize()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class TimedModel:
    """Install measurement-only wrappers around the existing model methods."""

    def __init__(self, model: ChatterboxTTS, t3_matmul_precision: str):
        self.model = model
        self.t3_matmul_precision = t3_matmul_precision
        self.metrics: dict[str, Any] = {}
        self._t3_inference = model.t3.inference
        self._s3gen_inference = model.s3gen.inference
        self._apply_watermark = model.watermarker.apply_watermark

        model.t3.inference = self._timed_t3
        model.s3gen.inference = self._timed_s3gen
        model.watermarker.apply_watermark = self._timed_watermark

    def reset(self) -> None:
        self.metrics = {}

    def close(self) -> None:
        self.model.t3.inference = self._t3_inference
        self.model.s3gen.inference = self._s3gen_inference
        self.model.watermarker.apply_watermark = self._apply_watermark

    def _timed_t3(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        original_multinomial = torch.multinomial
        original_matmul_precision = torch.get_float32_matmul_precision()
        ttft_ms: float | None = None

        torch.set_float32_matmul_precision(self.t3_matmul_precision)
        synchronize()
        started = time.perf_counter()

        def timed_multinomial(*sample_args: Any, **sample_kwargs: Any) -> torch.Tensor:
            nonlocal ttft_ms
            result = original_multinomial(*sample_args, **sample_kwargs)
            if ttft_ms is None:
                synchronize()
                ttft_ms = (time.perf_counter() - started) * 1000.0
            return result

        torch.multinomial = timed_multinomial
        try:
            tokens = self._t3_inference(*args, **kwargs)
        finally:
            torch.multinomial = original_multinomial
            torch.set_float32_matmul_precision(original_matmul_precision)

        synchronize()
        self.metrics["t3_ms"] = (time.perf_counter() - started) * 1000.0
        self.metrics["ttft_ms"] = ttft_ms
        self.metrics["tokens"] = tokens.detach()
        return tokens

    def _timed_s3gen(self, *args: Any, **kwargs: Any) -> Any:
        synchronize()
        started = time.perf_counter()
        result = self._s3gen_inference(*args, **kwargs)
        synchronize()
        self.metrics["s3gen_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def _timed_watermark(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self._apply_watermark(*args, **kwargs)
        self.metrics["watermark_ms"] = (time.perf_counter() - started) * 1000.0
        return result


def gpu_info() -> dict[str, str]:
    query = "name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        output = "unavailable"
    return {"nvidia_smi": output}


def summarize(samples: list[Sample]) -> dict[str, dict[str, float]]:
    metrics = [
        "e2e_ms",
        "ttft_ms",
        "t3_ms",
        "s3gen_ms",
        "watermark_ms",
        "other_ms",
        "audio_seconds",
        "rtf",
        "generated_tokens",
        "tokens_per_second",
        "peak_allocated_mib",
        "peak_reserved_mib",
    ]
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [float(getattr(sample, metric)) for sample in samples]
        result[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['label']} benchmark",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- GPU: `{payload['environment']['gpu']['nvidia_smi']}`",
        f"- PyTorch: `{payload['environment']['torch']}`",
        f"- CUDA runtime: `{payload['environment']['torch_cuda']}`",
        f"- Warmups per case: `{payload['configuration']['warmups']}`",
        f"- Timed runs per case: `{payload['configuration']['runs']}`",
        f"- Sampling: `{payload['configuration']['sampling']}`",
        "",
        "| Case | E2E ms | T3 TTFT ms | T3 ms | S3Gen ms | Audio s | RTF | Tokens | Tok/s | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in PROMPTS:
        if case not in payload["cases"]:
            continue
        summary = payload["cases"][case]["summary"]
        value = lambda name: summary[name]["mean"]
        lines.append(
            "| {case} | {e2e:.2f} | {ttft:.2f} | {t3:.2f} | {s3gen:.2f} | "
            "{audio:.3f} | {rtf:.4f} | {tokens:.1f} | {tps:.2f} | {memory:.1f} |".format(
                case=case,
                e2e=value("e2e_ms"),
                ttft=value("ttft_ms"),
                t3=value("t3_ms"),
                s3gen=value("s3gen_ms"),
                audio=value("audio_seconds"),
                rtf=value("rtf"),
                tokens=value("generated_tokens"),
                tps=value("tokens_per_second"),
                memory=value("peak_allocated_mib"),
            )
        )
    lines.extend(
        [
            "",
            "TTFT is measured from entry into `model.t3.inference()` until the first sampled "
            "speech token is materialized. Model loading and warmup are excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/baseline"))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cases", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--t3-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--t3-matmul-precision", choices=("highest", "high"), default="highest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    if args.t3_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support bfloat16")
        model.t3.to(dtype=torch.bfloat16)
        model.conds.t3.to(dtype=torch.bfloat16)

    # The upstream progress bar writes once per generated token and perturbs CPU loop timing.
    t3_module.tqdm = lambda iterable, *unused_args, **unused_kwargs: iterable
    timed_model = TimedModel(model, t3_matmul_precision=args.t3_matmul_precision)

    payload: dict[str, Any] = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "gpu": gpu_info(),
        },
        "configuration": {
            "warmups": args.warmups,
            "runs": args.runs,
            "base_seed": args.seed,
            "t3_dtype": args.t3_dtype,
            "t3_matmul_precision": args.t3_matmul_precision,
            "sampling": {
                "repetition_penalty": 1.2,
                "min_p": 0.05,
                "top_p": 1.0,
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
                "watermark": True,
            },
        },
        "cases": {},
    }

    try:
        for case_index, case in enumerate(args.cases):
            prompt = PROMPTS[case]
            case_seed = args.seed + case_index * 1000
            print(f"\n[{case}] {prompt}")

            for warmup in range(args.warmups):
                torch.manual_seed(case_seed)
                timed_model.reset()
                model.generate(prompt)
                synchronize()
                print(f"  warmup {warmup + 1}/{args.warmups} complete")

            samples: list[Sample] = []
            final_audio: torch.Tensor | None = None
            final_tokens: torch.Tensor | None = None
            for run in range(args.runs):
                torch.manual_seed(case_seed)
                timed_model.reset()
                torch.cuda.reset_peak_memory_stats()
                synchronize()
                started = time.perf_counter()
                audio = model.generate(prompt)
                synchronize()
                e2e_ms = (time.perf_counter() - started) * 1000.0

                metrics = timed_model.metrics
                tokens = metrics["tokens"]
                generated_tokens = int(tokens.shape[-1])
                t3_ms = float(metrics["t3_ms"])
                audio_seconds = float(audio.shape[-1] / model.sr)
                s3gen_ms = float(metrics["s3gen_ms"])
                watermark_ms = float(metrics["watermark_ms"])
                sample = Sample(
                    case=case,
                    run=run + 1,
                    seed=case_seed,
                    e2e_ms=e2e_ms,
                    ttft_ms=float(metrics["ttft_ms"]),
                    t3_ms=t3_ms,
                    s3gen_ms=s3gen_ms,
                    watermark_ms=watermark_ms,
                    other_ms=max(0.0, e2e_ms - t3_ms - s3gen_ms - watermark_ms),
                    audio_seconds=audio_seconds,
                    rtf=e2e_ms / 1000.0 / audio_seconds,
                    generated_tokens=generated_tokens,
                    tokens_per_second=generated_tokens / (t3_ms / 1000.0),
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                    token_sha256=tensor_sha256(tokens),
                )
                samples.append(sample)
                final_audio = audio
                final_tokens = tokens
                print(
                    f"  run {run + 1}/{args.runs}: e2e={sample.e2e_ms:.2f} ms, "
                    f"ttft={sample.ttft_ms:.2f} ms, rtf={sample.rtf:.4f}, "
                    f"tokens/s={sample.tokens_per_second:.2f}"
                )

            assert final_audio is not None and final_tokens is not None
            sf.write(
                args.output_dir / f"{case}.wav",
                np.asarray(final_audio.squeeze(0), dtype=np.float32),
                model.sr,
            )
            torch.save(final_tokens.detach().cpu(), args.output_dir / f"{case}_tokens.pt")
            payload["cases"][case] = {
                "prompt": prompt,
                "samples": [asdict(sample) for sample in samples],
                "summary": summarize(samples),
            }
    finally:
        timed_model.close()

    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    report = markdown_report(payload)
    (args.output_dir / "results.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
