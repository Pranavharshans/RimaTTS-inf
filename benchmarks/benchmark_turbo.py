#!/usr/bin/env python3
"""Reproducible latency and exact-output benchmark for Chatterbox Turbo."""

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

from chatterbox.tts_turbo import ChatterboxTurboTTS


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

SAMPLING = {
    "repetition_penalty": 1.2,
    "min_p": 0.0,
    "top_p": 0.95,
    "exaggeration": 0.0,
    "cfg_weight": 0.0,
    "temperature": 0.8,
    "top_k": 1000,
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
    audio_sha256: str


def synchronize() -> None:
    torch.cuda.synchronize()


def tensor_sha256(tensor: torch.Tensor, *, dtype: torch.dtype) -> str:
    array = tensor.detach().to(device="cpu", dtype=dtype).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class TimedTurboModel:
    """Install measurement-only wrappers around Turbo's public inference path."""

    def __init__(self, model: ChatterboxTurboTTS, *, t3_matmul_precision: str):
        self.model = model
        self.t3_matmul_precision = t3_matmul_precision
        self.metrics: dict[str, Any] = {}
        self._t3_inference = model.t3.inference_turbo
        self._s3gen_inference = model.s3gen.inference
        self._apply_watermark = model.watermarker.apply_watermark

        model.t3.inference_turbo = self._timed_t3
        model.s3gen.inference = self._timed_s3gen
        model.watermarker.apply_watermark = self._timed_watermark

    def reset(self) -> None:
        self.metrics = {}

    def close(self) -> None:
        self.model.t3.inference_turbo = self._t3_inference
        self.model.s3gen.inference = self._s3gen_inference
        self.model.watermarker.apply_watermark = self._apply_watermark

    def _timed_t3(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        original_multinomial = torch.multinomial
        ttft_ms: float | None = None

        synchronize()
        started = time.perf_counter()

        def timed_multinomial(*sample_args: Any, **sample_kwargs: Any) -> torch.Tensor:
            nonlocal ttft_ms
            result = original_multinomial(*sample_args, **sample_kwargs)
            if ttft_ms is None:
                synchronize()
                ttft_ms = (time.perf_counter() - started) * 1000.0
            return result

        original_matmul_precision = torch.get_float32_matmul_precision()
        torch.multinomial = timed_multinomial
        try:
            torch.set_float32_matmul_precision(self.t3_matmul_precision)
            tokens = self._t3_inference(*args, **kwargs)
        finally:
            torch.set_float32_matmul_precision(original_matmul_precision)
            torch.multinomial = original_multinomial

        synchronize()
        if ttft_ms is None:
            raise RuntimeError("Turbo T3 returned without sampling a speech token")
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


def compare_reference(
    reference_dir: Path | None,
    case: str,
    tokens: torch.Tensor,
    audio: torch.Tensor,
) -> dict[str, Any]:
    if reference_dir is None:
        return {"checked": False}

    token_path = reference_dir / f"{case}_tokens.pt"
    audio_path = reference_dir / f"{case}_audio.pt"
    if not token_path.exists() or not audio_path.exists():
        raise FileNotFoundError(
            f"Missing exact-output reference for {case}: {token_path} or {audio_path}"
        )

    expected_tokens = torch.load(token_path, map_location="cpu", weights_only=True)
    expected_audio = torch.load(audio_path, map_location="cpu", weights_only=True)
    actual_tokens = tokens.detach().cpu()
    actual_audio = audio.detach().cpu()
    token_equal = torch.equal(actual_tokens, expected_tokens)
    audio_equal = torch.equal(actual_audio, expected_audio)
    max_audio_abs_diff = (
        float((actual_audio - expected_audio).abs().max())
        if actual_audio.shape == expected_audio.shape
        else None
    )
    return {
        "checked": True,
        "token_equal": token_equal,
        "audio_equal": audio_equal,
        "expected_token_shape": list(expected_tokens.shape),
        "actual_token_shape": list(actual_tokens.shape),
        "expected_audio_shape": list(expected_audio.shape),
        "actual_audio_shape": list(actual_audio.shape),
        "max_audio_abs_diff": max_audio_abs_diff,
        "passed": token_equal and audio_equal,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['label']} Turbo benchmark",
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

    lines.extend(["", "## Exact-output gate", ""])
    for case in PROMPTS:
        if case not in payload["cases"]:
            continue
        case_payload = payload["cases"][case]
        hashes = case_payload["determinism"]
        quality = case_payload["reference_comparison"]
        line = (
            f"- {case}: timed runs deterministic=`{hashes['passed']}`; "
            f"token hash=`{hashes['token_sha256']}`; audio hash=`{hashes['audio_sha256']}`"
        )
        if quality["checked"]:
            line += (
                f"; reference token exact=`{quality['token_equal']}`; "
                f"reference audio exact=`{quality['audio_equal']}`; "
                f"max abs audio diff=`{quality['max_audio_abs_diff']}`"
            )
        lines.append(line)

    lines.extend(
        [
            "",
            "T3 TTFT is measured from entry into `model.t3.inference_turbo()` until the first "
            "sampled speech token is materialized. It is not time to first streamed audio. Model "
            "loading and warmup are excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/turbo_baseline"))
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--require-exact-reference", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cases", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--t3-matmul-precision", choices=("highest", "high"), default="highest"
    )
    parser.add_argument("--optimize-t3-loop", action="store_true")
    parser.add_argument("--optimize-t3-sync", action="store_true")
    parser.add_argument("--preallocate-t3-kv", action="store_true")
    parser.add_argument("--custom-t3-decode", action="store_true")
    parser.add_argument(
        "--custom-t3-cache-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--custom-t3-eager", action="store_true")
    parser.add_argument("--compile-native-t3-decode", action="store_true")
    parser.add_argument(
        "--native-t3-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--native-t3-cudagraph-until", type=int)
    parser.add_argument("--compile-native-t3-step", action="store_true")
    parser.add_argument("--dynamic-t3-decode", action="store_true")
    parser.add_argument(
        "--dynamic-t3-cache-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--dynamic-t3-eager", action="store_true")
    parser.add_argument("--hybrid-t3-after-tokens", type=int)
    parser.add_argument("--compile-t3-logits", action="store_true")
    parser.add_argument("--compact-t3-logits", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.require_exact_reference and args.reference_dir is None:
        raise ValueError("--require-exact-reference requires --reference-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    timed_model = TimedTurboModel(model, t3_matmul_precision=args.t3_matmul_precision)
    generation_kwargs = {
        **SAMPLING,
        "t3_optimize_loop": args.optimize_t3_loop,
        "t3_optimize_sync": args.optimize_t3_sync,
        "t3_preallocate_kv": args.preallocate_t3_kv,
        "t3_custom_decode": args.custom_t3_decode,
        "t3_custom_cache_dtype": args.custom_t3_cache_dtype,
        "t3_custom_compile": not args.custom_t3_eager,
        "t3_compile_native_decode": args.compile_native_t3_decode,
        "t3_native_compile_mode": args.native_t3_compile_mode,
        "t3_native_cudagraph_until": args.native_t3_cudagraph_until,
        "t3_compile_native_step": args.compile_native_t3_step,
        "t3_dynamic_decode": args.dynamic_t3_decode,
        "t3_dynamic_cache_dtype": args.dynamic_t3_cache_dtype,
        "t3_dynamic_compile": not args.dynamic_t3_eager,
        "t3_hybrid_decode_after": args.hybrid_t3_after_tokens,
        "t3_compile_logits": args.compile_t3_logits,
        "t3_compact_logits": args.compact_t3_logits,
        "show_progress": not args.hide_progress,
    }

    payload: dict[str, Any] = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "ResembleAI/chatterbox-turbo",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "gpu": gpu_info(),
        },
        "configuration": {
            "warmups": args.warmups,
            "runs": args.runs,
            "base_seed": args.seed,
            "t3_matmul_precision": args.t3_matmul_precision,
            "optimize_t3_loop": args.optimize_t3_loop,
            "optimize_t3_sync": args.optimize_t3_sync,
            "preallocate_t3_kv": args.preallocate_t3_kv,
            "custom_t3_decode": args.custom_t3_decode,
            "custom_t3_cache_dtype": args.custom_t3_cache_dtype,
            "custom_t3_compile": not args.custom_t3_eager,
            "compile_native_t3_decode": args.compile_native_t3_decode,
            "native_t3_compile_mode": args.native_t3_compile_mode,
            "native_t3_cudagraph_until": args.native_t3_cudagraph_until,
            "compile_native_t3_step": args.compile_native_t3_step,
            "dynamic_t3_decode": args.dynamic_t3_decode,
            "dynamic_t3_cache_dtype": args.dynamic_t3_cache_dtype,
            "dynamic_t3_compile": not args.dynamic_t3_eager,
            "hybrid_t3_after_tokens": args.hybrid_t3_after_tokens,
            "compile_t3_logits": args.compile_t3_logits,
            "compact_t3_logits": args.compact_t3_logits,
            "show_progress": not args.hide_progress,
            "sampling": {**SAMPLING, "watermark": True, "s3gen_cfm_steps": 2},
            "conditioning": "checkpoint built-in voice",
            "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        },
        "cases": {},
    }

    exact_reference_passed = True
    try:
        for case in args.cases:
            prompt = PROMPTS[case]
            case_seed = args.seed + tuple(PROMPTS).index(case) * 1000
            print(f"\n[{case}] {prompt}")

            for warmup in range(args.warmups):
                torch.manual_seed(case_seed)
                timed_model.reset()
                model.generate(prompt, **generation_kwargs)
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
                audio = model.generate(prompt, **generation_kwargs)
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
                    token_sha256=tensor_sha256(tokens, dtype=torch.int64),
                    audio_sha256=tensor_sha256(audio, dtype=torch.float32),
                )
                samples.append(sample)
                final_audio = audio.detach().cpu().clone()
                final_tokens = tokens.detach().cpu().clone()
                print(
                    f"  run {run + 1}/{args.runs}: e2e={sample.e2e_ms:.2f} ms, "
                    f"ttft={sample.ttft_ms:.2f} ms, rtf={sample.rtf:.4f}, "
                    f"tokens/s={sample.tokens_per_second:.2f}"
                )

            assert final_audio is not None and final_tokens is not None
            token_hashes = {sample.token_sha256 for sample in samples}
            audio_hashes = {sample.audio_sha256 for sample in samples}
            deterministic = len(token_hashes) == 1 and len(audio_hashes) == 1
            reference_comparison = compare_reference(
                args.reference_dir, case, final_tokens, final_audio
            )
            if reference_comparison["checked"]:
                exact_reference_passed &= bool(reference_comparison["passed"])

            sf.write(
                args.output_dir / f"{case}.wav",
                np.asarray(final_audio.squeeze(0), dtype=np.float32),
                model.sr,
            )
            torch.save(final_tokens, args.output_dir / f"{case}_tokens.pt")
            torch.save(final_audio, args.output_dir / f"{case}_audio.pt")
            payload["cases"][case] = {
                "prompt": prompt,
                "samples": [asdict(sample) for sample in samples],
                "summary": summarize(samples),
                "determinism": {
                    "passed": deterministic,
                    "token_sha256": sorted(token_hashes),
                    "audio_sha256": sorted(audio_hashes),
                },
                "reference_comparison": reference_comparison,
            }
    finally:
        timed_model.close()

    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    report = markdown_report(payload)
    (args.output_dir / "results.md").write_text(report, encoding="utf-8")
    print("\n" + report)

    if args.require_exact_reference and not exact_reference_passed:
        raise RuntimeError("One or more Turbo outputs differ from the exact baseline reference")


if __name__ == "__main__":
    main()
