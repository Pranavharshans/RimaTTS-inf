#!/usr/bin/env python3
"""Profile the Chatterbox Turbo T3 autoregressive path without S3Gen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm


DEFAULT_TEXT = "Hello, this is a short latency test."
INTERESTING_OPS = (
    "turbo::",
    "scaled_dot_product",
    "flash",
    "efficient_attention",
    "attention",
    "aten::cat",
    "aten::copy",
    "aten::clone",
    "aten::_to_copy",
    "aten::to",
    "aten::contiguous",
    "aten::linear",
    "aten::mm",
    "aten::addmm",
    "aten::matmul",
    "aten::multinomial",
    "aten::softmax",
    "aten::_softmax",
    "aten::topk",
    "aten::sort",
    "aten::scatter",
    "aten::gather",
    "aten::all",
    "aten::item",
    "aten::_local_scalar_dense",
    "cudaDeviceSynchronize",
)


def prepare_text(model: ChatterboxTurboTTS, text: str) -> torch.Tensor:
    tokenized = model.tokenizer(
        punc_norm(text), return_tensors="pt", padding=True, truncation=True
    )
    return tokenized.input_ids.to(model.device)


def tensor_info(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
        "nbytes": tensor.numel() * tensor.element_size(),
    }


def cache_info(cache: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(cache).__name__}
    try:
        result["length"] = len(cache)
    except TypeError:
        result["length"] = None

    layers = getattr(cache, "layers", None)
    if layers is None:
        try:
            layers = list(cache)
        except TypeError:
            layers = []
    result["layer_count"] = len(layers)
    if not layers:
        return result

    layer = layers[0]
    result["layer_type"] = type(layer).__name__
    if isinstance(layer, (tuple, list)):
        key = layer[0] if len(layer) > 0 else None
        value = layer[1] if len(layer) > 1 else None
    else:
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if key is None:
            key = getattr(layer, "key_cache", None)
        if value is None:
            value = getattr(layer, "value_cache", None)
    result["key"] = tensor_info(key)
    result["value"] = tensor_info(value)
    return result


def dtype_inventory(module: torch.nn.Module) -> dict[str, Any]:
    parameters = Counter(str(parameter.dtype) for parameter in module.parameters())
    buffers = Counter(str(buffer.dtype) for buffer in module.buffers())
    return {
        "parameters": dict(parameters),
        "buffers": dict(buffers),
        "parameter_mib": sum(p.numel() * p.element_size() for p in module.parameters()) / 2**20,
    }


def event_value(event: Any, name: str) -> float:
    value = getattr(event, name, 0.0)
    return float(value) if value is not None else 0.0


def patch_method(
    stack: ExitStack,
    owner: Any,
    name: str,
    label: str,
    callback: Callable[[tuple[Any, ...], dict[str, Any], Any], None] | None = None,
) -> None:
    original = getattr(owner, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with record_function(label):
            result = original(*args, **kwargs)
        if callback is not None:
            callback(args, kwargs, result)
        return result

    setattr(owner, name, wrapped)
    stack.callback(setattr, owner, name, original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/turbo_profile_baseline")
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    text_tokens = prepare_text(model, args.text)
    cache_snapshots: list[dict[str, Any]] = []
    final_cache: dict[str, Any] | None = None
    forward_calls = 0

    def capture_cache(
        call_args: tuple[Any, ...], call_kwargs: dict[str, Any], output: Any
    ) -> None:
        nonlocal final_cache, forward_calls
        forward_calls += 1
        inputs = call_kwargs.get("inputs_embeds")
        past = call_kwargs.get("past_key_values")
        snapshot = {
            "forward_call": forward_calls,
            "input_sequence_length": int(inputs.shape[1]) if inputs is not None else None,
            "input_past_type": type(past).__name__ if past is not None else None,
            "output_cache": cache_info(output.past_key_values),
        }
        if len(cache_snapshots) < 4:
            cache_snapshots.append(snapshot)
        final_cache = snapshot

    def run_t3() -> torch.Tensor:
        return model.t3.inference_turbo(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            temperature=0.8,
            top_k=1000,
            top_p=0.95,
            repetition_penalty=1.2,
            max_gen_len=1000,
        )

    original_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(args.matmul_precision)
    try:
        for _ in range(args.warmups):
            torch.manual_seed(args.seed)
            run_t3()
            torch.cuda.synchronize()

        with ExitStack() as patches:
            patch_method(
                patches,
                model.t3,
                "prepare_input_embeds",
                "turbo::prepare_input_embeds",
            )
            patch_method(
                patches,
                model.t3.tfmr,
                "forward",
                "turbo::transformer",
                capture_cache,
            )
            patch_method(patches, model.t3.speech_emb, "forward", "turbo::speech_embedding")
            patch_method(patches, model.t3.speech_head, "forward", "turbo::speech_head")

            processor_classes = (
                (TemperatureLogitsWarper, "turbo::temperature"),
                (TopKLogitsWarper, "turbo::top_k"),
                (TopPLogitsWarper, "turbo::top_p"),
                (RepetitionPenaltyLogitsProcessor, "turbo::repetition_penalty"),
            )
            for processor_class, label in processor_classes:
                patch_method(patches, processor_class, "__call__", label)

            cache_snapshots.clear()
            final_cache = None
            forward_calls = 0
            torch.manual_seed(args.seed)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=args.profile_memory,
                with_stack=False,
            ) as profiler:
                tokens = run_t3()
                torch.cuda.synchronize()
    finally:
        torch.set_float32_matmul_precision(original_matmul_precision)

    events = profiler.key_averages()
    selected = []
    for event in events:
        if any(fragment in event.key for fragment in INTERESTING_OPS):
            selected.append(
                {
                    "name": event.key,
                    "calls": int(event.count),
                    "self_cuda_ms": event_value(event, "self_device_time_total") / 1000.0,
                    "cuda_total_ms": event_value(event, "device_time_total") / 1000.0,
                    "self_cpu_ms": event_value(event, "self_cpu_time_total") / 1000.0,
                    "cpu_total_ms": event_value(event, "cpu_time_total") / 1000.0,
                    "self_cuda_memory_mib": event_value(
                        event, "self_device_memory_usage"
                    )
                    / 2**20,
                }
            )
    selected.sort(key=lambda item: item["self_cuda_ms"], reverse=True)

    config = model.t3.tfmr.config
    payload = {
        "generated_tokens": int(tokens.shape[-1]),
        "configuration": {
            "warmups": args.warmups,
            "matmul_precision": args.matmul_precision,
            "text": args.text,
        },
        "token_sha256": hashlib.sha256(
            tokens.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
        ).hexdigest(),
        "attention_implementation": {
            "config": getattr(config, "_attn_implementation", None),
            "internal": getattr(config, "_attn_implementation_internal", None),
        },
        "sdp_backends": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "t3_dtype_inventory": dtype_inventory(model.t3),
        "cache_snapshots": cache_snapshots,
        "final_cache": final_cache,
        "interesting_operators": selected,
    }
    (args.output_dir / "profile.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
    (args.output_dir / "profile.txt").write_text(table, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("\n" + table)


if __name__ == "__main__":
    main()
