#!/usr/bin/env python3
"""Profile the regular Chatterbox T3 autoregressive path without S3Gen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from chatterbox.tts import ChatterboxTTS, punc_norm
import chatterbox.models.t3.t3 as t3_module


DEFAULT_TEXT = "Hello, this is a short latency test."
INTERESTING_OPS = (
    "scaled_dot_product",
    "flash",
    "attention",
    "aten::cat",
    "aten::copy",
    "aten::_to_copy",
    "aten::to",
    "aten::contiguous",
    "aten::linear",
    "aten::mm",
    "aten::addmm",
    "aten::matmul",
    "aten::multinomial",
    "aten::softmax",
)


def prepare_text(model: ChatterboxTTS, text: str) -> torch.Tensor:
    tokens = model.tokenizer.text_to_tokens(punc_norm(text)).to(model.device)
    tokens = torch.cat([tokens, tokens], dim=0)
    tokens = F.pad(tokens, (1, 0), value=model.t3.hp.start_text_token)
    return F.pad(tokens, (0, 1), value=model.t3.hp.stop_text_token)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/profile_baseline"))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--profile-memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    text_tokens = prepare_text(model, args.text)
    t3_module.tqdm = lambda iterable, *unused_args, **unused_kwargs: iterable

    cache_snapshots: list[dict[str, Any]] = []
    original_forward = model.t3.tfmr.forward
    forward_calls = 0

    def capture_forward(*forward_args: Any, **forward_kwargs: Any) -> Any:
        nonlocal forward_calls
        forward_calls += 1
        inputs = forward_kwargs.get("inputs_embeds")
        past = forward_kwargs.get("past_key_values")
        output = original_forward(*forward_args, **forward_kwargs)
        if len(cache_snapshots) < 4:
            cache_snapshots.append(
                {
                    "forward_call": forward_calls,
                    "input_sequence_length": int(inputs.shape[1]) if inputs is not None else None,
                    "input_past_type": type(past).__name__ if past is not None else None,
                    "output_cache": cache_info(output.past_key_values),
                }
            )
        return output

    model.t3.tfmr.forward = capture_forward
    try:
        torch.manual_seed(args.seed)
        model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=0.8,
            top_p=1.0,
            min_p=0.05,
            repetition_penalty=1.2,
            cfg_weight=0.5,
        )
        torch.cuda.synchronize()

        cache_snapshots.clear()
        forward_calls = 0
        torch.manual_seed(args.seed)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=args.profile_memory,
            with_stack=False,
        ) as profiler:
            tokens = model.t3.inference(
                t3_cond=model.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=0.8,
                top_p=1.0,
                min_p=0.05,
                repetition_penalty=1.2,
                cfg_weight=0.5,
            )
            torch.cuda.synchronize()
    finally:
        model.t3.tfmr.forward = original_forward

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
                    "self_cuda_memory_mib": event_value(event, "self_device_memory_usage") / 2**20,
                }
            )
    selected.sort(key=lambda item: item["self_cuda_ms"], reverse=True)

    payload = {
        "generated_tokens": int(tokens.shape[-1]),
        "token_sha256": hashlib.sha256(
            tokens.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
        ).hexdigest(),
        "sdp_backends": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "t3_dtype_inventory": dtype_inventory(model.t3),
        "cache_snapshots": cache_snapshots,
        "interesting_operators": selected,
    }
    (args.output_dir / "profile.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
    (args.output_dir / "profile.txt").write_text(table, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("\n" + table)


if __name__ == "__main__":
    main()
