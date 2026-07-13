"""Model-specific GPT-2 decode path for Chatterbox Turbo."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _attention_scale(attention, head_dim: int, layer_index: int) -> float | None:
    scale = None if attention.scale_attn_weights else 1.0
    if attention.scale_attn_by_inverse_layer_idx:
        scale = (head_dim**-0.5 if scale is None else scale) / (layer_index + 1)
    return scale


class TurboGPT2Decoder:
    """Run one cached GPT-2 token through a full-graph decode callable."""

    _CACHE_DTYPES = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        transformer,
        *,
        batch_size: int,
        max_cache_len: int,
        cache_dtype: str = "float32",
        compile_decode: bool = True,
    ):
        if cache_dtype not in self._CACHE_DTYPES:
            choices = ", ".join(sorted(self._CACHE_DTYPES))
            raise ValueError(f"cache_dtype must be one of: {choices}")
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        if transformer.training:
            raise ValueError("TurboGPT2Decoder requires an eval-mode transformer")

        self.transformer = transformer
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.cache_dtype_name = cache_dtype
        self.cache_dtype = self._CACHE_DTYPES[cache_dtype]
        self.compile_decode = compile_decode
        self.num_layers = len(transformer.h)
        self.num_heads = transformer.h[0].attn.num_heads
        self.hidden_size = transformer.embed_dim
        self.head_dim = self.hidden_size // self.num_heads
        device = transformer.wpe.weight.device

        cache_shape = (
            self.num_layers,
            batch_size,
            self.num_heads,
            max_cache_len,
            self.head_dim,
        )
        self.key_cache = torch.empty(cache_shape, dtype=self.cache_dtype, device=device)
        self.value_cache = torch.empty(cache_shape, dtype=self.cache_dtype, device=device)

        self._decode_callable = self._decode
        if compile_decode:
            self._decode_callable = torch.compile(
                self._decode,
                dynamic=True,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )

    def load_cache(self, past_key_values) -> int:
        """Copy an upstream prefill cache into the fixed decode buffers."""
        if len(past_key_values.layers) != self.num_layers:
            raise ValueError("Prefill cache layer count does not match the transformer")

        context_length = past_key_values.layers[0].keys.shape[-2]
        if context_length > self.max_cache_len:
            raise ValueError(
                f"Prefill length {context_length} exceeds cache capacity {self.max_cache_len}"
            )
        for layer_index, layer_cache in enumerate(past_key_values.layers):
            if layer_cache.keys.shape[-2] != context_length:
                raise ValueError("Prefill cache layers have inconsistent sequence lengths")
            self.key_cache[layer_index, :, :, :context_length].copy_(
                layer_cache.keys.to(self.cache_dtype)
            )
            self.value_cache[layer_index, :, :, :context_length].copy_(
                layer_cache.values.to(self.cache_dtype)
            )
        return context_length

    def __call__(self, inputs_embeds: torch.Tensor, cache_position: int) -> torch.Tensor:
        end = cache_position + inputs_embeds.shape[1]
        if end > self.max_cache_len:
            raise ValueError(
                f"KV cache capacity {self.max_cache_len} exceeded by sequence length {end}"
            )
        return self._decode_callable(
            inputs_embeds,
            self.key_cache,
            self.value_cache,
            cache_position,
        )

    def _decode(
        self,
        inputs_embeds: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, width = inputs_embeds.shape
        position_embeds = self.transformer.wpe.weight[
            cache_position : cache_position + sequence_length
        ].unsqueeze(0)
        hidden_states = self.transformer.drop(inputs_embeds + position_embeds)
        end = cache_position + sequence_length

        for layer_index, layer in enumerate(self.transformer.h):
            residual = hidden_states
            hidden_states = layer.ln_1(hidden_states)
            query, key, value = layer.attn.c_attn(hidden_states).split(width, dim=2)
            query = query.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key = key.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            value = value.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

            key_cache[layer_index, :, :, cache_position:end].copy_(
                key.to(self.cache_dtype)
            )
            value_cache[layer_index, :, :, cache_position:end].copy_(
                value.to(self.cache_dtype)
            )
            key_states = key_cache[layer_index, :, :, :end]
            value_states = value_cache[layer_index, :, :, :end]
            attn_output = F.scaled_dot_product_attention(
                query.to(self.cache_dtype),
                key_states,
                value_states,
                dropout_p=0.0,
                is_causal=False,
                scale=_attention_scale(layer.attn, self.head_dim, layer_index),
            )
            attn_output = attn_output.to(hidden_states.dtype)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(
                batch_size, sequence_length, width
            ).contiguous()
            attn_output = layer.attn.resid_dropout(layer.attn.c_proj(attn_output))
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.ln_2(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        return self.transformer.ln_f(hidden_states)


class TurboGPT2DynamicDecoder:
    """Compile GPT-2 decode while keeping each growing K/V tensor contiguous."""

    _CACHE_DTYPES = TurboGPT2Decoder._CACHE_DTYPES

    def __init__(
        self,
        transformer,
        *,
        cache_dtype: str = "bfloat16",
        compile_decode: bool = True,
    ):
        if cache_dtype not in self._CACHE_DTYPES:
            choices = ", ".join(sorted(self._CACHE_DTYPES))
            raise ValueError(f"cache_dtype must be one of: {choices}")
        if transformer.training:
            raise ValueError("TurboGPT2DynamicDecoder requires an eval-mode transformer")

        self.transformer = transformer
        self.cache_dtype_name = cache_dtype
        self.cache_dtype = self._CACHE_DTYPES[cache_dtype]
        self.compile_decode = compile_decode
        self.num_layers = len(transformer.h)
        self.num_heads = transformer.h[0].attn.num_heads
        self.hidden_size = transformer.embed_dim
        self.head_dim = self.hidden_size // self.num_heads
        self.key_cache: tuple[torch.Tensor, ...] = ()
        self.value_cache: tuple[torch.Tensor, ...] = ()

        self._decode_callable = self._decode
        if compile_decode:
            self._decode_callable = torch.compile(
                self._decode,
                dynamic=True,
                fullgraph=True,
                options={"triton.cudagraphs": False},
            )

    def load_cache(self, past_key_values) -> int:
        """Convert an upstream FP32 prefill cache to contiguous decode tensors."""
        if len(past_key_values.layers) != self.num_layers:
            raise ValueError("Prefill cache layer count does not match the transformer")
        context_length = past_key_values.layers[0].keys.shape[-2]
        keys = []
        values = []
        for layer_cache in past_key_values.layers:
            if layer_cache.keys.shape[-2] != context_length:
                raise ValueError("Prefill cache layers have inconsistent sequence lengths")
            keys.append(layer_cache.keys.to(self.cache_dtype).contiguous())
            values.append(layer_cache.values.to(self.cache_dtype).contiguous())
        self.key_cache = tuple(keys)
        self.value_cache = tuple(values)
        return context_length

    def __call__(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if not self.key_cache:
            raise RuntimeError("load_cache() must be called before decode")
        hidden_states, self.key_cache, self.value_cache = self._decode_callable(
            inputs_embeds,
            self.key_cache,
            self.value_cache,
        )
        return hidden_states

    def _decode(
        self,
        inputs_embeds: torch.Tensor,
        key_cache: tuple[torch.Tensor, ...],
        value_cache: tuple[torch.Tensor, ...],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        batch_size, sequence_length, width = inputs_embeds.shape
        cache_position = key_cache[0].shape[-2]
        position_embeds = self.transformer.wpe.weight[
            cache_position : cache_position + sequence_length
        ].unsqueeze(0)
        hidden_states = self.transformer.drop(inputs_embeds + position_embeds)
        next_keys = []
        next_values = []

        for layer_index, layer in enumerate(self.transformer.h):
            residual = hidden_states
            hidden_states = layer.ln_1(hidden_states)
            query, key, value = layer.attn.c_attn(hidden_states).split(width, dim=2)
            query = query.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key = key.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)
            value = value.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

            key_states = torch.cat(
                (key_cache[layer_index], key.to(self.cache_dtype)), dim=-2
            )
            value_states = torch.cat(
                (value_cache[layer_index], value.to(self.cache_dtype)), dim=-2
            )
            next_keys.append(key_states)
            next_values.append(value_states)
            attn_output = F.scaled_dot_product_attention(
                query.to(self.cache_dtype),
                key_states,
                value_states,
                dropout_p=0.0,
                is_causal=False,
                scale=_attention_scale(layer.attn, self.head_dim, layer_index),
            )
            attn_output = attn_output.to(hidden_states.dtype)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(
                batch_size, sequence_length, width
            ).contiguous()
            attn_output = layer.attn.resid_dropout(layer.attn.c_proj(attn_output))
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.ln_2(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        return self.transformer.ln_f(hidden_states), tuple(next_keys), tuple(next_values)
