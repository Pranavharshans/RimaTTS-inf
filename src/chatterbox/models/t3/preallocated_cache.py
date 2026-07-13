"""A fixed-capacity KV cache with DynamicCache-compatible attention lengths."""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class PreallocatedDynamicLayer(CacheLayerMixin):
    """Append K/V states in place while returning only the populated prefix."""

    is_compileable = False
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        self.max_cache_len = max_cache_len
        self.cumulative_length = 0

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.max_batch_size, self.num_heads = key_states.shape[:2]
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]
        self.keys = torch.empty(
            self.max_batch_size,
            self.num_heads,
            self.max_cache_len,
            self.k_head_dim,
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.empty(
            self.max_batch_size,
            self.num_heads,
            self.max_cache_len,
            self.v_head_dim,
            dtype=value_states.dtype,
            device=value_states.device,
        )
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        update_length = key_states.shape[-2]
        end = self.cumulative_length + update_length
        if end > self.max_cache_len:
            raise ValueError(
                f"KV cache capacity {self.max_cache_len} exceeded by sequence length {end}"
            )

        self.keys[:, :, self.cumulative_length:end].copy_(key_states)
        self.values[:, :, self.cumulative_length:end].copy_(value_states)
        self.cumulative_length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.cumulative_length + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self.cumulative_length = 0


class PreallocatedDynamicCache(Cache):
    """Preallocate all transformer cache layers without padding attention."""

    def __init__(self, config, max_cache_len: int):
        decoder_config = config.get_text_config(decoder=True)
        layers = [
            PreallocatedDynamicLayer(max_cache_len)
            for _ in range(decoder_config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
