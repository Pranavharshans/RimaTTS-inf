"""T3-specific KV cache implementations."""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import Cache, StaticLayer


class PreallocatedDynamicLayer(StaticLayer):
    """Fixed backing storage with a dynamic populated-prefix view."""

    is_compileable = False

    def __init__(self, max_cache_len: int):
        super().__init__(max_cache_len=max_cache_len)
        self.current_length = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        super().lazy_initialization(key_states, value_states)
        self.current_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        update_length = key_states.shape[-2]
        end = self.current_length + update_length
        if end > self.max_cache_len:
            raise ValueError(
                f"KV cache capacity exceeded: requested {end}, capacity {self.max_cache_len}"
            )

        self.keys[:, :, self.current_length:end, :].copy_(key_states)
        self.values[:, :, self.current_length:end, :].copy_(value_states)
        self.current_length = end
        return (
            self.keys[:, :, : self.current_length, :],
            self.values[:, :, : self.current_length, :],
        )

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.current_length + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        return self.current_length

    def get_max_cache_shape(self) -> int:
        # Attention sees only the populated prefix, as with DynamicLayer.
        return -1

    def reset(self) -> None:
        super().reset()
        self.current_length = 0


class PreallocatedDynamicCache(Cache):
    """A full-model cache made from preallocated dynamic-prefix layers."""

    def __init__(self, config: Any, max_cache_len: int):
        config = config.get_text_config(decoder=True)
        if getattr(config, "sliding_window", None) is not None:
            raise ValueError("PreallocatedDynamicCache does not support sliding-window attention")
        layers = [
            PreallocatedDynamicLayer(max_cache_len=max_cache_len)
            for _ in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, None
