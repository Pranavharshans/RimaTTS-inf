import torch
import torch.nn.functional as F
from torch import nn


class TurboLogitsProcessor(nn.Module):
    """Apply Turbo's logits processors in the upstream order."""

    def __init__(
        self,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def forward(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temperature > 0 and self.temperature != 1.0:
            scores = scores / self.temperature

        if self.top_k > 0:
            top_k = min(self.top_k, scores.size(-1))
            threshold = torch.topk(scores, top_k)[0][..., -1, None]
            scores = scores.masked_fill(scores < threshold, -float("inf"))

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
            sorted_indices_to_remove[..., -1:] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1,
                sorted_indices,
                sorted_indices_to_remove,
            )
            scores = scores.masked_fill(indices_to_remove, -float("inf"))

        if self.repetition_penalty != 1.0:
            score = torch.gather(scores, 1, input_ids)
            score = torch.where(
                score < 0,
                score * self.repetition_penalty,
                score / self.repetition_penalty,
            )
            scores = scores.scatter(1, input_ids, score)

        return scores, F.softmax(scores, dim=-1)
