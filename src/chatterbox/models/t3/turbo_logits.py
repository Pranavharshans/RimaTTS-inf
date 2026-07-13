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
        compact_topk_topp: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.compact_topk_topp = compact_topk_topp

    def _standard_topk_topp(self, scores: torch.Tensor) -> torch.Tensor:
        top_k = min(self.top_k, scores.size(-1))
        threshold = torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(scores < threshold, -float("inf"))

        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        sorted_indices_to_remove[..., -1:] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            1,
            sorted_indices,
            sorted_indices_to_remove,
        )
        return scores.masked_fill(indices_to_remove, -float("inf"))

    def _compact_topk_topp(
        self,
        candidate_scores: torch.Tensor,
        candidate_indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        top_k = min(self.top_k, scores.size(-1))
        top_scores = candidate_scores[..., :top_k]
        top_indices = candidate_indices[..., :top_k]
        ascending_scores = top_scores.flip(-1)
        padding = scores.new_full(
            (*scores.shape[:-1], scores.size(-1) - top_k),
            -float("inf"),
        )
        sorted_logits = torch.cat((padding, ascending_scores), dim=-1)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove_ascending = cumulative_probs[..., -top_k:] <= (1 - self.top_p)
        remove_ascending[..., -1:] = False
        filtered_top_scores = top_scores.masked_fill(
            remove_ascending.flip(-1),
            -float("inf"),
        )
        return torch.full_like(scores, -float("inf")).scatter(
            1,
            top_indices,
            filtered_top_scores,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temperature > 0 and self.temperature != 1.0:
            scores = scores / self.temperature

        use_compact_topk_topp = (
            self.compact_topk_topp and self.top_k > 0 and self.top_p < 1.0
        )
        if use_compact_topk_topp:
            top_k = min(self.top_k, scores.size(-1))
            candidate_count = min(top_k + 1, scores.size(-1))
            candidate_scores, candidate_indices = torch.topk(
                scores,
                candidate_count,
            )
            has_relevant_tie = torch.any(
                candidate_scores[..., 1:] == candidate_scores[..., :-1]
            )
            scores = torch.cond(
                has_relevant_tie,
                lambda _, __, original_scores: self._standard_topk_topp(
                    original_scores
                ),
                self._compact_topk_topp,
                (candidate_scores, candidate_indices, scores),
            )
        elif self.top_k > 0 and self.top_p < 1.0:
            scores = self._standard_topk_topp(scores)
        elif self.top_k > 0:
            top_k = min(self.top_k, scores.size(-1))
            threshold = torch.topk(scores, top_k)[0][..., -1, None]
            scores = scores.masked_fill(scores < threshold, -float("inf"))

        if self.top_p < 1.0 and self.top_k <= 0:
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
