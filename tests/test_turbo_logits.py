import unittest

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from chatterbox.models.t3.turbo_logits import TurboLogitsProcessor


class TurboLogitsProcessorTest(unittest.TestCase):
    def test_matches_transformers_default_order_exactly(self):
        input_ids = torch.tensor([[4, 9, 4, 17, 31]], dtype=torch.long)
        scores = torch.linspace(-4.0, 4.0, 64).unsqueeze(0)
        scores[0, 4] = -0.75
        scores[0, 17] = 1.25

        reference = LogitsProcessorList(
            [
                TemperatureLogitsWarper(0.8),
                TopKLogitsWarper(20),
                TopPLogitsWarper(0.95),
                RepetitionPenaltyLogitsProcessor(1.2),
            ]
        )(input_ids, scores.clone())
        expected_probs = F.softmax(reference, dim=-1)

        actual, actual_probs = TurboLogitsProcessor(
            temperature=0.8,
            top_k=20,
            top_p=0.95,
            repetition_penalty=1.2,
        )(input_ids, scores.clone())

        self.assertTrue(torch.equal(actual, reference))
        self.assertTrue(torch.equal(actual_probs, expected_probs))

        compact, compact_probs = TurboLogitsProcessor(
            temperature=0.8,
            top_k=20,
            top_p=0.95,
            repetition_penalty=1.2,
            compact_topk_topp=True,
        )(input_ids, scores.clone())

        self.assertTrue(torch.equal(compact, reference))
        self.assertTrue(torch.equal(compact_probs, expected_probs))


if __name__ == "__main__":
    unittest.main()
