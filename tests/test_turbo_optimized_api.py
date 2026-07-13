import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS


class TurboOptimizedApiTest(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(ChatterboxTurboTTS)
        self.model.device = "cpu"
        self.model.sr = 24_000
        self.model.conds = SimpleNamespace(t3=SimpleNamespace(), gen={})
        tokenized = SimpleNamespace(input_ids=torch.tensor([[10, 11]], dtype=torch.long))
        self.model.tokenizer = Mock(return_value=tokenized)
        self.model.t3 = SimpleNamespace(
            inference_turbo=Mock(return_value=torch.tensor([[100, 101]], dtype=torch.long))
        )
        self.model.s3gen = SimpleNamespace(
            inference=Mock(return_value=(torch.zeros((1, 16)), None))
        )
        self.model.watermarker = SimpleNamespace(
            apply_watermark=Mock(return_value=np.zeros(16, dtype=np.float32))
        )

    def test_loop_options_reach_turbo_t3(self):
        wav = self.model.generate(
            "public Turbo API test",
            t3_optimize_loop=True,
            t3_optimize_sync=True,
            show_progress=False,
        )

        kwargs = self.model.t3.inference_turbo.call_args.kwargs
        self.assertTrue(kwargs["optimize_loop"])
        self.assertTrue(kwargs["optimize_sync"])
        self.assertFalse(kwargs["show_progress"])
        self.assertEqual(tuple(wav.shape), (1, 16))


if __name__ == "__main__":
    unittest.main()
