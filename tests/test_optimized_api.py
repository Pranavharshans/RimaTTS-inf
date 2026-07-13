import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from chatterbox.tts import ChatterboxTTS


class OptimizedApiTest(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(ChatterboxTTS)
        self.model.device = "cpu"
        self.model.sr = 24_000
        self.model.conds = SimpleNamespace(
            t3=SimpleNamespace(emotion_adv=torch.full((1, 1, 1), 0.5)),
            gen={},
        )
        self.model.tokenizer = SimpleNamespace(
            text_to_tokens=Mock(return_value=torch.tensor([[10, 11]], dtype=torch.long))
        )
        self.model.t3 = SimpleNamespace(
            hp=SimpleNamespace(start_text_token=1, stop_text_token=2),
            inference=Mock(return_value=torch.tensor([[100, 101]], dtype=torch.long)),
        )
        self.model.s3gen = SimpleNamespace(
            inference=Mock(return_value=(torch.zeros((1, 1, 16)), None))
        )
        self.model.watermarker = SimpleNamespace(
            apply_watermark=Mock(return_value=np.zeros(16, dtype=np.float32))
        )

    def test_optimized_options_reach_t3_and_precision_is_restored(self):
        original_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            wav = self.model.generate(
                "public API test",
                compile_t3_decode=True,
                t3_compile_mode="reduce-overhead",
                t3_matmul_precision="high",
                show_progress=False,
            )

            kwargs = self.model.t3.inference.call_args.kwargs
            self.assertTrue(kwargs["compile_decode"])
            self.assertEqual(kwargs["compile_mode"], "reduce-overhead")
            self.assertIsNone(kwargs["tf32_after_tokens"])
            self.assertFalse(kwargs["show_progress"])
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertEqual(tuple(wav.shape), (1, 16))
        finally:
            torch.set_float32_matmul_precision(original_precision)

    def test_invalid_matmul_precision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported t3_matmul_precision"):
            self.model.generate("public API test", t3_matmul_precision="invalid")


if __name__ == "__main__":
    unittest.main()
