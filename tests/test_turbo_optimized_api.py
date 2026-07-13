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
            t3_preallocate_kv=True,
            t3_custom_decode=True,
            t3_custom_cache_dtype="bfloat16",
            t3_custom_compile=False,
            t3_dynamic_decode=False,
            show_progress=False,
        )

        kwargs = self.model.t3.inference_turbo.call_args.kwargs
        self.assertTrue(kwargs["optimize_loop"])
        self.assertTrue(kwargs["optimize_sync"])
        self.assertTrue(kwargs["preallocate_kv"])
        self.assertTrue(kwargs["custom_decode"])
        self.assertEqual(kwargs["custom_cache_dtype"], "bfloat16")
        self.assertFalse(kwargs["custom_compile"])
        self.assertFalse(kwargs["compile_native_decode"])
        self.assertFalse(kwargs["dynamic_decode"])
        self.assertFalse(kwargs["show_progress"])
        self.assertEqual(tuple(wav.shape), (1, 16))

    def test_native_compile_option_reaches_turbo_t3(self):
        self.model.generate(
            "native compile API test",
            t3_compile_native_decode=True,
            t3_native_compile_mode="max-autotune-no-cudagraphs",
            t3_compile_logits=True,
        )

        kwargs = self.model.t3.inference_turbo.call_args.kwargs
        self.assertTrue(kwargs["compile_native_decode"])
        self.assertEqual(
            kwargs["native_compile_mode"],
            "max-autotune-no-cudagraphs",
        )
        self.assertTrue(kwargs["compile_logits"])
        self.assertFalse(kwargs["custom_decode"])

    def test_native_step_option_reaches_turbo_t3(self):
        self.model.generate(
            "native step API test",
            t3_compile_native_step=True,
        )

        kwargs = self.model.t3.inference_turbo.call_args.kwargs
        self.assertTrue(kwargs["compile_native_step"])
        self.assertFalse(kwargs["compile_native_decode"])
        self.assertFalse(kwargs["compile_logits"])

    def test_dynamic_decode_options_reach_turbo_t3(self):
        self.model.generate(
            "dynamic decode API test",
            t3_dynamic_decode=True,
            t3_dynamic_cache_dtype="float32",
            t3_dynamic_compile=False,
            t3_hybrid_decode_after=None,
        )

        kwargs = self.model.t3.inference_turbo.call_args.kwargs
        self.assertTrue(kwargs["dynamic_decode"])
        self.assertEqual(kwargs["dynamic_cache_dtype"], "float32")
        self.assertFalse(kwargs["dynamic_compile"])
        self.assertFalse(kwargs["custom_decode"])
        self.assertFalse(kwargs["compile_native_decode"])
        self.assertIsNone(kwargs["hybrid_decode_after"])

    def test_hybrid_decode_options_reach_turbo_t3(self):
        self.model.generate(
            "hybrid decode API test",
            t3_hybrid_decode_after=192,
        )

        kwargs = self.model.t3.inference_turbo.call_args.kwargs
        self.assertEqual(kwargs["hybrid_decode_after"], 192)
        self.assertFalse(kwargs["dynamic_decode"])
        self.assertFalse(kwargs["custom_decode"])
        self.assertFalse(kwargs["compile_native_decode"])


if __name__ == "__main__":
    unittest.main()
