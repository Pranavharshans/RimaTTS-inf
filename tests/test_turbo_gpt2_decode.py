import unittest

import torch
from transformers import GPT2Config, GPT2Model

from chatterbox.models.t3.turbo_gpt2_decode import TurboGPT2Decoder


class TurboGPT2DecoderTest(unittest.TestCase):
    def test_eager_fp32_decode_matches_transformers(self):
        torch.manual_seed(7)
        config = GPT2Config(
            vocab_size=16,
            n_positions=16,
            n_embd=16,
            n_layer=2,
            n_head=2,
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_implementation="sdpa",
        )
        transformer = GPT2Model(config).eval()
        prefill_embeds = torch.randn(1, 3, config.n_embd)
        next_embed = torch.randn(1, 1, config.n_embd)

        prefill = transformer(inputs_embeds=prefill_embeds, use_cache=True)
        decoder = TurboGPT2Decoder(
            transformer,
            batch_size=1,
            max_cache_len=8,
            cache_dtype="float32",
            compile_decode=False,
        )
        cache_position = decoder.load_cache(prefill.past_key_values)
        expected = transformer(
            inputs_embeds=next_embed,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        ).last_hidden_state
        actual = decoder(next_embed, cache_position)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
