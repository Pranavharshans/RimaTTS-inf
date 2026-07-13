# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List

logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .preallocated_cache import PreallocatedDynamicCache
from .turbo_gpt2_decode import TurboGPT2Decoder, TurboGPT2DynamicDecoder
from .turbo_logits import TurboLogitsProcessor
from ..utils import AttrDict


logger = logging.getLogger(__name__)


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp

        config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"

        if self.is_gpt:
            self.cfg = GPT2Config(**config_dict)
            self.tfmr = GPT2Model(self.cfg)
        else:
            self.cfg = LlamaConfig(**config_dict)
            self.tfmr = LlamaModel(self.cfg)

        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=self.is_gpt)
        self.compiled = False
        self.patched_model = None
        self._compiled_decode_callables = {}
        self._compile_rotary_disabled = False
        self._turbo_custom_decoders = {}
        self._turbo_native_decode_callables = {}
        self._turbo_native_step_callables = {}
        self._turbo_dynamic_decoders = {}
        self._turbo_logits_callables = {}

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if not self.is_gpt:
                t3_cond.cond_prompt_speech_emb += self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0 and not self.is_gpt:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]  # final tfmr layer output, (B, seq, dim)

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(out.text_logits, masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(out.speech_logits, masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor]=None,

        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor]=None,

        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0.5,
        tf32_after_tokens: Optional[int] = None,
        compile_decode: bool = False,
        compile_mode: str = "default",
        show_progress: bool = True,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        if max_new_tokens is None:
            max_new_tokens = self.hp.max_speech_tokens

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.

        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if self.patched_model is None:
            self.patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
            )
            self.compiled = True

        # # Run normal generate method, which calls our custom extended methods
        # return self.patched_model.generate(
        #     inputs=initial_speech_tokens,
        #     decoder_cond=embeds,
        #     bos_token_id=self.hp.start_speech_token,
        #     eos_token_id=(self.hp.stop_speech_token if stop_on_eos else -1),
        #     pad_token_id=self.hp.stop_speech_token,
        #     max_new_tokens=max_new_tokens or self.hp.max_speech_tokens,
        #     num_return_sequences=num_return_sequences,
        #     temperature=temperature,
        #     min_p=min_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated token ids in a fixed buffer to avoid a per-token concat.
        generated_ids = torch.empty((1, max_new_tokens + 1), dtype=torch.long, device=device)
        generated_ids[:, :1].copy_(bos_token)
        generated_length = 1

        # Instantiate the logits processors.
        top_p_warper = None if top_p == 1.0 else TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values
        cfg = torch.as_tensor(cfg_weight, device=output.logits.device, dtype=output.logits.dtype)
        if compile_decode and compile_mode not in self._compiled_decode_callables:
            if compile_mode not in {"default", "reduce-overhead", "max-autotune-no-cudagraphs"}:
                raise ValueError(f"Unsupported compile_mode: {compile_mode}")
            if not self._compile_rotary_disabled:
                self.tfmr.rotary_emb.forward = torch.compiler.disable(
                    self.tfmr.rotary_emb.forward,
                    recursive=False,
                    reason="RoPE buffer capture is incompatible with dynamic T3 decode",
                )
                self._compile_rotary_disabled = True
            self._compiled_decode_callables[compile_mode] = torch.compile(
                self.patched_model.forward,
                dynamic=True,
                fullgraph=False,
                mode=compile_mode,
            )
        decode_forward = (
            self._compiled_decode_callables[compile_mode]
            if compile_decode
            else self.patched_model.forward
        )

        if tf32_after_tokens is not None and tf32_after_tokens < 1:
            raise ValueError("tf32_after_tokens must be positive")
        original_matmul_precision = (
            torch.get_float32_matmul_precision() if tf32_after_tokens is not None else None
        )

        # ---- Generation Loop using kv_cache ----
        try:
            for i in tqdm(
                range(max_new_tokens),
                desc="Sampling",
                dynamic_ncols=True,
                disable=not show_progress,
            ):
                logits_step = output.logits[:, -1, :]
                # CFG combine  → (1, V)
                cond   = logits_step[0:1, :]
                uncond = logits_step[1:2, :]
                logits = cond + cfg * (cond - uncond)

                # Apply repetition penalty
                ids_for_proc = generated_ids[:, :generated_length]
                logits = repetition_penalty_processor(ids_for_proc, logits)  # expects (B,V)

                # Apply temperature scaling.
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply min_p and top_p filtering
                logits = min_p_warper(ids_for_proc, logits)
                if top_p_warper is not None:
                    logits = top_p_warper(ids_for_proc, logits)

                # Convert logits to probabilities and sample the next token.
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

                generated_ids[:, generated_length : generated_length + 1].copy_(next_token)
                generated_length += 1

                # Check for EOS token.
                if next_token.view(-1) == self.hp.stop_speech_token:
                    logger.info(f"✅ EOS token detected! Stopping generation at step {i+1}")
                    break

                # Get embedding for the new token.
                next_token_embed = self.speech_emb(next_token)
                next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

                #  For CFG
                next_token_embed = torch.cat([next_token_embed, next_token_embed])

                if tf32_after_tokens is not None and i + 1 == tf32_after_tokens:
                    torch.set_float32_matmul_precision("high")

                # Forward pass with only the new token and the cached past.
                if compile_decode and compile_mode == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                output = decode_forward(
                    inputs_embeds=next_token_embed,
                    past_key_values=past,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                # Update the kv_cache.
                past = output.past_key_values
                if compile_decode and compile_mode == "reduce-overhead":
                    for layer in past.layers:
                        if layer.is_initialized:
                            layer.keys = layer.keys.clone()
                            layer.values = layer.values.clone()
        finally:
            if original_matmul_precision is not None:
                torch.set_float32_matmul_precision(original_matmul_precision)

        return generated_ids[:, 1:generated_length]

    @torch.inference_mode()
    def inference_turbo(self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2,
                        max_gen_len=1000, optimize_loop=False, optimize_sync=False,
                        preallocate_kv=False, custom_decode=False,
                        custom_cache_dtype="float32", custom_compile=True,
                        compile_native_decode=False,
                        native_compile_mode="default",
                        compile_native_step=False,
                        dynamic_decode=False, dynamic_cache_dtype="bfloat16",
                        dynamic_compile=True,
                        hybrid_decode_after=None,
                        compile_logits=False,
                        show_progress=True):

        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        compiled_logits = None
        if compile_logits:
            logits_key = (temperature, top_k, top_p, repetition_penalty)
            compiled_logits = self._turbo_logits_callables.get(logits_key)
            if compiled_logits is None:
                compiled_logits = torch.compile(
                    TurboLogitsProcessor(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    ),
                    dynamic=True,
                    fullgraph=True,
                    options={"triton.cudagraphs": False},
                )
                self._turbo_logits_callables[logits_key] = compiled_logits

        native_step = None
        if compile_native_step:
            if compile_logits:
                raise ValueError(
                    "compile_native_step already includes logits processing"
                )
            step_key = (temperature, top_k, top_p, repetition_penalty)
            native_step = self._turbo_native_step_callables.get(step_key)
            if native_step is None:
                step_logits = TurboLogitsProcessor(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

                def native_step_fn(current_token, input_ids, past_key_values):
                    current_embed = self.speech_emb(current_token)
                    outputs = self.tfmr.forward(
                        inputs_embeds=current_embed,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    speech_logits = self.speech_head(outputs[0])[:, -1, :]
                    processed_logits, probs = step_logits(input_ids, speech_logits)
                    return processed_logits, probs, outputs.past_key_values

                native_step = torch.compile(
                    native_step_fn,
                    dynamic=True,
                    fullgraph=True,
                    options={"triton.cudagraphs": False},
                )
                self._turbo_native_step_callables[step_key] = native_step


        speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        if optimize_loop:
            generated_speech_tokens = torch.empty(
                text_tokens.size(0), max_gen_len + 1, dtype=torch.long, device=self.device
            )
            generated_length = 0
        else:
            generated_speech_tokens = []

        past_key_values = (
            PreallocatedDynamicCache(
                config=self.tfmr.config,
                max_cache_len=embeds.size(1) + max_gen_len,
            )
            if preallocate_kv
            else None
        )
        llm_outputs = self.tfmr(
            inputs_embeds=embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )

        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_hidden = hidden_states[:, -1:]
        speech_logits = self.speech_head(speech_hidden)

        if compiled_logits is None:
            processed_logits = logits_processors(
                speech_start_token,
                speech_logits[:, -1, :],
            )
            probs = F.softmax(processed_logits, dim=-1)
        else:
            processed_logits, probs = compiled_logits(
                speech_start_token,
                speech_logits[:, -1, :],
            )
        next_speech_token = torch.multinomial(probs, num_samples=1)

        if optimize_loop:
            generated_speech_tokens[:, generated_length : generated_length + 1].copy_(
                next_speech_token
            )
            generated_length += 1
        else:
            generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token

        custom_decoder = None
        dynamic_decoder = None
        dynamic_decoder_loaded = False
        custom_cache_position = 0
        native_decode = self.tfmr.forward
        selected_decode_modes = sum(
            (
                custom_decode,
                compile_native_decode,
                compile_native_step,
                dynamic_decode,
                hybrid_decode_after is not None,
            )
        )
        if selected_decode_modes > 1:
            raise ValueError(
                "custom_decode, compile_native_decode, compile_native_step, "
                "dynamic_decode, and hybrid_decode_after are mutually exclusive"
            )
        if hybrid_decode_after is not None and hybrid_decode_after < 1:
            raise ValueError("hybrid_decode_after must be positive")
        if native_compile_mode not in (
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ):
            raise ValueError(
                "native_compile_mode must be default, reduce-overhead, or "
                "max-autotune-no-cudagraphs"
            )
        if compile_native_decode or hybrid_decode_after is not None:
            compile_key = f"fullgraph_dynamic_{native_compile_mode}"
            if compile_key not in self._turbo_native_decode_callables:
                compile_kwargs = {
                    "dynamic": True,
                    "fullgraph": True,
                }
                if native_compile_mode == "default":
                    compile_kwargs["options"] = {"triton.cudagraphs": False}
                else:
                    compile_kwargs["mode"] = native_compile_mode
                self._turbo_native_decode_callables[compile_key] = torch.compile(
                    self.tfmr.forward,
                    **compile_kwargs,
                )
            native_decode = self._turbo_native_decode_callables[compile_key]
        if custom_decode:
            required_cache_len = embeds.size(1) + max_gen_len
            decoder_key = (
                text_tokens.size(0),
                custom_cache_dtype,
                custom_compile,
            )
            custom_decoder = self._turbo_custom_decoders.get(decoder_key)
            if (
                custom_decoder is None
                or custom_decoder.max_cache_len < required_cache_len
            ):
                custom_decoder = TurboGPT2Decoder(
                    self.tfmr,
                    batch_size=text_tokens.size(0),
                    max_cache_len=max(2048, required_cache_len),
                    cache_dtype=custom_cache_dtype,
                    compile_decode=custom_compile,
                )
                self._turbo_custom_decoders[decoder_key] = custom_decoder
            custom_cache_position = custom_decoder.load_cache(past_key_values)
        if dynamic_decode or hybrid_decode_after is not None:
            decoder_key = (dynamic_cache_dtype, dynamic_compile)
            dynamic_decoder = self._turbo_dynamic_decoders.get(decoder_key)
            if dynamic_decoder is None:
                dynamic_decoder = TurboGPT2DynamicDecoder(
                    self.tfmr,
                    cache_dtype=dynamic_cache_dtype,
                    compile_decode=dynamic_compile,
                )
                self._turbo_dynamic_decoders[decoder_key] = dynamic_decoder
            if dynamic_decode:
                dynamic_decoder.load_cache(past_key_values)
                dynamic_decoder_loaded = True

        stopped_on_eos = False
        for decode_step in tqdm(range(max_gen_len), disable=not show_progress):
            if native_step is not None:
                input_ids = (
                    generated_speech_tokens[:, :generated_length]
                    if optimize_loop
                    else torch.cat(generated_speech_tokens, dim=1)
                )
                processed_logits, probs, past_key_values = native_step(
                    current_speech_token,
                    input_ids,
                    past_key_values,
                )
            else:
                current_speech_embed = self.speech_emb(current_speech_token)

                if (
                    hybrid_decode_after is not None
                    and decode_step == hybrid_decode_after
                ):
                    dynamic_decoder.load_cache(past_key_values)
                    dynamic_decoder_loaded = True

                if dynamic_decoder_loaded:
                    hidden_states = dynamic_decoder(current_speech_embed)
                elif custom_decoder is None:
                    if native_compile_mode == "reduce-overhead":
                        torch.compiler.cudagraph_mark_step_begin()
                    llm_outputs = native_decode(
                        inputs_embeds=current_speech_embed,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    hidden_states = llm_outputs[0]
                    past_key_values = llm_outputs.past_key_values
                    if native_compile_mode == "reduce-overhead":
                        for layer in past_key_values.layers:
                            if layer.is_initialized:
                                layer.keys = layer.keys.clone()
                                layer.values = layer.values.clone()
                else:
                    hidden_states = custom_decoder(
                        current_speech_embed,
                        custom_cache_position,
                    )
                    custom_cache_position += current_speech_embed.size(1)
                speech_logits = self.speech_head(hidden_states)

                input_ids = (
                    generated_speech_tokens[:, :generated_length]
                    if optimize_loop
                    else torch.cat(generated_speech_tokens, dim=1)
                )
                if compiled_logits is None:
                    processed_logits = logits_processors(
                        input_ids,
                        speech_logits[:, -1, :],
                    )
                    probs = F.softmax(processed_logits, dim=-1)
                else:
                    processed_logits, probs = compiled_logits(
                        input_ids,
                        speech_logits[:, -1, :],
                    )
            if not optimize_sync and torch.all(processed_logits == -float("inf")):
                print("Warning: All logits are -inf")
                break

            next_speech_token = torch.multinomial(probs, num_samples=1)

            if optimize_loop:
                generated_speech_tokens[:, generated_length : generated_length + 1].copy_(
                    next_speech_token
                )
                generated_length += 1
            else:
                generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token
            if optimize_sync and next_speech_token.numel() == 1:
                stopped_on_eos = next_speech_token.item() == self.hp.stop_speech_token
            else:
                stopped_on_eos = bool(
                    torch.all(next_speech_token == self.hp.stop_speech_token).item()
                )
            if stopped_on_eos:
                break

        all_tokens = (
            generated_speech_tokens[:, :generated_length]
            if optimize_loop
            else torch.cat(generated_speech_tokens, dim=1)
        )

        # Remove EOS token if present
        if stopped_on_eos or (
            not optimize_sync
            and all_tokens.size(1) > 0
            and all_tokens[0, -1] == self.hp.stop_speech_token
        ):
            all_tokens = all_tokens[:, :-1]

        return all_tokens
