import os
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torchinductor").resolve()))

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


device = "cuda"
model = ChatterboxTTS.from_pretrained(device=device)
text = "This request uses the optimized regular English inference path."

fast_t3 = {
    "compile_t3_decode": True,
    "t3_compile_mode": "reduce-overhead",
    "t3_matmul_precision": "high",
    "show_progress": False,
}

# The first call compiles kernels and captures graph shapes for this token path.
torch.manual_seed(0)
model.generate(text, **fast_t3)

torch.manual_seed(0)
wav = model.generate(text, **fast_t3)
ta.save("optimized-english.wav", wav, model.sr)
