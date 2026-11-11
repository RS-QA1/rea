"""Child-trained XTTS English model lab v0.2 — separate tested backend."""
import os
import uuid
from pathlib import Path
from functools import lru_cache

import gradio as gr

MODEL_ID = "Aniket2003333333/xtts-v2-child-voice-finetuned"
STORE = Path("/tmp/child_voice_lab")
STORE.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=1)
def engine():
    import torch
    from huggingface_hub import snapshot_download
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    folder = Path(snapshot_download(MODEL_ID))
    ckpt = folder / "model.pth"
    source = folder / "best_model.pth"
    if not ckpt.exists() and source.exists():
        ckpt.symlink_to(source)
    reference = folder / "reference_child.wav"
    if not reference.is_file():
        raise RuntimeError("Child-trained model does not include a reference WAV.")
    cfg = XttsConfig()
    cfg.load_json(str(folder / "config.json"))
    obj = Xtts.init_from_config(cfg)
    obj.load_checkpoint(cfg, checkpoint_dir=str(folder), eval=True)
    obj.eval()
    if torch.cuda.is_available():
        obj.cuda()
    return obj, cfg, str(reference), ("cuda" if torch.cuda.is_available() else "cpu")

def generate(text, temperature):
    text = (text or "").strip()
    if not text or len(text) > 260:
        raise gr.Error("Enter 1–260 English characters.")
    try:
        import numpy as np
        import soundfile as sf
        import torch
        obj, cfg, ref, device = engine()
        with torch.inference_mode():
            result = obj.synthesize(
                text, cfg, speaker_wav=ref, language="en",
                temperature=float(temperature)
            )
        audio = np.asarray(result["wav"], dtype=np.float32).reshape(-1)
        if audio.size < 12000 or not np.isfinite(audio).all():
            raise RuntimeError("No valid audio generated.")
        path = STORE / (uuid.uuid4().hex + ".wav")
        sf.write(path, np.clip(audio, -0.98, 0.98), 24000)
        return str(path), str(path), (
            f"Child-trained XTTS reference model on {device}. "
            "English output only. Verify quality and model/voice licensing."
        )
    except Exception as exc:
        raise gr.Error("Child-model generation failed: "+str(exc)[:600]) from exc

with gr.Blocks(title="Child Voice Model Lab") as ui:
    gr.Markdown("# 🎤 Child-trained Voice Model Lab")
    gr.Markdown(
        "Experimental **real child-speech-trained** XTTS fine-tune. "
        "English only — Telugu not supported by this checkpoint. "
        "The checkpoint is over 2 GB and may exceed free CPU resources. "
        "This model uses its bundled reference voice; no recording upload required."
    )
    script=gr.Textbox(label="English children's dialogue",lines=4,
         value="Mom! Look! I found a little dragon in our garden! Wow!")
    temp=gr.Slider(0.6,1.0,value=0.8,step=0.05,label="Expressiveness sampling temperature")
    btn=gr.Button("Generate child-model speech",variant="primary")
    audio=gr.Audio(label="Preview",type="filepath")
    download=gr.File(label="Download WAV")
    details=gr.Textbox(label="Status",interactive=False)
    btn.click(generate,[script,temp],[audio,download,details],api_name="generate_child_voice")
    gr.Markdown("Research model: [XTTS child voice fine-tune](https://huggingface.co/Aniket2003333333/xtts-v2-child-voice-finetuned). "
                "Check rights and the underlying Coqui model license before commercial use.")

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0",server_port=7860,show_error=True)
