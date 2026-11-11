"""Open Telugu character TTS service: no Sarvam or Google credentials required."""
import os
import sys
import uuid
import shutil
import subprocess
from pathlib import Path
from functools import lru_cache

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

MODEL_ID = "sulabhkatiyar/indic-tiny-tts-telugu"
OUTPUTS = Path("/tmp/telugu_voice_outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

# These are distinct character sound profiles using two genuine base speakers.
# They are NOT separate child-trained identities or guaranteed ages.
CHARACTERS = {
    "Chinni · playful girl (~6 style)": {"base": "female", "pitch": 1.43, "speed": 1.12},
    "Bujji · gentle girl (~6 style)": {"base": "female", "pitch": 1.27, "speed": 0.94},
    "Chintu · energetic boy (~6 style)": {"base": "male", "pitch": 1.54, "speed": 1.17},
    "Balu · curious boy (~6 style)": {"base": "male", "pitch": 1.38, "speed": 1.00},
    "Akka · older sister (~12 style)": {"base": "female", "pitch": 1.13, "speed": 1.03},
    "Anna · older brother (~12 style)": {"base": "male", "pitch": 1.22, "speed": 1.06},
    "Adult Telugu female": {"base": "female", "pitch": 1.00, "speed": 1.00},
    "Adult Telugu male": {"base": "male", "pitch": 1.00, "speed": 1.00},
}

@lru_cache(maxsize=1)
def model():
    folder = snapshot_download(MODEL_ID, ignore_patterns=["*.safetensors", "samples/*"])
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from indic_tiny_tts import TinyTTSOnnx
    return TinyTTSOnnx.from_pretrained(folder)

def render(text, character, tempo=1.0, pitch_offset=0.0):
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter Telugu dialogue.")
    if len(text) > 1200:
        raise ValueError("Limit dialogue to 1200 characters per request.")
    if character not in CHARACTERS:
        raise ValueError("Unknown character.")
    if not 0.8 <= tempo <= 1.25 or not -2 <= pitch_offset <= 2:
        raise ValueError("Speed and pitch adjustment out of range.")
    profile = CHARACTERS[character]
    engine = model()
    audio = engine.synthesize(text, voice=profile["base"], speed=profile["speed"])
    base = OUTPUTS / (uuid.uuid4().hex)
    raw = base.with_suffix(".wav")
    output = base.with_suffix(".mp3")
    engine.save_wav(audio, str(raw))
    factor = profile["pitch"] * 2 ** (pitch_offset / 12.0)
    # Pitch and vocal-resonance rise together; there is no claim this creates
    # authentic child anatomy. Audio length remains controllable independently.
    filters = (f"aresample=24000,asetrate=24000*{factor:.7f},"
               f"aresample=24000,atempo={1/factor:.7f},"
               f"atempo={tempo:.7f},alimiter=limit=0.96")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
           str(raw), "-af", filters, "-codec:a", "libmp3lame", "-q:a", "3",
           str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    raw.unlink(missing_ok=True)
    if result.returncode or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Audio processing failed: " + result.stderr[-300:])
    return str(output)

def from_ui(text, character, tempo, pitch_offset):
    try:
        filepath = render(text, character, tempo, pitch_offset)
        return filepath, filepath, ("Generated using our own API and an open Telugu model. "
                                    "These are character-style approximations, not trained child voices.")
    except Exception as exc:
        raise gr.Error(str(exc)[:500]) from exc

api = FastAPI(title="Independent Telugu Character TTS API", version="0.2")
class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    character: str = "Chinni · playful girl (~6 style)"
    tempo: float = Field(default=1.0, ge=0.8, le=1.25)
    pitch_offset: float = Field(default=0.0, ge=-2, le=2)

@api.get("/api/health")
def health():
    return {"ok": True, "engine": "indic-tiny-tts-telugu-onnx", "model_loaded": model.cache_info().currsize > 0}

@api.get("/api/voices")
def voices():
    return [{"name": name, "base_speaker": p["base"],
             "age_claim": "stylized only"} for name, p in CHARACTERS.items()]

@api.post("/api/generate")
def generate(request: SpeechRequest):
    try:
        filename = render(request.text, request.character, request.tempo, request.pitch_offset)
        return FileResponse(filename, media_type="audio/mpeg", filename="telugu-character.mp3")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Voice generation failed: {str(exc)[:200]}") from exc

with gr.Blocks(title="Telugu Kids Character Voice Studio", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 🎙️ తెలుగు పిల్లల వాయిస్ స్టూడియో\n### Paste a Telugu dialogue, select a character, listen and download MP3.")
    gr.Markdown("**Independent open model — no Sarvam key needed.** Voices are synthetic age-*style* effects, not real six-year-old voices.")
    script = gr.Textbox(label="తెలుగు డైలాగ్ / Telugu dialogue", lines=5,
        value="హాయ్! నా పేరు చింటూ! అమ్మా, చూడూ! నేను ఒక చిన్న డ్రాగన్‌ని చూశాను!")
    with gr.Row():
        character = gr.Dropdown(choices=list(CHARACTERS), value="Chintu · energetic boy (~6 style)",
                                label="Character — choose boy, girl, or older voice")
        tempo = gr.Slider(minimum=0.8, maximum=1.25, value=1.0, step=0.05,
                          label="Speaking speed")
        pitch = gr.Slider(minimum=-2, maximum=2, value=0, step=0.5,
                          label="Fine pitch adjustment (semitones)")
    go = gr.Button("🎤 Generate voice", variant="primary")
    preview = gr.Audio(label="Listen", type="filepath")
    download = gr.File(label="Download MP3")
    note = gr.Textbox(label="Voice info", interactive=False)
    go.click(from_ui, inputs=[script, character, tempo, pitch],
             outputs=[preview, download, note], api_name="generate_voice")
    gr.Markdown("Open API: `GET /api/voices`, `GET /api/health`, `POST /api/generate`. "
                "Model attribution: Indic Tiny TTS Telugu by Sulabh Katiyar; AI4Bharat Rasa dataset (CC BY 4.0). "
                "Use native Telugu script for best pronunciation.")

app = gr.mount_gradio_app(api, ui, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
