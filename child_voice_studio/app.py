"""Open Telugu character TTS service: no Sarvam or Google credentials required."""
import os
import sys
import uuid
import shutil
import subprocess
from pathlib import Path
from functools import lru_cache

import gradio as gr
from huggingface_hub import snapshot_download

MODEL_ID = "sulabhkatiyar/indic-tiny-tts-telugu"
OUTPUTS = Path("/tmp/telugu_voice_outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

# These are distinct character sound profiles using two genuine base speakers.
# They are NOT separate child-trained identities or guaranteed ages.
CHARACTERS = {
    "Chinni · playful girl (~6 style)": {"base": "female", "pitch": 1.29, "speed": 1.07},
    "Bujji · gentle girl (~6 style)": {"base": "female", "pitch": 1.20, "speed": 0.97},
    "Chintu · energetic boy (~6 style)": {"base": "male", "pitch": 1.34, "speed": 1.09},
    "Balu · curious boy (~6 style)": {"base": "male", "pitch": 1.25, "speed": 1.01},
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

@lru_cache(maxsize=1)
def has_rubberband():
    """Query installed FFmpeg filters once; never assume Space image has it."""
    try:
        check = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=10,
        )
        return check.returncode == 0 and " rubberband " in check.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False

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
    # Previous pipeline resampled at 24 kHz and compensated large pitch boosts
    # with atempo; this can cause metallic/warbling artifacts on child presets.
    # Rubber Band's quality algorithm processes pitch and tempo together.
    # A restrained pitch range reduces unnatural distortion further.
    child_profile = "(~6 style)" in character
    factor = min(factor, 1.42 if child_profile else 1.55)
    source_rate = 48000
    shared = "highpass=f=70,lowpass=f=10000,alimiter=limit=0.92:level=0"
    if abs(factor - 1.0) < 0.005 and abs(tempo - 1.0) < 0.005:
        filters = "aresample=48000," + shared
    elif has_rubberband():
        filters = (f"aresample={source_rate},"
                   f"rubberband=pitch={factor:.7f}:tempo={tempo:.7f}:"
                   "transients=smooth:detector=soft:window=long:"
                   "smoothing=on:pitchq=quality:formant=shifted,"
                   + shared)
    else:
        # Safe fallback when the system FFmpeg lacks librubberband.
        filters = (f"aresample={source_rate},"
                   f"asetrate={source_rate}*{factor:.7f},"
                   f"aresample={source_rate},"
                   f"atempo={1/factor:.7f},atempo={tempo:.7f},"
                   + shared)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
           str(raw), "-af", filters, "-codec:a", "libmp3lame",
           "-b:a", "192k", "-ar", "48000", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    raw.unlink(missing_ok=True)
    if result.returncode or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Audio processing failed: " + result.stderr[-300:])
    return str(output)

def from_ui(text, character, tempo, pitch_offset):
    try:
        filepath = render(text, character, tempo, pitch_offset)
        return filepath, filepath, ("Clean audio v0.3 · " + ("high-quality Rubber Band processing" if has_rubberband() else "compatible resampling") + ". Synthetic character approximations; not child-trained voices.")
    except Exception as exc:
        raise gr.Error(str(exc)[:500]) from exc

with gr.Blocks(title="Telugu Kids Character Voice Studio") as ui:
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
    gr.Markdown("Gradio API: `generate_voice` is available through the Space API. "
                "Model attribution: Indic Tiny TTS Telugu by Sulabh Katiyar; AI4Bharat Rasa dataset (CC BY 4.0). "
                "Use native Telugu script for best pronunciation.")

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=True)
