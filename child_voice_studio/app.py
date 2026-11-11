"""Open Telugu character TTS service: no Sarvam or Google credentials required."""
import os
import sys
import uuid
import shutil
import subprocess
import wave
import numpy as np
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
    "Chinni · playful girl (experimental)": {"base": "female", "pitch": 1.13, "speed": 1.07},
    "Bujji · gentle girl (experimental)": {"base": "female", "pitch": 1.08, "speed": 0.97},
    "Chintu · energetic boy (experimental)": {"base": "male", "pitch": 1.16, "speed": 1.09},
    "Balu · curious boy (experimental)": {"base": "male", "pitch": 1.10, "speed": 1.01},
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

def shorten_long_gaps(wav_path):
    """Shorten only long, low-energy internal pauses; preserve sentence starts/ends."""
    with wave.open(str(wav_path), "rb") as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        raw_bytes = reader.readframes(reader.getnframes())
    if width != 2 or channels != 1:
        return False
    samples = np.frombuffer(raw_bytes, dtype="<i2").copy()
    if len(samples) < rate:
        return False
    frame = max(1, rate // 100)
    count = len(samples) // frame
    if count < 40:
        return False
    rms = np.sqrt(np.mean(
        (samples[:count * frame].astype(np.float32).reshape(count, frame) / 32768.0) ** 2,
        axis=1
    ))
    level = max(0.0025, float(np.percentile(rms, 95)) * 0.065)
    silence = rms < level
    spans = []
    start = None
    for i, quiet in enumerate(silence):
        if quiet and start is None:
            start = i
        elif not quiet and start is not None:
            if i - start >= 36 and start > 0 and i < count:
                spans.append((start * frame, i * frame))
            start = None
    if not spans:
        return False
    result = []
    cursor = 0
    keep = int(rate * 0.18)
    for left, right in spans:
        amount = max(0, right - left - keep)
        remove_left = left + max(0, ((right - left) - amount) // 2)
        remove_right = remove_left + amount
        result.append(samples[cursor:remove_left])
        cursor = remove_right
    result.append(samples[cursor:])
    combined = np.concatenate(result)
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(combined.astype("<i2").tobytes())
    return True

def render(text, character, tempo=1.0, pitch_offset=0.0, pause_cleanup=True, tuning="Natural / clean"):
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
    if pause_cleanup:
        shorten_long_gaps(raw)
    base_factor = profile["pitch"]
    if tuning == "Original / previous pitch" and "experimental" in character:
        legacy = {"Chinni": 1.43, "Bujji": 1.27, "Chintu": 1.54, "Balu": 1.38}
        base_factor = next((v for n, v in legacy.items() if character.startswith(n)), base_factor)
    factor = base_factor * 2 ** (pitch_offset / 12.0)
    # Pitch and vocal-resonance rise together; there is no claim this creates
    # authentic child anatomy. Audio length remains controllable independently.
    if abs(factor - 1.0) < 0.001 and abs(tempo - 1.0) < 0.001:
        filters = "aresample=24000"
    else:
        filters = (f"aresample=24000,asetrate=24000*{factor:.7f},"
                   f"aresample=24000,atempo={1/factor:.7f},"
                   f"atempo={tempo:.7f}")
    filters += ",alimiter=limit=0.96"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
           str(raw), "-af", filters, "-codec:a", "libmp3lame", "-q:a", "3",
           str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    raw.unlink(missing_ok=True)
    if result.returncode or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Audio processing failed: " + result.stderr[-300:])
    return str(output)

def from_ui(text, character, tempo, pitch_offset, pause_cleanup, tuning):
    try:
        filepath = render(text, character, tempo, pitch_offset, pause_cleanup, tuning)
        return filepath, filepath, ("Audio v0.4 — " + tuning + ". Adult-trained base voice. " + ("Long-pause cleanup on." if pause_cleanup else "Original pauses preserved."))
    except Exception as exc:
        raise gr.Error(str(exc)[:500]) from exc

with gr.Blocks(title="Telugu Kids Character Voice Studio") as ui:
    gr.Markdown("# 🎙️ తెలుగు పిల్లల వాయిస్ స్టూడియో\n### Paste a Telugu dialogue, select a character, listen and download MP3.")
    gr.Markdown("**Independent open model — no Sarvam key needed.** Child profiles are experimental processing of adult voices, not genuine six-year-old voices.")
    script = gr.Textbox(label="తెలుగు డైలాగ్ / Telugu dialogue", lines=5,
        value="హాయ్! నా పేరు చింటూ! అమ్మా, చూడూ! నేను ఒక చిన్న డ్రాగన్‌ని చూశాను!")
    with gr.Row():
        character = gr.Dropdown(choices=list(CHARACTERS), value="Chintu · energetic boy (experimental)",
                                label="Character — choose boy, girl, or older voice")
        tempo = gr.Slider(minimum=0.8, maximum=1.25, value=1.0, step=0.05,
                          label="Speaking speed")
        pitch = gr.Slider(minimum=-2, maximum=2, value=0, step=0.5,
                          label="Fine pitch adjustment (semitones)")
    with gr.Row():
        tuning = gr.Radio(
            choices=["Natural / clean", "Original / previous pitch"],
            value="Natural / clean",
            label="Child voice tuning — compare natural against original",
        )
        pause_cleanup = gr.Checkbox(
            label="Shorten long silent gaps (optional)", value=True
        )
    go = gr.Button("🎤 Generate voice", variant="primary")
    preview = gr.Audio(label="Listen", type="filepath")
    download = gr.File(label="Download MP3")
    note = gr.Textbox(label="Voice info", interactive=False)
    go.click(from_ui, inputs=[script, character, tempo, pitch, pause_cleanup, tuning],
             outputs=[preview, download, note], api_name="generate_voice")
    gr.Markdown("Gradio API: `generate_voice` is available through the Space API. "
                "Model attribution: Indic Tiny TTS Telugu by Sulabh Katiyar; AI4Bharat Rasa dataset (CC BY 4.0). "
                "Use native Telugu script for best pronunciation.")

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=True)
