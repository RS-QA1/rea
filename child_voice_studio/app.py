"""Open Telugu character TTS service: no Sarvam or Google credentials required."""
import os
import sys
import uuid
import re
import tempfile
import requests
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

DELIVERY_PRESETS = {
    "Natural": {"tempo": 1.00, "pitch": 0.0},
    "Excited": {"tempo": 1.10, "pitch": 0.35},
    "Playful": {"tempo": 1.06, "pitch": 0.30},
    "Shy": {"tempo": 0.91, "pitch": -0.15},
    "Sad": {"tempo": 0.88, "pitch": -0.30},
    "Angry": {"tempo": 1.08, "pitch": -0.20},
    "Whisper-like": {"tempo": 0.90, "pitch": -0.15},
}

LAUGH_SOURCE_URL = "https://commons.wikimedia.org/wiki/Special:Redirect/file/Four_Years_Old_child_laughing.ogg"
LAUGH_CREDIT = "Four Years Old child laughing by morgantj (CC BY 3.0, Wikimedia Commons)"
EVENT = re.compile(r"(\\[(?:giggle|laugh|chuckle|hehe|haha)\\])", re.IGNORECASE)

def interpret_delivery(prompt, preset, text):
    """Turn supported instructions into explicit audio events, never spoken ha-ha."""
    p = (prompt or "").lower()
    mood = preset if preset in DELIVERY_PRESETS else "Natural"
    keywords = [
        ("Excited", ("excited", "exciting", "ఉత్సాహ")),
        ("Playful", ("playful", "naughty", "సరదా")),
        ("Shy", ("shy", "softly", "సిగ్గు")),
        ("Sad", ("sad", "crying", "దుఃఖ", "ఏడుస్తూ")),
        ("Angry", ("angry", "కోపం")),
        ("Whisper-like", ("whisper", "గుసగుస")),
    ]
    for candidate, needles in keywords:
        if any(word in p for word in needles):
            mood = candidate
            break
    laugh = any(w in p for w in ("laugh", "giggle", "chuckle", "hehe", "haha", "నవ్వ"))
    if laugh and not EVENT.search(text):
        before = any(w in p for w in ("before", "start", "first", "ముందు"))
        after = any(w in p for w in ("after", "end", "finish", "తర్వాత", "చివర"))
        if before:
            text = "[giggle] " + text
        elif after or laugh:
            text = text + " [giggle]"
    return text, mood, bool(EVENT.search(text))

def run_ffmpeg(args, timeout=90):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args,
                       text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError("Audio processing failed: " + p.stderr[-350:])

def laugh_source(uploaded):
    if uploaded:
        candidate = Path(uploaded)
        if not candidate.is_file() or candidate.stat().st_size > 15_000_000:
            raise ValueError("Uploaded laugh recording is missing or larger than 15 MB.")
        return str(candidate), "User-provided laughter recording"
    cache = OUTPUTS / "licensed_child_giggle.ogg"
    if not cache.exists() or cache.stat().st_size < 4096:
        try:
            response = requests.get(
                LAUGH_SOURCE_URL, timeout=20,
                headers={"User-Agent": "TeluguVoiceStudio/0.5 (audio attribution test)"}
            )
            response.raise_for_status()
            if len(response.content) > 5_000_000 or len(response.content) < 4096:
                raise ValueError("Invalid laugh clip size.")
            temp = cache.with_suffix(".download")
            temp.write_bytes(response.content)
            # Only publish a complete successful download
            temp.replace(cache)
        except Exception as exc:
            raise RuntimeError(
                "Built-in laugh sample could not be fetched from Wikimedia. "
                "Upload a permitted giggle recording instead. "
                + str(exc)[:100]
            ) from exc
    return str(cache), LAUGH_CREDIT

def render_acted(text, character, tempo, pitch_offset, pause_cleanup, tuning, laugh_upload):
    parts = EVENT.split(text)
    events = [p for p in parts if p and p.strip()]
    if sum(bool(EVENT.fullmatch(p.strip())) for p in events) > 6:
        raise ValueError("Maximum 6 laugh markers per dialogue.")
    if not any(EVENT.fullmatch(p.strip()) for p in events):
        return render(text, character, tempo, pitch_offset, pause_cleanup, tuning), "No laugh event"
    source, credit = laugh_source(laugh_upload)
    with tempfile.TemporaryDirectory(dir=OUTPUTS) as directory:
        root = Path(directory)
        segments = []
        for index, item in enumerate(events):
            pcm = root / f"{index:02d}.wav"
            if EVENT.fullmatch(item.strip()):
                # Separate real non-speech laugh, not TTS syllables.
                # Select 1.75 seconds of the recorded child's laughter.
                run_ffmpeg(["-ss", "0.5", "-t", "1.75", "-i", source,
                            "-af", "aresample=24000,highpass=f=100,"
                                   "afade=t=in:st=0:d=0.06,afade=t=out:st=1.54:d=0.18,"
                                   "volume=0.85",
                            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(pcm)])
            else:
                spoken = render(item.strip(), character, tempo, pitch_offset, pause_cleanup, tuning)
                run_ffmpeg(["-i", spoken, "-ac", "1", "-ar", "24000",
                            "-c:a", "pcm_s16le", str(pcm)])
            if not pcm.exists() or pcm.stat().st_size < 100:
                raise RuntimeError("One speech or giggle segment produced no audio.")
            segments.append(pcm)
        listing = root / "concat.txt"
        listing.write_text("".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'\\n" for p in segments), encoding="utf-8")
        output = OUTPUTS / (uuid.uuid4().hex + "_acted.mp3")
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing),
                    "-codec:a", "libmp3lame", "-b:a", "160k", str(output)], timeout=150)
        return str(output), credit

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

def from_ui(text, character, tempo, pitch_offset, pause_cleanup, tuning, delivery, acting_prompt, laugh_upload):
    try:
        prepared, mood, laugh = interpret_delivery(acting_prompt, delivery, text or "")
        style = DELIVERY_PRESETS[mood]
        applied_tempo = max(0.8, min(1.25, tempo * style["tempo"]))
        applied_pitch = max(-2.0, min(2.0, pitch_offset + style["pitch"]))
        if not re.sub(EVENT, "", prepared).strip():
            raise ValueError("Add spoken Telugu dialogue alongside the laugh.")
        if len(prepared) > 1200:
            raise ValueError("Dialogue exceeds 1200 characters.")
        filepath, credit = render_acted(
            prepared, character, applied_tempo, applied_pitch,
            pause_cleanup, tuning, laugh_upload
        )
        info = (f"Delivery: {mood}. Speed {applied_tempo:.2f}x. " +
                (f"Real recorded laughter inserted. Credit: {credit}. " if laugh else "") +
                "Current Telugu model remains adult-trained. Emotions alter pacing; "
                "they do not produce native acting or character-matched laughter.")
        return filepath, filepath, info
    except Exception as exc:
        raise gr.Error(str(exc)[:600]) from exc

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
    gr.Markdown("### 🎭 Acting direction and laughs")
    delivery = gr.Dropdown(
        choices=list(DELIVERY_PRESETS), value="Natural",
        label="Emotion / delivery",
    )
    acting_prompt = gr.Textbox(
        label="How should the character speak? (English or Telugu keywords)",
        placeholder="Excited and playful. Giggle before speaking, then say the line quickly.",
        lines=3,
    )
    gr.Markdown("Add `[giggle]` or `[laugh]` anywhere in the script to insert a real, "
                "separate recorded laugh. Default sample: [Four Years Old child laughing]"
                "(https://commons.wikimedia.org/wiki/File:Four_Years_Old_child_laughing.ogg), "
                "morgantj, CC BY 3.0. Laughter may differ from the spoken character voice.")
    laugh_upload = gr.Audio(
        label="Optional: upload a licensed laugh clip matching your character",
        sources=["upload"], type="filepath")

    go = gr.Button("🎤 Generate voice", variant="primary")
    preview = gr.Audio(label="Listen", type="filepath")
    download = gr.File(label="Download MP3")
    note = gr.Textbox(label="Voice info", interactive=False)
    go.click(from_ui, inputs=[script, character, tempo, pitch, pause_cleanup, tuning, delivery, acting_prompt, laugh_upload],
             outputs=[preview, download, note], api_name="generate_voice")
    gr.Markdown("Gradio API: `generate_voice` is available through the Space API. "
                "Model attribution: Indic Tiny TTS Telugu by Sulabh Katiyar; AI4Bharat Rasa dataset (CC BY 4.0). "
                "Use native Telugu script for best pronunciation.")

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=True)
