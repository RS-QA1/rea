"""Telugu Voice Studio — two free CPU models, four base voices, no gated models."""
import os
import re
import sys
import uuid
import wave
import shutil
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache

import gradio as gr
from huggingface_hub import snapshot_download, hf_hub_download

STORE = Path("/tmp/telugu_voice_v2")
STORE.mkdir(parents=True, exist_ok=True)

BASE_VOICES = {
    "Indic Tiny · Female": ("indic", "female"),
    "Indic Tiny · Male": ("indic", "male"),
    "Piper · Padmavathi": ("piper", "padmavathi"),
    "Piper · Venkatesh": ("piper", "venkatesh"),
}
CHARACTERS = {
    "Natural voice": (0.0, 1.00),
    "Playful younger character (approximation)": (1.2, 1.05),
    "Gentle younger character (approximation)": (0.6, 0.95),
    "Energetic cartoon character (approximation)": (1.6, 1.10),
}
EMOTIONS = {
    "Natural": 1.00,
    "Playful": 1.05,
    "Excited": 1.10,
    "Gentle": 0.94,
    "Sad / quiet": 0.91,
    "Storytelling": 0.98,
}
MARKER = re.compile(r"(\[(?:giggle|laugh|chuckle)\])", re.IGNORECASE)

@lru_cache(maxsize=1)
def load_indic():
    folder = snapshot_download(
        "sulabhkatiyar/indic-tiny-tts-telugu",
        ignore_patterns=["*.safetensors", "samples/*"],
    )
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from indic_tiny_tts import TinyTTSOnnx
    return TinyTTSOnnx.from_pretrained(folder)

@lru_cache(maxsize=2)
def load_piper(voice):
    from piper import PiperVoice
    sub = f"te/te_IN/{voice}/medium/te_IN-{voice}-medium.onnx"
    repo = "rhasspy/piper-voices"
    path = hf_hub_download(repo_id=repo, filename=sub)
    hf_hub_download(repo_id=repo, filename=sub + ".json")
    return PiperVoice.load(path)

def command(arguments, timeout=90):
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"] + arguments,
        capture_output=True, text=True, timeout=timeout
    )
    if p.returncode:
        raise RuntimeError("Audio processing error: " + p.stderr[-300:])

def synthesize_base(text, selected, destination):
    backend, speaker = BASE_VOICES[selected]
    if backend == "indic":
        engine = load_indic()
        audio = engine.synthesize(text, voice=speaker, speed=1.0)
        engine.save_wav(audio, str(destination))
    else:
        engine = load_piper(speaker)
        with wave.open(str(destination), "wb") as wf:
            engine.synthesize_wav(text, wf)
    if not destination.exists() or destination.stat().st_size < 1024:
        raise RuntimeError("No synthesized audio was returned.")

def render_speech(text, voice, pitch, pace, out):
    with tempfile.TemporaryDirectory(dir=STORE) as folder:
        raw = Path(folder) / "raw.wav"
        synthesize_base(text, voice, raw)
        # The base voice remains unmodified for natural mode.
        if abs(pitch) < 0.01 and abs(pace - 1.0) < 0.01:
            command(["-i", str(raw), "-ac", "1", "-ar", "24000",
                     "-c:a", "pcm_s16le", str(out)])
        else:
            semitones = max(-1.0, min(2.0, pitch))
            factor = 2 ** (semitones / 12.0)
            filters = (
                f"aresample=24000,asetrate=24000*{factor:.7f},"
                f"aresample=24000,atempo={1/factor:.7f},"
                f"atempo={pace:.7f}"
            )
            command(["-i", str(raw), "-af", filters, "-ac", "1",
                     "-ar", "24000", "-c:a", "pcm_s16le", str(out)])

def make_audio(text, voice, character, emotion, speed, laugh_upload, instructions):
    text = (text or "").strip()
    if not text or len(text) > 1500:
        raise gr.Error("Enter 1–1500 Telugu characters.")
    if voice not in BASE_VOICES or character not in CHARACTERS or emotion not in EMOTIONS:
        raise gr.Error("Select supported engine, character and delivery.")
    direction = (instructions or "").lower()
    # Limited deterministic acting instructions; no claim of full prompt following.
    if "giggle before" in direction or "laugh before" in direction:
        if not MARKER.search(text):
            text = "[giggle] " + text
    elif "giggle after" in direction or "laugh after" in direction:
        if not MARKER.search(text):
            text += " [giggle]"
    parts = [p.strip() for p in MARKER.split(text) if p.strip()]
    laugh_count = sum(bool(MARKER.fullmatch(p)) for p in parts)
    if laugh_count > 5:
        raise gr.Error("Use no more than five laugh markers.")
    if laugh_count and not laugh_upload:
        raise gr.Error("For real laughter, upload a permitted laugh clip or remove [giggle]/[laugh].")
    if laugh_upload and Path(laugh_upload).stat().st_size > 15_000_000:
        raise gr.Error("Laugh upload must be 15 MB or less.")
    pitch, char_pace = CHARACTERS[character]
    pace = max(0.80, min(1.25, float(speed) * char_pace * EMOTIONS[emotion]))
    job = STORE / uuid.uuid4().hex
    job.mkdir()
    fragments = []
    for i, part in enumerate(parts):
        destination = job / f"{i:03d}.wav"
        if MARKER.fullmatch(part):
            command(["-ss", "0", "-t", "1.6", "-i", str(laugh_upload),
                     "-af", "afade=t=in:st=0:d=0.04,afade=t=out:st=1.35:d=0.2",
                     "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(destination)])
        else:
            render_speech(part, voice, pitch, pace, destination)
        if destination.stat().st_size < 1024:
            raise RuntimeError("An audio segment was empty.")
        fragments.append(destination)
    if not fragments:
        raise gr.Error("Enter spoken dialogue.")
    manifest = job / "list.txt"
    manifest.write_text(
        "".join("file '" + str(p) + "'\n" for p in fragments), encoding="utf-8"
    )
    final = STORE / (uuid.uuid4().hex + ".mp3")
    command(["-f", "concat", "-safe", "0", "-i", str(manifest),
             "-b:a", "160k", "-codec:a", "libmp3lame", str(final)], timeout=180)
    shutil.rmtree(job, ignore_errors=True)
    if not final.exists() or final.stat().st_size < 1024:
        raise RuntimeError("Final audio export failed.")
    info = (f"{voice} · {character} · {emotion} · {pace:.2f}×. "
            + ("Recorded laugh inserted. " if laugh_count else "")
            + "Character effects approximate youth; authentic child-trained voices are not included.")
    return str(final), str(final), info

def safe_generate(*args):
    try:
        return make_audio(*args)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error("Audio generation failed: " + str(exc)[:500]) from exc

with gr.Blocks(title="Telugu Voice Studio — Free Multi-Engine") as demo:
    gr.Markdown("# 🎙️ Telugu Voice Studio — Free Multi-Engine")
    gr.Markdown("**Two free CPU TTS engines, four Telugu base voices.** No Sarvam account, "
                "gated DhVaani model or paid API required. Real children's speech is not "
                "guaranteed; youthful presets are mild approximations.")
    with gr.Row():
        with gr.Column():
            dialogue = gr.Textbox(label="తెలుగు డైలాగ్ / Telugu dialogue",
                                  lines=5, value="అమ్మా! చూడు! నేను ఒక చిన్న డ్రాగన్‌ని చూశాను!")
            voice = gr.Dropdown(list(BASE_VOICES), value="Indic Tiny · Female",
                                label="Voice model + speaker")
            character = gr.Dropdown(list(CHARACTERS), value="Natural voice",
                                    label="Character style")
            emotion = gr.Dropdown(list(EMOTIONS), value="Natural", label="Expression approximation")
            speed = gr.Slider(0.85, 1.15, value=1.0, step=0.05, label="Speech speed")
            instructions = gr.Textbox(label="Optional acting directions (limited)",
                placeholder="Giggle after speaking", lines=2)
            gr.Markdown("Use `[giggle]`, `[laugh]` or `[chuckle]` at the exact position "
                        "you want a **recorded** laugh. Supply a recording that you have the rights to use.")
            laugh_upload = gr.Audio(label="Optional laugh recording", sources=["upload"], type="filepath")
            generate = gr.Button("Generate voice", variant="primary")
        with gr.Column():
            player = gr.Audio(label="Listen", type="filepath")
            downloadable = gr.File(label="Download MP3")
            details = gr.Textbox(label="Result details", interactive=False)
    generate.click(safe_generate,
                   inputs=[dialogue, voice, character, emotion, speed, laugh_upload, instructions],
                   outputs=[player, downloadable, details], api_name="generate_voice")
    gr.Markdown("Model attribution: [Indic Tiny TTS Telugu](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-telugu) "
                "(acoustic CC BY 4.0, code MIT); [Piper Telugu models](https://huggingface.co/rhasspy/piper-voices) "
                "(check each voice model card and dataset license).")
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
