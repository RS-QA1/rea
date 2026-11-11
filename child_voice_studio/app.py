"""Independent CPU-friendly childlike character voice demo (not Sarvam AI)."""
import os
import shutil
import subprocess
import tempfile
import wave
from functools import lru_cache
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from piper import PiperVoice

VOICE_REPO = "rhasspy/piper-voices"
VOICES = {
    "Amy (US English)": "en/en_US/amy/medium/en_US-amy-medium.onnx",
    "Lessac (US English)": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
}
PRESETS = {
    "Little storyteller": (4.0, 1.03),
    "Playful cartoon kid": (6.0, 1.12),
    "Gentle young character": (2.5, 0.94),
}
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=2)
def load_voice(name):
    relative_path = VOICES[name]
    model = hf_hub_download(repo_id=VOICE_REPO, filename=relative_path, local_dir=MODEL_DIR)
    hf_hub_download(repo_id=VOICE_REPO, filename=relative_path + ".json", local_dir=MODEL_DIR)
    return PiperVoice.load(str(model))

def filters(sample_rate, semitones, speed):
    factor = 2 ** (semitones / 12.0)
    return (f"asetrate={sample_rate}*{factor:.9f},aresample={sample_rate},"
            f"atempo={1/factor:.9f},atempo={speed:.9f},alimiter=limit=0.95")

def generate(text, voice_name, preset_name, pitch, speed, use_preset):
    if not text or not text.strip():
        raise gr.Error("Enter some text first.")
    if len(text) > 2500:
        raise gr.Error("Limit text to 2,500 characters per generation.")
    if use_preset:
        pitch, speed = PRESETS[preset_name]
    if not (0 <= pitch <= 8 and 0.7 <= speed <= 1.5):
        raise gr.Error("Pitch or speed outside supported range.")
    try:
        voice = load_voice(voice_name)
    except Exception as exc:
        raise gr.Error(f"Voice model unavailable: {exc}") from exc
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "raw.wav"
        with wave.open(str(raw), "wb") as wav_file:
            voice.synthesize_wav(text.strip(), wav_file)
        with wave.open(str(raw), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
        output = OUTPUT_DIR / (next(tempfile._get_candidate_names()) + ".wav")
        if abs(pitch) < 0.01 and abs(speed - 1) < 0.01:
            shutil.copy2(raw, output)
        else:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise gr.Error("Install FFmpeg and restart the app.")
            result = subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                "-af", filters(sample_rate, pitch, speed), str(output)
            ], capture_output=True, text=True, timeout=120)
            if result.returncode != 0 or not output.exists() or output.stat().st_size < 128:
                raise gr.Error("Audio processing failed: " + (result.stderr[-600:] or "Unknown error"))
    return str(output), str(output), f"{voice_name}; pitch +{pitch:g} semitones; speed {speed:.2f}x."

with gr.Blocks(title="Child Voice Studio") as demo:
    gr.Markdown("# Child Voice Studio\nIndependent character voice generation using Piper TTS.")
    with gr.Row():
        with gr.Column():
            script = gr.Textbox(label="Script", lines=7, value="Hey! Guess what? I found a little dragon hiding in our garden!")
            name = gr.Dropdown(list(VOICES), value="Amy (US English)", label="Base voice")
            preset = gr.Dropdown(list(PRESETS), value="Little storyteller", label="Character preset")
            use_preset = gr.Checkbox(label="Use preset pitch and speed", value=True)
            pitch = gr.Slider(0, 8, value=4, step=0.5, label="Custom pitch (semitones)")
            speed = gr.Slider(0.7, 1.5, value=1.03, step=0.01, label="Custom speed")
            button = gr.Button("Generate voice", variant="primary")
        with gr.Column():
            audio = gr.Audio(label="Preview", type="filepath")
            download = gr.File(label="Download WAV")
            status = gr.Textbox(label="Generation details", interactive=False)
    button.click(generate, [script, name, preset, pitch, speed, use_preset], [audio, download, status])
    gr.Markdown("English-only prototype. Childlike voice effects use pitch processing, not child-trained voice models.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=True)
