"""Telugu Dialogue Voice Studio: free baseline and optional Sarvam TTS."""
import base64
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import gradio as gr
from gtts import gTTS
import requests

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SPEAKERS = ["shreya", "priya", "pooja", "shubh", "rahul", "kavya", "anand", "ritu"]

def generate(dialogue, engine, speaker, pace, youthful):
    dialogue = (dialogue or "").strip()
    if not dialogue:
        raise gr.Error("తెలుగు డైలాగ్ రాయండి / Please enter Telugu dialogue.")
    if len(dialogue) > 2500:
        raise gr.Error("Maximum 2500 characters per generation.")
    name = uuid.uuid4().hex
    raw = OUT / (name + "_raw.mp3")
    try:
        if engine == "Free Telugu voice":
            gTTS(text=dialogue, lang="te", slow=False).save(str(raw))
        else:
            key = os.environ.get("SARVAM_API_KEY", "").strip()
            if not key:
                raise gr.Error("Sarvam API key is missing. Add SARVAM_API_KEY in Space Settings > Secrets, or use Free Telugu voice.")
            response = requests.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": key, "Content-Type":"application/json"},
                json={"text": dialogue, "target_language_code":"te-IN",
                      "model":"bulbul:v3", "speaker":speaker,
                      "pace":pace, "output_audio_codec":"mp3"},
                timeout=90,
            )
            if not response.ok:
                raise gr.Error(f"Sarvam returned HTTP {response.status_code}: {response.text[:250]}")
            audios = response.json().get("audios") or []
            if not audios:
                raise gr.Error("Sarvam returned no audio.")
            raw.write_bytes(base64.b64decode(audios[0]))
        final = OUT / (name + ".mp3")
        if youthful:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise gr.Error("FFmpeg missing. Turn off childlike pitch or install FFmpeg.")
            # Raise pitch ~3 semitones and preserve tempo.
            f = 2 ** (3 / 12)
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-i", str(raw), "-af",
                   f"asetrate=24000*{f:.7f},aresample=24000,atempo={1/f:.7f}",
                   str(final)]
            # Force an exact sample rate before altering pitch, to avoid assuming the input file's rate.
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-i", str(raw), "-af",
                   f"aresample=24000,asetrate=24000*{f:.7f},aresample=24000,atempo={1/f:.7f}",
                   str(final)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if p.returncode:
                raise gr.Error("FFmpeg processing failed: " + p.stderr[-250:])
        else:
            shutil.copyfile(raw, final)
        return str(final), str(final), "Generated. Listen and download below."
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error("Voice generation failed: " + str(exc)[:350]) from exc

with gr.Blocks(title="Telugu Dialogue Voice Studio") as app:
    gr.Markdown("# 🎙️ Telugu Dialogue Voice Studio\nPaste dialogue in Telugu, generate speech, play it, and download MP3.\n\n*Free voice is a baseline. The Sarvam option needs your API key, set privately as a Space Secret. 'Childlike' is pitch processing, not a genuine child voice.*")
    dialogue = gr.Textbox(label="తెలుగు డైలాగ్ / Telugu Dialogue", lines=6,
        value="హాయ్! నా పేరు చింటూ. ఈ రోజు మనం ఒక అద్భుతమైన కథ వినబోతున్నాం!")
    with gr.Row():
        engine = gr.Radio(["Free Telugu voice", "Sarvam Bulbul v3 (API key required)"],
                          value="Free Telugu voice", label="Voice engine")
        speaker = gr.Dropdown(SPEAKERS, value="shreya", label="Sarvam speaker")
    with gr.Row():
        pace = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Sarvam speech pace")
        youthful = gr.Checkbox(label="Childlike pitch effect (experimental)", value=False)
    go = gr.Button("🎤 Generate Telugu voice", variant="primary")
    preview = gr.Audio(label="Listen to voice", type="filepath")
    download = gr.File(label="Download MP3")
    status = gr.Markdown()
    go.click(generate, [dialogue, engine, speaker, pace, youthful], [preview,download,status])
    gr.Markdown("Do not paste API keys in the dialogue box. The free voice requires network access to Google's translation TTS backend.")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
