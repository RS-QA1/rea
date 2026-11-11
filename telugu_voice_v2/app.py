"""Telugu Reference Voice Studio v2 — isolated from the previous broken Space."""
import os
import uuid
from pathlib import Path
from functools import lru_cache

import gradio as gr

STORE = Path("/tmp/telugu_voice_v2")
STORE.mkdir(parents=True, exist_ok=True)

# Built-in voices use generated speech, not third-party human recordings.

@lru_cache(maxsize=1)
def load_preset_engine():
    import sys
    from huggingface_hub import snapshot_download
    folder = snapshot_download("sulabhkatiyar/indic-tiny-tts-telugu", ignore_patterns=["*.safetensors", "samples/*"])
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from indic_tiny_tts import TinyTTSOnnx
    return TinyTTSOnnx.from_pretrained(folder)

STYLES = {
    "Natural": 1.00,
    "Playful (faster)": 1.06,
    "Excited (faster)": 1.12,
    "Gentle (slower)": 0.94,
    "Storytelling (slower)": 0.97,
}

def synthesize(text, speaker_name, style, speed):
    """Only ungated ONNX voices, no reference-cloning dependency."""
    text = (text or "").strip()
    if not text or len(text) > 400:
        raise gr.Error("Enter Telugu dialogue between 1 and 400 characters.")
    voice = {"Telugu female": "female", "Telugu male": "male"}.get(speaker_name)
    if not voice:
        raise gr.Error("Choose one of the two available Telugu voices.")
    if style not in STYLES:
        raise gr.Error("Unknown delivery style.")
    try:
        engine = load_preset_engine()
        pace = max(0.8, min(1.25, float(speed) * STYLES[style]))
        signal = engine.synthesize(text, voice=voice, speed=pace)
        output = STORE / (str(uuid.uuid4()) + ".wav")
        engine.save_wav(signal, str(output))
        if not output.exists() or output.stat().st_size < 512:
            raise RuntimeError("Model returned no audio.")
        return str(output), str(output), (
            f"Ungated Telugu {voice} voice generated at {pace:.2f}x. "
            "This is adult-trained speech, not genuine child-voice cloning."
        )
    except Exception as exc:
        raise gr.Error("Ungated Telugu generation failed: " + str(exc)[:500]) from exc

def diagnostics():
    from importlib.metadata import version
    return ("UI ready. Model loads only when Generate is clicked. "
            f"Gradio {version('gradio')}; "
            f"Transformers {version('transformers')}. "
            "This is NOT proof that model inference works.")

with gr.Blocks(title="Telugu Voice Studio — Ungated") as demo:
    gr.Markdown("# 🎙️ Telugu Voice Studio — No API key needed")
    gr.Markdown("Select a Telugu voice, enter your dialogue, and download WAV audio. "
                "These are adult-trained models; expressive child voice cloning is not available in this build.")
    with gr.Row():
        with gr.Column():
            speaker = gr.Radio(choices=["Telugu female", "Telugu male"], value="Telugu female",
                               label="Voice")
            text = gr.Textbox(lines=5, value="అమ్మా! నేను ఒక చిన్న డ్రాగన్‌ని చూశాను!",
                              label="Telugu dialogue")
            delivery = gr.Dropdown(choices=list(STYLES), value="Natural", label="Delivery")
            speed = gr.Slider(0.85, 1.15, value=1.0, step=0.05, label="Speed")
            generate = gr.Button("Generate Telugu voice", variant="primary")
        with gr.Column():
            preview = gr.Audio(label="Listen", type="filepath")
            download = gr.File(label="Download WAV")
            status = gr.Textbox(label="Status")
    generate.click(synthesize, inputs=[text, speaker, delivery, speed],
                   outputs=[preview, download, status], api_name="generate_voice")
    gr.Markdown("Engine: Indic Tiny TTS Telugu ONNX. No gated DhVaani model or upload "
                "requirements. These voices are not authentic six-year-old voices.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
