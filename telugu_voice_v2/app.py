"""Telugu Reference Voice Studio v2 — isolated from the previous broken Space."""
import os
import uuid
import tempfile
import subprocess
from pathlib import Path
from functools import lru_cache

import gradio as gr
import numpy as np
import soundfile as sf

MODEL_ID = "ARTPARK-IISc/DhVaani-0.5"
STORE = Path("/tmp/telugu_voice_v2")
STORE.mkdir(parents=True, exist_ok=True)

# Built-in voices use generated speech, not third-party human recordings.
REFERENCE_PRESETS = {"Built-in Telugu female": "female", "Built-in Telugu male": "male", "Upload authorized reference": "upload"}
SYNTH_REF_TEXT = "నమస్కారం! నా పేరు చిన్నారి. ఈరోజు నేను ఒక చిన్న కథ చెబుతాను."

@lru_cache(maxsize=1)
def load_preset_engine():
    import sys
    from huggingface_hub import snapshot_download
    folder = snapshot_download("sulabhkatiyar/indic-tiny-tts-telugu", ignore_patterns=["*.safetensors", "samples/*"])
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from indic_tiny_tts import TinyTTSOnnx
    return TinyTTSOnnx.from_pretrained(folder)

@lru_cache(maxsize=2)
def synthetic_reference(voice):
    if voice not in ("female", "male"):
        raise ValueError("Unknown preset voice")
    target = STORE / ("synthetic_" + voice + ".wav")
    engine = load_preset_engine()
    audio = engine.synthesize(SYNTH_REF_TEXT, voice=voice, speed=1.0)
    engine.save_wav(audio, str(target))
    return str(target)
STYLES = {
    "Natural": 1.00,
    "Playful (faster)": 1.06,
    "Excited (faster)": 1.12,
    "Gentle (slower)": 0.94,
    "Storytelling (slower)": 0.97,
}

def convert_audio(filepath, out):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(filepath), "-vn", "-ac", "1", "-ar", "24000", "-c:a",
           "pcm_s16le", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if result.returncode != 0:
        raise RuntimeError("Cannot read the reference audio: " + result.stderr[-350:])

@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.getenv("HF_MODEL_TOKEN") or os.getenv("HF_TOKEN")
    # The source model may require account approval; model retrieval errors are
    # surfaced directly instead of making the UI fail during startup.
    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    try:
        model = AutoModel.from_pretrained(MODEL_ID, **kwargs).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            "Model download/loading failed. Check access to " + MODEL_ID +
            ", any required HF_MODEL_TOKEN Space secret, and available RAM. " +
            str(exc)[:350]
        ) from exc
    return model, device

def synthesize(text, reference_mode, audio_reference, exact_transcript, style, speed, steps, consent):
    using_upload = REFERENCE_PRESETS.get(reference_mode) == "upload"
    if reference_mode not in REFERENCE_PRESETS:
        raise gr.Error("Select a reference voice")
    if using_upload and not consent:
        raise gr.Error("Confirm permission to use the uploaded speaker's voice.")
    text = (text or "").strip()
    exact_transcript = (exact_transcript or "").strip()
    if not text or len(text) > 400:
        raise gr.Error("Enter Telugu dialogue between 1 and 400 characters.")
    # Ungated built-in Telugu voices use their own ONNX inference directly.
    # Do NOT fetch the gated DhVaani model unless custom cloning was selected.
    if not using_upload:
        try:
            engine = load_preset_engine()
            speaker = REFERENCE_PRESETS[reference_mode]
            pace = float(speed) * STYLES[style]
            signal = engine.synthesize(text, voice=speaker, speed=pace)
            target = STORE / (str(uuid.uuid4()) + "_preset.wav")
            engine.save_wav(signal, str(target))
            if not target.exists() or target.stat().st_size < 512:
                raise RuntimeError("Built-in model returned no audio")
            return str(target), str(target), (
                f"Generated Telugu with built-in {speaker} ONNX voice, "
                f"speed {pace:.2f}. No gated model or API key required. "
                "This is an adult-trained voice, not child voice cloning."
            )
        except Exception as exc:
            raise gr.Error("Built-in Telugu generation failed: " + str(exc)[:500]) from exc
    if using_upload:
        if not exact_transcript or len(exact_transcript) > 500:
            raise gr.Error("Enter the exact transcript of the reference speech.")
        if not audio_reference:
            raise gr.Error("Upload spoken reference audio, not laughter-only audio.")
        if Path(audio_reference).stat().st_size > 15_000_000:
            raise gr.Error("Reference exceeds 15 MB.")
    with tempfile.TemporaryDirectory(dir=STORE) as temp_dir:
        audio_wav = Path(temp_dir) / "reference.wav"
        reference_source = audio_reference if using_upload else synthetic_reference(REFERENCE_PRESETS[reference_mode])
        prompt_text = exact_transcript if using_upload else SYNTH_REF_TEXT
        convert_audio(reference_source, audio_wav)
        info = sf.info(str(audio_wav))
        seconds = info.duration
        if not 3 <= seconds <= 20:
            raise gr.Error(f"Reference is {seconds:.1f} seconds. Use 3–20 seconds of clear speech.")
        try:
            model, device = load_model()
            import torch
            with torch.inference_mode():
                signal = model.synthesize(
                    text=text,
                    prompt_wav=str(audio_wav),
                    prompt_text=prompt_text,
                    speed=float(speed) * STYLES[style],
                    num_step=int(steps),
                )
            if hasattr(signal, "detach"):
                signal = signal.detach().cpu().numpy()
            signal = np.asarray(signal, dtype=np.float32).reshape(-1)
            if signal.size == 0 or not np.isfinite(signal).all():
                raise RuntimeError("TTS returned empty or invalid audio.")
            output = STORE / (str(uuid.uuid4()) + ".wav")
            sf.write(str(output), np.clip(signal, -0.98, 0.98),
                     int(model.sampling_rate), subtype="PCM_16")
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error("Generation failed: " + str(exc)[:650]) from exc
    return str(output), str(output), (
        f"Generated {len(signal) / model.sampling_rate:.1f}s of Telugu audio on {device}. "
        "Style changes speed only; the model does not support native emotional acting "
        "or guaranteed child voice identity. Check the recording by listening."
    )

def diagnostics():
    from importlib.metadata import version
    return ("UI ready. Model loads only when Generate is clicked. "
            f"Gradio {version('gradio')}; "
            f"Transformers {version('transformers')}. "
            "This is NOT proof that model inference works.")

with gr.Blocks(title="Telugu Reference Voice Studio v2") as demo:
    gr.Markdown("# 🎙️ Telugu Reference Voice Studio v2")
    gr.Markdown("**Built-in Telugu voices use an ungated ONNX model with no token required.** "
                "Only `Upload authorized reference` uses gated DhVaani for voice cloning. "
                "These built-in voices are adult-trained, not authentic child voices.")
    with gr.Row():
        with gr.Column():
            reference_mode = gr.Radio(choices=list(REFERENCE_PRESETS), value="Built-in Telugu female", label="1. Ready-to-use voice preset")
            reference = gr.Audio(sources=["upload"], type="filepath",
                                 label="1. Reference SPEECH (3–20 seconds)")
            reference_text = gr.Textbox(lines=2, label="2. Exact words in that reference speech")
            consent = gr.Checkbox(label="I have the speaker's authorization (and guardian consent if applicable)")
            text = gr.Textbox(lines=4, value="అమ్మా! నేను ఒక చిన్న డ్రాగన్‌ని చూశాను!",
                              label="3. New Telugu dialogue")
            with gr.Row():
                style = gr.Dropdown(choices=list(STYLES), value="Natural",
                                    label="Delivery (speed approximation)")
                speed = gr.Slider(0.85, 1.15, value=1.0, step=0.05, label="Speed")
            steps = gr.Slider(8, 16, value=12, step=2, label="Quality steps")
            generate = gr.Button("Generate Telugu voice", variant="primary")
        with gr.Column():
            preview = gr.Audio(label="Listen", type="filepath")
            download = gr.File(label="Download WAV")
            message = gr.Textbox(label="Status", interactive=False)
            check = gr.Button("Check app dependencies")
            check_output = gr.Textbox(label="Environment", interactive=False)
    generate.click(synthesize, [text, reference_mode, reference, reference_text, style, speed, steps, consent],
                   [preview, download, message], api_name="generate_reference_voice")
    check.click(diagnostics, outputs=check_output)
    gr.Markdown("Built-in voices: Indic Tiny TTS Telugu (check model and dataset attribution). "
                "Custom-reference mode: DhVaani-0.5 (gated; CC BY-NC 4.0, noncommercial testing only). "
                "For cloning, accept model access and set `HF_MODEL_TOKEN` in Space Secrets. "
                "Reference speech needs authorization, including guardian consent for a child.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
