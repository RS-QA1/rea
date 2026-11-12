"""Experimental Telugu zero-shot reference-voice studio (DhVaani)."""
import os
import pathlib
import subprocess
import tempfile
import uuid
from functools import lru_cache

import gradio as gr
import numpy as np
import soundfile as sf

MODEL_ID = "ARTPARK-IISc/DhVaani-0.5"
ROOT = pathlib.Path("/tmp/telugu_reference_studio")
ROOT.mkdir(parents=True, exist_ok=True)
EXPRESSIONS = {
    "Natural": (1.00, 1.00),
    "Cute / playful": (1.04, 1.02),
    "Excited": (1.10, 1.02),
    "Shy / gentle": (0.92, 0.99),
    "Sad / quiet": (0.90, 1.00),
    "Surprised": (1.06, 1.02),
}

@lru_cache(maxsize=1)
def get_model():
    import torch
    from transformers import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_MODEL_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "This model is gated. Accept model access conditions at "
            "https://huggingface.co/ARTPARK-IISc/DhVaani-0.5 "
            "and add a read-only HF_MODEL_TOKEN under this Space's Settings > Secrets."
        )
    # DhVaani downloads backend files internally without forwarding token=.
    os.environ["HF_TOKEN"] = token
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, token=token,
            revision="582a342b3eb310f973a8e0af06798b453234b6ab",
        ).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            "Could not load DhVaani model. Confirm you've accepted its gated "
            "access conditions, your Space secret HF_MODEL_TOKEN has read "
            "permission, and the Space has enough RAM. "
            + str(exc)[:220]
        ) from exc
    return model, device

def check_audio(path, transcript):
    if not path:
        raise gr.Error("Upload a 5–15 second sample of the voice SPEAKING, not just laughing.")
    if not transcript or not transcript.strip():
        raise gr.Error("Type the exact words spoken in the uploaded reference.")
    if pathlib.Path(path).stat().st_size > 15_000_000:
        raise gr.Error("Reference is too large (15 MB maximum).")
    with sf.SoundFile(path) as f:
        duration = len(f) / f.samplerate
    if duration < 3 or duration > 30:
        raise gr.Error(f"Reference is {duration:.1f}s. Use 5–15 seconds of clean speaking audio.")
    return duration

def generate(text, reference_audio, reference_text, expression, speed, steps, permission):
    if not permission:
        raise gr.Error("Confirm that you have the speaker's permission to use their voice.")
    if not text or not text.strip():
        raise gr.Error("Enter Telugu dialogue.")
    if len(text) > 500:
        raise gr.Error("Use up to 500 characters per generation during testing.")
    duration = check_audio(reference_audio, reference_text)
    model, device = get_model()
    style_speed, style_gain = EXPRESSIONS[expression]
    # Does not pretend to generate natural emotion or giggling natively.
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        ref = pathlib.Path(tmp) / "reference.wav"
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(reference_audio), "-ac", "1", "-ar", "24000", str(ref)],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode:
            raise gr.Error("Reference conversion failed: " + proc.stderr[-250:])
        import torch
        with torch.inference_mode():
            waveform = model.synthesize(
                text=text.strip(), prompt_wav=str(ref), prompt_text=reference_text.strip(),
                num_step=int(steps), speed=float(speed) * style_speed,
            )
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        signal = np.asarray(waveform, dtype=np.float32).flatten()
        if not signal.size or not np.isfinite(signal).all():
            raise gr.Error("Voice model returned invalid audio.")
        signal = np.clip(signal * style_gain, -0.96, 0.96)
        output = ROOT / f"{uuid.uuid4().hex}.wav"
        sf.write(str(output), signal, int(model.sampling_rate))
    return str(output), str(output), (
        f"Generated from a {duration:.1f}s spoken reference using DhVaani on {device}. "
        f"{expression} is a light speed adjustment, not guaranteed emotional acting. "
        "Voice similarity and childlike tone must be assessed by listening."
    )

with gr.Blocks(title="Telugu Reference Voice Studio") as demo:
    gr.Markdown("# Telugu Reference Voice Studio — experimental")
    gr.Markdown(
        "Make Telugu dialogue in the style of a reference **speaking** voice. "
        "This is an independent open-source model, not Sarvam. "
        "**Laughter-only recordings cannot establish a speaking voice.** "
        "Only upload recordings that you have permission to use; for a child's voice, "
        "obtain the parent/guardian's authorization."
    )
    with gr.Row():
        with gr.Column():
            reference = gr.Audio(label="1. Upload reference SPEECH (5–15 sec)",
                                 sources=["upload"], type="filepath")
            transcript = gr.Textbox(label="2. Exact transcript of the reference recording",
                                    lines=2, placeholder="The exact words spoken in that recording")
            permission = gr.Checkbox(label="I have permission to synthesize speech in this speaker's voice")
            dialogue = gr.Textbox(label="3. Telugu dialogue to generate", lines=5,
                value="అమ్మా! నేను ఒక చిన్న డ్రాగన్‌ని చూశాను!")
            mood = gr.Dropdown(list(EXPRESSIONS), value="Natural", label="Expression (approximate)")
            with gr.Row():
                speed = gr.Slider(0.8, 1.2, value=1.0, step=0.05, label="Speech speed")
                steps = gr.Slider(8, 16, value=12, step=2, label="Inference quality steps")
            button = gr.Button("Generate Telugu voice", variant="primary")
        with gr.Column():
            preview = gr.Audio(label="Listen", type="filepath")
            download = gr.File(label="Download WAV")
            status = gr.Textbox(label="Model details", interactive=False)
    button.click(generate, [dialogue, reference, transcript, mood, speed, steps, permission],
                 [preview, download, status], api_name="generate_reference_voice")
    gr.Markdown("Model: ARTPARK-IISc/DhVaani-0.5 · CC BY-NC 4.0, "
                "subject to the rights of reference recordings and training-data licenses. "
                "Model requires accepting gated access conditions and a read-only "
                "HF_MODEL_TOKEN Space Secret. CPU Basic may be too slow or run out of RAM.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
                show_error=True)
