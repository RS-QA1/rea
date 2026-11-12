"""Experimental Telugu zero-shot reference-voice studio (DhVaani)."""
import os
import math
import pathlib
import re
import subprocess
import tempfile
import uuid
from functools import lru_cache
from difflib import SequenceMatcher
import unicodedata

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

def check_audio(path):
    with sf.SoundFile(path) as f:
        duration = len(f) / f.samplerate
    if duration < 3 or duration > 30:
        raise gr.Error(f"Reference is {duration:.1f}s. Use 5–15 seconds of clean speaking audio.")
    return duration

@lru_cache(maxsize=1)
def get_recognizer():
    from faster_whisper import WhisperModel

    return WhisperModel(
        "itsskofficial/livewhisper-te-medium",
        device="cpu", compute_type="int8", cpu_threads=2,
    )


def transcribe(path, language="te"):
    samples, rate = sf.read(str(path), dtype="float32")
    from scipy.signal import resample_poly

    if rate != 16000:
        samples = resample_poly(samples, 16000, rate)
    segments, _ = get_recognizer().transcribe(
        samples, language=language, beam_size=5, condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def comparable_text(text):
    return "".join(
        character for character in unicodedata.normalize("NFC", text).casefold()
        if unicodedata.category(character)[0] in "LMN"
    )


def matches_dialogue(expected, recognized):
    expected = comparable_text(expected)
    recognized = comparable_text(recognized)
    if not recognized or not any("\u0c05" <= c <= "\u0c7f" for c in recognized):
        return False
    if len(expected) < 12:
        return expected == recognized
    return SequenceMatcher(None, expected, recognized, autojunk=False).ratio() >= 0.85


def generate(text, reference_audio, reference_text, expression, speed, steps, permission,
             start_seconds=0, clip_seconds=0):
    if not permission:
        raise gr.Error("Confirm that you have the speaker's permission to use their voice.")
    if not text or not text.strip():
        raise gr.Error("Enter Telugu dialogue.")
    if not any("\u0c05" <= character <= "\u0c7f" for character in text):
        return None, None, "Enter your new dialogue in Telugu script."
    if len(text) > 500:
        raise gr.Error("Use up to 500 characters per generation during testing.")
    if not reference_audio:
        raise gr.Error("Upload an audio or video file with clear speech.")
    if pathlib.Path(reference_audio).stat().st_size > 100_000_000:
        raise gr.Error("Reference upload exceeds 100 MB.")
    if not all(math.isfinite(value) for value in (start_seconds, clip_seconds)):
        raise gr.Error("Enter valid reference times.")
    if start_seconds < 0 or (clip_seconds != 0 and not 3 <= clip_seconds <= 30):
        raise gr.Error("Use a nonnegative start and a clip length of 3–30 seconds.")
    style_speed, style_gain = EXPRESSIONS[expression]
    # Does not pretend to generate natural emotion or giggling natively.
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        ref = pathlib.Path(tmp) / "reference.wav"
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", str(start_seconds), "-i", str(reference_audio),
             "-map", "0:a:0", "-vn", "-t", str(clip_seconds or 31),
             "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(ref)],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode:
            raise gr.Error("Reference conversion failed: " + proc.stderr[-250:])
        duration = check_audio(ref)
        reference_text = (reference_text or "").strip()
        if not reference_text:
            try:
                reference_text = transcribe(ref)
            except Exception:
                return None, None, "Automatic transcription could not run. Enter the reference words and try again."
            if not reference_text:
                return None, None, "Reference speech could not be transcribed. Select clear speech or enter the spoken words."
        model, device = get_model()
        import torch

        candidate = pathlib.Path(tmp) / "candidate.wav"
        recognized = ""
        sentences = [
            part.strip() for part in re.findall(r"[^.!?।\n]+[.!?।]*", text)
            if part.strip()
        ]
        for inference_steps in (int(steps), 16):
            parts = []
            pause = np.zeros(int(model.sampling_rate * 0.16), dtype=np.float32)
            with torch.inference_mode():
                for sentence in sentences:
                    waveform = model.synthesize(
                        text=sentence, prompt_wav=str(ref), prompt_text=reference_text,
                        num_step=inference_steps, speed=float(speed) * style_speed,
                    )
                    if hasattr(waveform, "detach"):
                        waveform = waveform.detach().cpu().numpy()
                    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
                    if not samples.size or not np.isfinite(samples).all():
                        break
                    parts.extend((samples, pause))
            if len(parts) != 2 * len(sentences):
                continue
            signal = np.concatenate(parts[:-1])
            sf.write(str(candidate), np.clip(signal * style_gain, -0.96, 0.96), int(model.sampling_rate))
            try:
                recognized = transcribe(candidate, language="te")
            except Exception:
                return None, None, "The Telugu speech checker could not run. Please try again."
            if matches_dialogue(text, recognized):
                output = ROOT / f"{uuid.uuid4().hex}.wav"
                candidate.replace(output)
                return str(output), str(output), (
                    f"Telugu transcript check passed. Recognized: {recognized}\n"
                    f"Reference: {duration:.1f}s · DhVaani on {device}. "
                    "Listen to compare the generated voice with the reference."
                )
    return None, None, (
        "The generated speech did not match the requested Telugu dialogue after two attempts. "
        f"Recognized: {recognized or 'no clear speech'}. "
        "Check the reference transcript and use a short, clear speaking segment."
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
            reference = gr.File(label="1. Upload reference audio or video",
                                file_types=["audio", "video"], type="filepath")
            with gr.Row():
                start_seconds = gr.Number(value=0, minimum=0, label="Reference start (seconds)")
                clip_seconds = gr.Number(value=0, minimum=0, maximum=30,
                                         label="Reference length (0 = whole short clip)")
            transcript = gr.Textbox(label="2. Exact words spoken in the selected segment",
                                    lines=2, placeholder="Type the words as spoken, or leave blank to transcribe automatically")
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
            status = gr.Textbox(label="Generation status", interactive=False)
    button.click(
        generate,
        [dialogue, reference, transcript, mood, speed, steps, permission, start_seconds, clip_seconds],
        [preview, download, status],
        api_name="generate_reference_voice",
    )
    gr.Markdown("Model: ARTPARK-IISc/DhVaani-0.5 · CC BY-NC 4.0, "
                "subject to the rights of reference recordings and training-data licenses. "
                "Model requires accepting gated access conditions and a read-only "
                "HF_MODEL_TOKEN Space Secret. CPU Basic may be too slow or run out of RAM.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
                show_error=True)
