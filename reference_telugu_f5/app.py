"""Reference-conditioned Telugu TTS: use authorized speaking audio from a video.
Not a claim of guaranteed voice identity. No user reference files are committed.
"""
import os
import tempfile
import uuid
import subprocess
from pathlib import Path
from functools import lru_cache

import gradio as gr
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "multilingual-tts/F5-TTS-OpenBible-Telugu"
HOME = Path("/tmp/telugu-reference-child")
HOME.mkdir(parents=True, exist_ok=True)

def ffmpeg(*arguments):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, arguments)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=75)
    if p.returncode:
        raise RuntimeError("Audio conversion failed: " + p.stderr[-350:])

@lru_cache(maxsize=1)
def load_models():
    import torch
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_model, load_vocoder

    config = hf_hub_download(REPO, "F5-TTS_OpenBible_Telugu.yaml")
    checkpoint = hf_hub_download(REPO, "model_last.pt")
    vocab = hf_hub_download(REPO, "vocab.txt")
    cfg = OmegaConf.load(config)
    model_class = get_class(f"f5_tts.model.{cfg.model.backbone}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        model_class, cfg.model.arch, checkpoint,
        mel_spec_type="vocos", vocab_file=vocab,
        use_ema=True, device=device,
    )
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
    return model, vocoder, device

def make_reference(video, clip_start, duration, folder):
    if not video:
        raise ValueError("Upload your authorized reference video or audio.")
    path = Path(video)
    if not path.exists() or path.stat().st_size > 100_000_000:
        raise ValueError("Reference missing or larger than 100 MB.")
    if clip_start < 0 or not 3.0 <= duration <= 12.0:
        raise ValueError("Choose 3–12 seconds of speaking audio.")
    ref = folder / "ref.wav"
    # Reference stays in this request's temporary directory; never hosted in GitHub.
    ffmpeg("-ss", f"{clip_start:.3f}", "-t", f"{duration:.3f}",
           "-i", path, "-vn", "-ac", "1", "-ar", "24000",
           "-c:a", "pcm_s16le", ref)
    info = sf.info(str(ref))
    if info.duration < 2.5:
        raise ValueError("The selected voice sample is too short.")
    return ref

def generate(dialogue, video, start, duration, transcript, steps, speed, authorized):
    if not authorized:
        raise gr.Error("Confirm you have permission to synthesize this speaker's voice; for a minor, guardian permission is needed.")
    dialogue = (dialogue or "").strip()
    if not dialogue or len(dialogue) > 450:
        raise gr.Error("Enter 1–450 characters of Telugu dialogue.")
    if not any(0x0C00 <= ord(c) <= 0x0C7F for c in dialogue):
        raise gr.Error("This checkpoint is Telugu-focused; please enter Telugu-script dialogue.")
    try:
        import torch
        import torchaudio
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        # F5 utils_infer calls torchaudio.load(), which in recent TorchAudio
        # versions delegates to TorchCodec's AudioDecoder. TorchCodec 0.1.x
        # lacks that interface on PyTorch 2.5.1. All input has already been
        # decoded to mono PCM WAV by FFmpeg, so read it with libsndfile.
        def load_pcm_wav(path, *args, **kwargs):
            if args or kwargs:
                raise ValueError("Only standard WAV reads are supported here.")
            data, sample_rate = sf.read(
                str(path), dtype="float32", always_2d=True
            )
            return torch.from_numpy(np.ascontiguousarray(data.T)), sample_rate
        torchaudio.load = load_pcm_wav

        with tempfile.TemporaryDirectory(dir=HOME) as temp:
            folder = Path(temp)
            reference = make_reference(video, float(start), float(duration), folder)
            ref_text = (transcript or "").strip()
            # The upstream library can attempt automatic transcription when empty;
            # accuracy for noisy/foreign-language reference speech is not guaranteed.
            ref_audio, ref_text = preprocess_ref_audio_text(str(reference), ref_text)
            model, vocoder, device = load_models()
            wav, sr, _ = infer_process(
                ref_audio, ref_text, dialogue, model, vocoder,
                mel_spec_type="vocos", device=device,
                nfe_step=int(steps), speed=float(speed),
            )
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            if audio.size < sr // 2 or not np.isfinite(audio).all():
                raise RuntimeError("Model returned empty or invalid audio.")
            out = HOME / (uuid.uuid4().hex + ".wav")
            sf.write(str(out), np.clip(audio, -0.97, 0.97), sr)
        return str(out), str(out), (
            "Generated Telugu with F5 voice conditioning from the video. "
            "Similarity to the original voice is not guaranteed; check the result. "
            "A 5–10 second single-speaker section and accurate reference transcript help."
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error("Reference voice synthesis failed: " + str(exc)[:650]) from exc

with gr.Blocks(title="Telugu Child Voice Reference Studio") as demo:
    gr.Markdown("# 🎙️ Telugu Child Voice Reference Studio")
    gr.Markdown(
        "Your chosen speaking reference → new Telugu dialogue. "
        "No Sarvam API or adult pitch shifting. Uses ungated F5-TTS Telugu reference conditioning. "
        "**Voice similarity is experimental, not exact reproduction.**"
    )
    with gr.Row():
        with gr.Column():
            source = gr.Video(label="Reference video with clear speech (or audio file)", sources=["upload"])
            # Separate audio upload supports WAV/MP3/Ogg without forcing a video.
            audio_source = gr.Audio(label="Or upload reference audio", sources=["upload"], type="filepath")
            use_audio = gr.Checkbox(label="Use the audio file instead of video", value=False)
            start = gr.Slider(0, 30, value=2, step=0.5, label="Reference starts at (seconds)")
            duration = gr.Slider(3, 12, value=8, step=0.5, label="Speaking-reference length")
            transcript = gr.Textbox(
                label="Exact words spoken in the reference (recommended)",
                placeholder="Leave empty for automatic transcription (may be inaccurate)",
                lines=2,
            )
            authorized = gr.Checkbox(label="I have permission to synthesize this voice, including guardian consent if applicable")
            dialogue = gr.Textbox(label="New Telugu dialogue", lines=5,
                value="అమ్మా! చూడు! నేను ఒక చిన్న డ్రాగన్‌ని చూశాను!")
            steps = gr.Slider(8, 32, value=16, step=4, label="Inference steps")
            speed = gr.Slider(0.85, 1.15, value=1.0, step=0.05, label="Delivery speed")
            button = gr.Button("Generate Telugu in reference voice", variant="primary")
        with gr.Column():
            player = gr.Audio(label="Listen", type="filepath")
            download = gr.File(label="Download WAV")
            details = gr.Textbox(label="Status", interactive=False)

    def submit(text, video, audio, prefer_audio, start, duration, transcript, steps, speed, permission):
        return generate(text, audio if prefer_audio else video, start, duration,
                        transcript, steps, speed, permission)

    button.click(submit,
        inputs=[dialogue, source, audio_source, use_audio, start, duration, transcript, steps, speed, authorized],
        outputs=[player, download, details], api_name="generate_telugu_reference_voice")
    gr.Markdown("Model: [F5-TTS OpenBible Telugu](https://huggingface.co/multilingual-tts/F5-TTS-OpenBible-Telugu) "
                "(CC BY-SA 4.0). CPU inference may be slow. Do not use unauthorized real people's voices.")
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
