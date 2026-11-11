# Child Voice Studio — v0.1

An independent, CPU-friendly English text-to-speech character voice prototype. It is **not Sarvam AI** and does not use, reproduce, or extract its proprietary model weights. The childlike effect comes from pitch/tempo processing of adult-trained open-source Piper voices, not true child-trained models.

## Run on Windows

1. Install Python 3.11 or 3.12 and FFmpeg. In PowerShell, you can install FFmpeg with `winget install Gyan.FFmpeg`. Restart PowerShell and verify with `ffmpeg -version`.
2. In this `child_voice_studio` directory run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

3. Open http://127.0.0.1:7860. First voice generation downloads public Piper model files from Hugging Face. WAV files are saved in `outputs/` and exposed in the app.

## Deploy on Hugging Face

Create a Gradio Space and copy `app.py`, `requirements.txt`, and `packages.txt` to the Space root. This repository branch does **not** automatically deploy the app.

## Limitations

- English only. Telugu, Hindi, natural child-trained voices, emotion controls, multi-character dialogues and singing are not implemented.
- Piper voice/model licenses must be checked separately for commercial use.
- Actual model inference and sound quality have not been verified in this environment.
- No child's reference voice or private Sarvam API is needed.

## Privacy

Scripts are processed locally after downloading the voice model. No voice samples of real children are collected.
