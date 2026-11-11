# Telugu Dialogue Voice Studio

Paste Telugu dialogue, generate audio, play and download MP3.

## Deploy on Hugging Face Spaces
Create a **Gradio / CPU Basic** Space. Copy `app.py`, `requirements.txt`, and `packages.txt` from this directory into the **root** of the new Space (not inside a subfolder). For the optional Sarvam voice provider, add `SARVAM_API_KEY` as a **Secret** in Settings; do not commit it to GitHub.

The free engine uses Google Translate TTS via gTTS and needs internet access. It is not child-trained and might sound artificial. Sarvam v3 provides Telugu speakers but none of these presets are guaranteed child voices. Pitch effect is an experimental approximation.

For a Space repository root, its README must include this metadata:

```yaml
---
title: Telugu Dialogue Voice Studio
emoji: 🎤
sdk: gradio
app_file: app.py
---
```

Free voice synthesis and deployment are not yet runtime-verified. For usage limitations, check Google and Sarvam service terms.
