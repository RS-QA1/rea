---
title: Telugu and Child Voice Studio
emoji: "🎙️"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Telugu + Child Voice Studio

One interface for three free-to-test speech model backends.

- **Indic Tiny TTS Telugu**: Female and male Telugu, CPU/ONNX, adult-trained.
- **Piper Telugu**: Padmavathi and Venkatesh, CPU.
- **Child-trained XTTS English (experimental)**: An English voice fine-tuned on child speech, hosted separately at [Child Voice Model Lab](https://huggingface.co/spaces/Revankruthi/child-voice-model-lab) to isolate its 2.3 GB weights. It is **English-only** and may be slow on free CPU hardware.

The interface provides text input, output preview/download, basic pace controls, and optional separately recorded laugh segments. It does **not** yet generate native Telugu children's speech or natural emotional acting.

No Sarvam API or gated DhVaani model is needed. The English child voice model is a research checkpoint. Check its base model and source-recording licenses before commercial or monetized use; the repository's MIT label does not necessarily override the underlying Coqui model license.

Sources:
- https://huggingface.co/sulabhkatiyar/indic-tiny-tts-telugu
- https://huggingface.co/rhasspy/piper-voices
- https://huggingface.co/Aniket2003333333/xtts-v2-child-voice-finetuned
