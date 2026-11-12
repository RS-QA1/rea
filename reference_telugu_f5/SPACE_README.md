---
title: Telugu Child Voice Reference Studio
emoji: "🎙️"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Reference-conditioned Telugu TTS

Generates new Telugu sentences in the style of an authorized reference speaking voice, using [F5-TTS OpenBible Telugu](https://huggingface.co/multilingual-tts/F5-TTS-OpenBible-Telugu), CC BY-SA 4.0.

Upload a 5–10 second reference section of clean speech from your own video, preferably with an accurate transcript. The app can auto-transcribe if omitted, but accuracy and cross-language voice similarity are unverified. This is not exact cloning or an authentic child-trained voice model. No reference files are committed to any public repo.

Test access/compute before production deployment. CPU model may exceed Hugging Face free CPU memory or take considerable inference time. For minors' voices, appropriate guardian permission is required.
