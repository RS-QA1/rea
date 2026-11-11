---
title: Telugu Reference Voice v2
emoji: "🎙️"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Telugu Reference Voice v2

Independent reference-based Telugu voice demo, Docker deployment, DhVaani model. Only reference speakers with authorization. For child voices, obtain guardian consent. Model CC BY-NC 4.0, so noncommercial testing only. A laugh-only clip is not enough to synthesize that speaker's speech.

## Built-in voice presets

The app now prepares two synthetic Telugu speaking references automatically using [Indic Tiny TTS Telugu](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-telugu): female and male. Select one and generate dialogue without providing a recording or transcript. These are adult-style TTS voices, **not actual six-year-old voices**. Model/code and dataset attributions are listed on the source model card (CC BY 4.0 for acoustic weights/Rasa data; MIT for code and vocoder). DhVaani output remains subject to its own noncommercial model license and gated model access restrictions. The separate upload option is for authorized custom voices only.
