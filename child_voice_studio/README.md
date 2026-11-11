# Telugu Character Voice Studio (Independent API v0.2)

Browser app: paste dialogue in **native Telugu script**, choose a boy/girl character profile, generate and download MP3.

Open model: [Indic Tiny TTS Telugu](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-telugu) by Sulabh Katiyar. Uses ONNX CPU inference. The underlying male/female voices were trained on adult Rasa speech data. The character presets modify pitch, resonance and tempo but do not constitute unique child-trained voice identities. They are **not** guaranteed to sound like 6-year-old children.

### Gradio API — no Sarvam required
The browser UI exposes a Gradio endpoint named `generate_voice`. Use Hugging Face Space **View API** to see its request schema and client usage. This release removes the separate FastAPI server that conflicted with Gradio on port 7860.

### Deploy
GitHub workflow copies `app.py`, `requirements.txt`, and `packages.txt` into the Hugging Face Gradio Space root. First generation downloads the ONNX model, which may take additional startup time. Space requires FFmpeg (`packages.txt`).

### Reference
An uploaded reference video helped specify the target (younger, brighter and more animated speech), but the video is **not** used for cloning or uploaded to the Space. Matching a real six-year-old timbre needs licensed, child-trained data and a different model or properly consented reference-speaker training. Independent model inference replaces the earlier gTTS/Sarvam API dependency.

### Licenses
Acoustic model CC-BY-4.0, Vocos vocoder MIT, model code MIT; see the original model card and Rasa attribution.


## Acting and actual laughter (v0.5)

The UI now accepts `[giggle]`, `[laugh]` and `[chuckle]` at exact positions in the Telugu script. These markers trigger a **real non-speech laughter recording**, not synthetic speech pronouncing "హి హి". An acting prompt such as "giggle before speaking" or "laugh at the end" also inserts a laugh marker. The spoken dialogue remains generated using the independent Telugu ONNX voice model.

Default sound credit: *Four Years Old child laughing.ogg* by morgantj, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Four_Years_Old_child_laughing.ogg), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The app downloads the sound on demand and uses a short excerpt; if that retrieval fails, upload a laughter recording you have permission to use.

You can upload a matching character's giggle in the UI; only use recordings you have the rights and consent to use. Uploaded clips are converted to 24 kHz mono with gentle fades and concatenated between speech segments.

**Known limitations:** Default laugh may not sound like the same actor as the spoken voice; emotions still adjust basic rate/pitch, not model-level delivery. No verification of Sarvam's proprietary laughter generation is implied. This is a practical sound-effect hybrid, not a new child-trained speech model.
