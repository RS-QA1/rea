# Telugu Character Voice Studio (Independent API v0.2)

Browser app: paste dialogue in **native Telugu script**, choose a boy/girl character profile, generate and download MP3.

Open model: [Indic Tiny TTS Telugu](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-telugu) by Sulabh Katiyar. Uses ONNX CPU inference. The underlying male/female voices were trained on adult Rasa speech data. The character presets modify pitch, resonance and tempo but do not constitute unique child-trained voice identities. They are **not** guaranteed to sound like 6-year-old children.

### Custom API — no Sarvam required
- `GET /api/health` readiness details (does not load model)
- `GET /api/voices` available profiles
- `POST /api/generate` with JSON `{"text":"హాయ్!","character":"Chintu · energetic boy (~6 style)","tempo":1.0,"pitch_offset":0.0}` responds with MP3 bytes.
- `/docs` OpenAPI docs may be accessible depending on Gradio mount routing.

### Deploy
GitHub workflow copies `app.py`, `requirements.txt`, and `packages.txt` into the Hugging Face Gradio Space root. First generation downloads the ONNX model, which may take additional startup time. Space requires FFmpeg (`packages.txt`).

### Reference
An uploaded reference video helped specify the target (younger, brighter and more animated speech), but the video is **not** used for cloning or uploaded to the Space. Matching a real six-year-old timbre needs licensed, child-trained data and a different model or properly consented reference-speaker training. Independent model inference replaces the earlier gTTS/Sarvam API dependency.

### Licenses
Acoustic model CC-BY-4.0, Vocos vocoder MIT, model code MIT; see the original model card and Rasa attribution.
