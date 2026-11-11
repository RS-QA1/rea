# Telugu Reference Voice Studio (experimental)

Separate from the existing working Telugu Voice Studio. Uses [DhVaani 0.5](https://huggingface.co/ARTPARK-IISc/DhVaani-0.5) for zero-shot reference-conditioned Telugu speech. Model card lists **CC BY-NC 4.0**, so use this prototype for noncommercial testing only without additional rights. Check reference-audio rights separately.

## How to test
1. Upload 5–15 seconds of clean **spoken** reference audio with clear Telugu speech.
2. Type an accurate transcript of exactly that reference speech.
3. Provide the new Telugu dialogue, choose a lightweight expression/speed, generate and listen.

A laugh-only recording (including Four Years Old child laughing.ogg) is not sufficient to obtain the child's speaking voice. Do not collect or clone a minor's voice without appropriate parental or guardian authorization.

This is not yet a true expressive speech model: emotion control affects speed, and real giggles are still inserted in the original Studio using separate licensed recordings. CPU inference may be slow or fail due to resource limits. Model and audio quality have not yet been validated in deployment.

## Model access requirement
The DhVaani model is **gated**, even though its model card is public. Open [the model card](https://huggingface.co/ARTPARK-IISc/DhVaani-0.5) while logged into your Hugging Face account and accept the access conditions. Create a fine-grained **read-only** Hugging Face access token with permission for this model and add it to the **Space Settings → Secrets** as `HF_MODEL_TOKEN`. Never paste a token into the app's dialogue or chat and don't commit it to GitHub.

A green GitHub deployment only means the Space source uploaded. It does not confirm model access, successful inference, or enough CPU/RAM for synthesis.
