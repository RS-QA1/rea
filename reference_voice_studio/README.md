# Telugu Reference Voice Studio (experimental)

Separate from the existing working Telugu Voice Studio. Uses [DhVaani 0.5](https://huggingface.co/ARTPARK-IISc/DhVaani-0.5) for zero-shot reference-conditioned Telugu speech. Model license **CC BY-NC 4.0**: non-commercial evaluation only, not monetized YouTube use without appropriate licensing.

## How to test
1. Upload 5–15 seconds of clean **spoken** reference audio with clear Telugu speech.
2. Type an accurate transcript of exactly that reference speech.
3. Provide the new Telugu dialogue, choose a lightweight expression/speed, generate and listen.

A laugh-only recording (including Four Years Old child laughing.ogg) is not sufficient to obtain the child's speaking voice. Do not collect or clone a minor's voice without appropriate parental or guardian authorization.

This is not yet a true expressive speech model: emotion control affects speed, and real giggles are still inserted in the original Studio using separate licensed recordings. CPU inference may be slow or fail due to resource limits. Model and audio quality have not yet been validated in deployment.
