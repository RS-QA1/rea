import os
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

VOICES = {"Girl": "female, child", "Boy": "male, child"}


@lru_cache(maxsize=1)
def load_model():
    import torch
    from omnivoice import OmniVoice

    torch.set_num_threads(2)
    return OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32,
    )


def generate(dialogue, voice):
    dialogue = dialogue.strip()
    if len(dialogue) > 300 or not any("అ" <= c <= "౿" for c in dialogue):
        raise gr.Error("Enter Telugu dialogue, up to 300 characters.")
    model = load_model()
    samples = np.asarray(model.generate(
        text=dialogue, language="te", instruct=VOICES[voice],
    )[0], dtype=np.float32).reshape(-1)
    if samples.size < 12000 or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 0.001:
        raise gr.Error("The model returned no usable speech.")
    with tempfile.NamedTemporaryFile(prefix="telugu-child-", suffix=".wav", delete=False) as output:
        path = Path(output.name)
    sf.write(path, samples, 24000)
    return str(path), str(path)


with gr.Blocks(title="Telugu Child Voice", delete_cache=(900, 3600)) as demo:
    gr.Markdown("# Telugu Child Voice")
    gr.Markdown("Generate a synthetic child voice directly from Telugu dialogue.")
    dialogue = gr.Textbox(label="Telugu dialogue", lines=4,
                          value="అమ్మా! ఇటు చూడు! మన తోటలో ఓ చిన్న డ్రాగన్ ఉంది!")
    voice = gr.Radio(list(VOICES), value="Girl", label="Voice")
    button = gr.Button("Generate voice", variant="primary")
    preview = gr.Audio(label="Listen")
    download = gr.File(label="Download WAV")
    button.click(generate, [dialogue, voice], [preview, download], api_name="generate")
    gr.Markdown("OmniVoice model weights are licensed CC BY-NC. Voice quality may vary by line.")

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
