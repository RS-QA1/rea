"""Child-character Telugu voice experiment: helium acoustics, no gated weights."""
import os
import re
import wave
import uuid
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache

import gradio as gr
from huggingface_hub import snapshot_download

ROOT=Path("/tmp/child_character_helium")
ROOT.mkdir(parents=True, exist_ok=True)
ENGINE_REPO="sulabhkatiyar/indic-tiny-tts-telugu"
CHARACTERS={
    "Chinni — bright cartoon girl": ("female", 3.0, 1.00),
    "Bujji — little soft character": ("female", 2.0, 0.96),
    "Chintu — playful cartoon boy": ("male", 4.0, 1.03),
    "Balu — excited little character": ("male", 3.0, 1.01),
}
STYLE={
    "Natural cartoon":1.0,
    "Excited / playful":1.08,
    "Shy / gentle":0.95,
    "Surprised":1.06,
}
@lru_cache(maxsize=1)
def model():
    import sys
    folder=snapshot_download(ENGINE_REPO,ignore_patterns=["*.safetensors","samples/*"])
    if folder not in sys.path:
        sys.path.insert(0,folder)
    from indic_tiny_tts import TinyTTSOnnx
    return TinyTTSOnnx.from_pretrained(folder)

def ffmpeg(args):
    cmd=["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y"]+args
    result=subprocess.run(cmd,check=False,text=True,capture_output=True,timeout=100)
    if result.returncode:
        raise RuntimeError("Audio effect failed: "+result.stderr[-300:])

def synthesize(text,character,mode,delivery,strength):
    text=(text or "").strip()
    if not text or len(text)>1200:
        raise gr.Error("Enter 1–1200 characters of Telugu dialogue.")
    if character not in CHARACTERS or delivery not in STYLE:
        raise gr.Error("Select a supported character and delivery.")
    # Speech model does not follow arbitrary directions.
    # Remove simple laugh markers instead of reading them aloud.
    text=re.sub(r"\\[(?:giggle|laugh|hehe|haha)\\]","",text,flags=re.I).strip()
    if not text:
        raise gr.Error("Enter spoken Telugu dialogue.")
    base_voice,semitones,pace=CHARACTERS[character]
    semitones=min(6.0,max(0.0,semitones+float(strength)))
    factor=2**(semitones/12)
    rate_factor=STYLE[delivery]*pace
    engine=model()
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        raw=Path(folder)/"original.wav"
        engine.save_wav(engine.synthesize(text,voice=base_voice,speed=1.0),str(raw))
        output=ROOT/(uuid.uuid4().hex+".mp3")
        if mode=="Helium cartoon":
            # Intentionally shift both pitch AND vocal resonances, shortening
            # duration like a helium voice (no atempo to reverse pitch shift).
            # Small 2–6-semitone changes minimize distortion.
            af=(f"aresample=32000,asetrate=32000*{factor:.8f},"
                f"aresample=32000,atempo={rate_factor:.7f},"
                "highpass=f=85,alimiter=limit=0.92:level=0")
        elif mode=="Soft cartoon":
            # Milder pitch shift: keep character articulation closer to source.
            mild=2**(min(2.5,semitones*.5)/12)
            af=(f"aresample=32000,asetrate=32000*{mild:.8f},"
                f"aresample=32000,atempo={rate_factor:.7f},"
                "alimiter=limit=0.92:level=0")
        else:
            raise gr.Error("Unsupported voice effect mode.")
        ffmpeg(["-i",str(raw),"-af",af,"-ar","32000","-b:a","192k","-c:a","libmp3lame",str(output)])
        if output.stat().st_size<2000:
            raise RuntimeError("Output was empty.")
    return str(output),str(output),(
        f"{character}; {mode}; pitch {semitones:.1f} semitones; {delivery}. "
        "This is a cartoon/helium voice effect on an adult-trained Telugu base, "
        "not a child-trained model. Laugh markers are not synthesized.")

with gr.Blocks(title="Telugu Cartoon Child Voice Lab") as demo:
    gr.Markdown("# 🎈 Telugu Cartoon Child Voice Lab")
    gr.Markdown("**Only childlike character presets are shown.** Test a real helium-style "
                "pitch/formant effect on Telugu dialogue, with no paid API. "
                "These are synthetic cartoon approximations, not authentic children's voices.")
    with gr.Row():
        with gr.Column():
            dialogue=gr.Textbox(label="తెలుగు డైలాగ్",lines=5,
                value="అమ్మా! చూడు! నేను ఒక చిన్న డ్రాగన్‌ని చూశాను!")
            who=gr.Dropdown(list(CHARACTERS),value="Chintu — playful cartoon boy",
                            label="Cartoon character")
            effect=gr.Radio(["Helium cartoon","Soft cartoon"],value="Helium cartoon",
                            label="Voice effect")
            delivery=gr.Dropdown(list(STYLE),value="Natural cartoon",label="Delivery approximation")
            strength=gr.Slider(-1.0,2.0,value=0.0,step=0.5,
                               label="Extra helium strength (semitones)")
            btn=gr.Button("Generate cartoon voice",variant="primary")
        with gr.Column():
            preview=gr.Audio(label="Listen",type="filepath")
            download=gr.File(label="Download MP3")
            details=gr.Textbox(label="Voice details",interactive=False)
    btn.click(synthesize,[dialogue,who,effect,delivery,strength],
              [preview,download,details],api_name="generate_cartoon_voice")
    gr.Markdown("Underlying TTS: [Indic Tiny TTS Telugu](https://huggingface.co/sulabhkatiyar/indic-tiny-tts-telugu). "
                "No child reference recordings are used. Genuine child-voice conversion requires "
                "an appropriately licensed, consented speaking reference or child-trained model.")
if __name__=="__main__":
    demo.launch(server_name="0.0.0.0",server_port=7860,show_error=True)
