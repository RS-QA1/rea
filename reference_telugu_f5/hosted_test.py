"""Verify hosted F5 Telugu reference UI and real synthesis using synthetic reference."""
import os
import time
import subprocess
import urllib.request
from pathlib import Path
from huggingface_hub import HfApi
from gradio_client import Client, handle_file

api=HfApi(token=os.environ["HF_TOKEN"])
owner=api.whoami()["name"]
space=f"{owner}/telugu-video-reference-voice"
card=api.space_info(space).card_data
sdk=card.get("sdk") if isinstance(card,dict) else getattr(card,"sdk",None)
assert sdk=="docker",f"Unexpected SDK {sdk!r}"
stage=None
for attempt in range(90):
    stage=str(api.get_space_runtime(space).stage)
    print("STAGE",stage,flush=True)
    if stage=="RUNNING":break
    if stage in ("RUNTIME_ERROR","BUILD_ERROR","CONFIG_ERROR"):
        raise RuntimeError(f"Hosted Space failed: {stage}")
    time.sleep(6)
assert stage=="RUNNING",f"Space not ready: {stage}"

url=f"https://{owner.lower()}-telugu-video-reference-voice.hf.space/"
for _ in range(20):
    try:
        with urllib.request.urlopen(url,timeout=20) as res:
            if res.status==200:break
    except Exception as exc:
        print("HTTP_STARTUP",repr(exc)[:150],flush=True)
    time.sleep(5)
else:
    raise RuntimeError("Hosted app did not serve HTTP 200")
print("HOSTED_UI_READY",url,flush=True)

ref=Path("/tmp/hosted_f5_reference.wav")
subprocess.run(["espeak-ng","-v","te","-s","110","-w",str(ref),
    "నమస్కారం. నేను చిన్న కథ చెబుతాను. ఇది నా మాట్లాడే స్వరం."],check=True)
client=Client(f"{owner}/telugu-video-reference-voice")
result=client.predict(
    "అమ్మా! నాకు ఒక కథ చెప్పు!",
    None,handle_file(str(ref)),True,0,8,
    "నమస్కారం. నేను చిన్న కథ చెబుతాను. ఇది నా మాట్లాడే స్వరం.",8,1.0,True,
    api_name="/generate_telugu_reference_voice",
)
file=result[0] if isinstance(result,(tuple,list)) else result
if isinstance(file,dict):file=file.get("path") or file.get("name")
file=Path(str(file))
assert file.exists() and file.stat().st_size>4000,"No hosted output WAV"
import wave
with wave.open(str(file),"rb") as wav:
    duration=wav.getnframes()/wav.getframerate()
assert duration>.5,f"Invalid duration {duration}"
print("HOSTED_F5_TELUGU_VOICE_PASS",round(duration,2),"seconds",flush=True)
