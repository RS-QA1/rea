"""Test the actual hosted child speech API before wiring it to the main UI."""
from pathlib import Path
import wave
from gradio_client import Client

client=Client("Revankruthi/child-voice-model-lab")
result=client.predict(
    "Mom! Look! I found a dragon!",
    0.8,
    api_name="/generate_child_voice",
)
path=result[0] if isinstance(result,(list,tuple)) else result
if isinstance(path,dict):
    path=path.get("path") or path.get("name")
file=Path(str(path))
assert file.is_file() and file.stat().st_size>10240, "No voice file downloaded"
with wave.open(str(file),"rb") as sample:
    duration=sample.getnframes()/sample.getframerate()
    assert duration>0.5, f"Invalid child voice duration: {duration}"
print("HOSTED_CHILD_VOICE_API_PASS",round(duration,2),"seconds")
