"""Verify the actual hosted v2 Space without rebuilding Docker."""
import os
import time
import urllib.request
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("Missing HF_TOKEN")
api=HfApi(token=token)
owner=api.whoami()["name"]
space=f"{owner}/telugu-voice-reference-v2"
url=f"https://huggingface.co/spaces/{space}"
host=f"https://{owner.lower()}-telugu-voice-reference-v2.hf.space/"
info=api.space_info(space)
card=getattr(info,"card_data",None)
sdk=card.get("sdk") if isinstance(card,dict) else getattr(card,"sdk",None)
print("HOSTED_SDK",sdk,flush=True)
if sdk!="docker":
    raise SystemExit(f"Expected Docker SDK, got {sdk}")
stage="UNKNOWN"
response="NOT_CHECKED"
for i in range(30):
    stage=str(api.get_space_runtime(space).stage)
    print(f"CHECK {i+1}: {stage}",flush=True)
    if stage in ("BUILD_ERROR","RUNTIME_ERROR","CONFIG_ERROR"):
        break
    if stage=="RUNNING":
        try:
            req=urllib.request.Request(host,headers={"User-Agent":"TeluguVoiceStudio-v2-check"})
            with urllib.request.urlopen(req,timeout=12) as r:
                response=str(r.status)
            if response=="200":
                break
        except Exception as e:
            response=type(e).__name__+": "+str(e)[:140]
    time.sleep(8)
report=("# Telugu Reference Voice Studio v2 — verification\n\n"
    +f"- Space: {url}\n- SDK: {sdk}\n- Runtime stage: {stage}\n"
    +f"- Web UI HTTP result: {response}\n"
    +"\nThis verifies deployment and UI availability only. "
     "Model access, synthesis, reference-voice fidelity, and audio quality remain untested.\n")
Path("TELUGU_V2_STATUS.md").write_text(report,encoding="utf-8")
print(report,flush=True)
if stage!="RUNNING" or response!="200":
    raise SystemExit(f"Space/UI not healthy: {stage}, {response}")
