import os,time,urllib.request
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
space=f'{api.whoami()["name"]}/child-voice-model-lab'
info=api.space_info(space)
card=info.card_data
sdk=card.get("sdk") if isinstance(card,dict) else getattr(card,"sdk",None)
print("SDK",sdk,flush=True)
assert sdk=="docker"
stage=""
for attempt in range(45):
    stage=str(api.get_space_runtime(space).stage)
    print("Stage",stage,"attempt",attempt+1,flush=True)
    if stage=="RUNNING":
        break
    if stage in ("BUILD_ERROR","RUNTIME_ERROR","CONFIG_ERROR","PAUSED"):
        raise RuntimeError("Space failed: "+stage)
    time.sleep(8)
if stage!="RUNNING":
    raise RuntimeError("Space not RUNNING: "+stage)
url=f'https://{api.whoami()["name"].lower()}-child-voice-model-lab.hf.space/'
with urllib.request.urlopen(url,timeout=30) as response:
    print("HTTP_STATUS",response.status,flush=True)
    assert response.status==200
print("CHILD_SPACE_UI_READY",url)
