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
healthy = False
for attempt in range(18):
    try:
        request=urllib.request.Request(url,headers={"User-Agent":"Child-Voice-Model-Lab-Health/1.0"})
        with urllib.request.urlopen(request,timeout=20) as response:
            print("HTTP_STATUS",response.status,"attempt",attempt+1,flush=True)
            healthy = response.status == 200
        if healthy:
            break
    except Exception as exc:
        print("HTTP_CHECK_ERROR",repr(exc)[:320],"attempt",attempt+1,flush=True)
    time.sleep(10)
if not healthy:
    raise RuntimeError("Hosted UI did not reach HTTP 200")
print("CHILD_SPACE_UI_READY",url)
