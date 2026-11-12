import os,requests,re
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
space=f'{api.whoami()["name"]}/telugu-video-reference-voice'
print("RUNTIME_STAGE",api.get_space_runtime(space).stage,flush=True)
url=f"https://huggingface.co/api/spaces/{space}/logs/run?tail=150"
with requests.get(url,headers={"Authorization":"Bearer "+os.environ["HF_TOKEN"]},stream=True,timeout=(10,20)) as r:
    print("LOG_HTTP",r.status_code,flush=True)
    r.raise_for_status()
    n=0
    for raw in r.iter_lines(decode_unicode=True):
        if raw and str(raw).startswith("data:"):
            line=re.sub(r"hf_[A-Za-z0-9_-]{15,}","[REDACTED_TOKEN]",str(raw)[5:])
            print(line[:2500],flush=True)
            n+=1
            if n>=150:break
