import os, requests
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
space=f'{api.whoami()["name"]}/child-voice-model-lab'
print("STAGE",api.get_space_runtime(space).stage,flush=True)
url=f"https://huggingface.co/api/spaces/{space}/logs/run?tail=120"
try:
    with requests.get(url,headers={"Authorization":"Bearer "+os.environ["HF_TOKEN"]},
                      stream=True,timeout=(10,25)) as r:
        print("LOGS_HTTP",r.status_code,flush=True)
        r.raise_for_status()
        count=0
        for line in r.iter_lines(decode_unicode=True):
            if line and str(line).startswith("data:"):
                text=str(line)[5:]
                # No runtime secrets should be logged; redact any token-looking strings.
                import re
                text=re.sub(r"hf_[A-Za-z0-9_\-]{15,}", "[REDACTED_TOKEN]",text)
                print(text[:2000],flush=True)
                count+=1
                if count>=120: break
except Exception as exc:
    print("LOG_FETCH_ERROR",type(exc).__name__,str(exc)[:400])
