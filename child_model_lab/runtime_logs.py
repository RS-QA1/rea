import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
space=f'{api.whoami()["name"]}/child-voice-model-lab'
print("STAGE",api.get_space_runtime(space).stage,flush=True)
try:
    lines=list(api.fetch_space_logs(repo_id=space,follow=False))
    text="".join(str(x) for x in lines)
    print("RUNTIME_LAST_LINES",flush=True)
    print("\n".join(text.splitlines()[-100:])[-12000:],flush=True)
except Exception as exc:
    print("LOG_FETCH_ERROR",type(exc).__name__,str(exc)[:600],flush=True)
