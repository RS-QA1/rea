"""Publish independent cartoon/helium Telugu voice Space only after tests pass."""
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

api=HfApi(token=os.environ["HF_TOKEN"])
owner=api.whoami()["name"]
space=f"{owner}/telugu-cartoon-child-voice"
api.create_repo(repo_id=space,repo_type="space",space_sdk="docker",
                private=False,exist_ok=True)
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as temp:
    dst=Path(temp)
    for name in ("app.py","requirements.txt","Dockerfile"):
        (dst/name).write_bytes((root/name).read_bytes())
    (dst/"README.md").write_bytes((root/"SPACE_README.md").read_bytes())
    meta=(dst/"README.md").read_text(encoding="utf-8")
    if "\nsdk: docker\n" not in meta or "\napp_port: 7860\n" not in meta:
        raise ValueError("Invalid Docker Space card")
    api.upload_folder(folder_path=temp,repo_id=space,repo_type="space",
                      commit_message="Deploy Telugu cartoon helium character voice lab")
card=api.space_info(space).card_data
sdk=card.get("sdk") if isinstance(card,dict) else getattr(card,"sdk",None)
assert sdk=="docker",f"Wrong Space SDK {sdk}"
print("APP_URL=https://huggingface.co/spaces/"+space)
