"""Publish child-model lab only after GitHub Docker checks pass."""
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

api=HfApi(token=os.environ["HF_TOKEN"])
owner=api.whoami()["name"]
space=f"{owner}/child-voice-model-lab"
api.create_repo(repo_id=space, repo_type="space", space_sdk="docker",private=False,exist_ok=True)
root=Path(__file__).parent
with tempfile.TemporaryDirectory() as folder:
    target=Path(folder)
    for n in ("app.py","requirements.txt","Dockerfile"):
        (target/n).write_bytes((root/n).read_bytes())
    (target/"README.md").write_bytes((root/"SPACE_README.md").read_bytes())
    card=(target/"README.md").read_text()
    assert "\nsdk: docker\n" in card and "\napp_port: 7860\n" in card
    api.upload_folder(folder_path=folder,repo_id=space,repo_type="space",
                      commit_message="Deploy isolated child-trained XTTS prototype")
info=api.space_info(space)
card=info.card_data
sdk=card.get("sdk") if isinstance(card,dict) else getattr(card,"sdk",None)
assert sdk=="docker",f"Wrong SDK: {sdk}"
print("CHILD_MODEL_SPACE=https://huggingface.co/spaces/"+space)
