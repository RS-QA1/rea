"""Deploy reference-conditioned Telugu F5 test app with Docker metadata."""
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

token=os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN GitHub secret missing")
api=HfApi(token=token)
name=f'{api.whoami()["name"]}/telugu-video-reference-voice'
api.create_repo(repo_id=name,repo_type="space",space_sdk="docker",
                private=False,exist_ok=True)
source=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as folder:
    dst=Path(folder)
    for fn in ("app.py","requirements.txt","Dockerfile"):
        (dst/fn).write_bytes((source/fn).read_bytes())
    (dst/"README.md").write_bytes((source/"SPACE_README.md").read_bytes())
    card=(dst/"README.md").read_text(encoding="utf-8")
    assert "\nsdk: docker\n" in card and "\napp_port: 7860\n" in card
    api.upload_folder(folder_path=folder,repo_id=name,repo_type="space",
                      commit_message="Deploy Telugu voice reference F5")
    info=api.space_info(name)
    metadata=info.card_data
    sdk=metadata.get("sdk") if isinstance(metadata,dict) else getattr(metadata,"sdk",None)
    assert sdk=="docker",f"Unexpected SDK {sdk}"
print(f"APP_URL=https://huggingface.co/spaces/{name}")
