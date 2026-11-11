"""Publish just the Telugu Gradio app to the authenticated HF user's Space."""
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    raise SystemExit("Missing HF_TOKEN GitHub Actions secret.")
api = HfApi(token=token)
who = api.whoami()
owner = who.get("name")
if not owner:
    raise RuntimeError("Unable to determine Hugging Face account name.")
space_id = f"{owner}/telugu-dialogue-voice-studio"
api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio",
                private=False, exist_ok=True)
source = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as folder:
    dst = Path(folder)
    for filename in ("app.py", "requirements.txt", "packages.txt"):
        (dst / filename).write_bytes((source / filename).read_bytes())
    (dst / "README.md").write_text(
        "---\ntitle: Telugu Dialogue Voice Studio\nemoji: 🎤\nsdk: gradio\n"
        "app_file: app.py\npython_version: '3.11'\n"
        "---\n\nPaste Telugu dialogue and download MP3 audio.\n",
        encoding="utf-8",
    )
    api.upload_folder(folder_path=str(dst), repo_id=space_id, repo_type="space",
                      commit_message="Deploy Telugu Dialogue Voice Studio")
print(f"DEPLOYED_SPACE=https://huggingface.co/spaces/{space_id}")
print("Check the Space build logs and App tab before treating it as live.")
