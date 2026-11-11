"""Deploy independent reference-voice test Space without changing production Studio."""
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")
if not token:
    raise SystemExit("HF_TOKEN secret required")
api = HfApi(token=token)
owner = api.whoami()["name"]
space_id = f"{owner}/telugu-reference-voice-studio"
api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio",
                private=False, exist_ok=True)
source = Path(__file__).parent
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    for filename in ("app.py", "requirements.txt", "packages.txt"):
        (root / filename).write_bytes((source / filename).read_bytes())
    (root / "README.md").write_text(
        "---\ntitle: Telugu Reference Voice Studio\nemoji: 🎭\nsdk: gradio\n"
        "app_file: app.py\npython_version: '3.11'\n"
        "---\n\nExperimental Telugu reference-conditioned TTS.\n",
        encoding="utf-8",
    )
    api.upload_folder(folder_path=str(root), repo_id=space_id,
                      repo_type="space", commit_message="Deploy experimental reference voice UI")
print(f"REFERENCE_SPACE=https://huggingface.co/spaces/{space_id}")
