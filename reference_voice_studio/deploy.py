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
api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker",
                private=False, exist_ok=True)
source = Path(__file__).parent
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    for filename in ("app.py", "requirements.txt", "Dockerfile"):
        (root / filename).write_bytes((source / filename).read_bytes())
    (root / "README.md").write_text(
        "---\ntitle: Telugu Reference Voice Studio\nemoji: 🎭\nsdk: docker\n"
        "app_port: 7860\n"
        "---\n\nExperimental Telugu reference-conditioned TTS on a pinned Python environment.\n",
        encoding="utf-8",
    )
    metadata = (root / "README.md").read_text(encoding="utf-8")
    if not metadata.startswith("---\n") or "\nsdk: docker\n" not in metadata:
        raise RuntimeError("Invalid Docker Space README metadata; refusing upload")
    if not (root / "Dockerfile").exists():
        raise RuntimeError("Missing Dockerfile")
    api.upload_folder(folder_path=str(root), repo_id=space_id,
                      repo_type="space", commit_message="Deploy experimental reference voice UI")
print(f"REFERENCE_SPACE=https://huggingface.co/spaces/{space_id}")
