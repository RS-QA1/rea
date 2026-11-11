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
    (root / "README.md").write_bytes((source / "SPACE_README.md").read_bytes())
    metadata = (root / "README.md").read_text(encoding="utf-8")
    if not metadata.startswith("---\n") or "\nsdk: docker\n" not in metadata or "\napp_port: 7860\n" not in metadata:
        raise RuntimeError("Docker metadata invalid; refusing upload")
    if not (root / "Dockerfile").is_file():
        raise RuntimeError("Dockerfile missing")
    # Publish the card first so Hugging Face can classify this as a Docker Space.
    api.upload_file(path_or_fileobj=str(root / "README.md"), path_in_repo="README.md",
                    repo_id=space_id, repo_type="space",
                    commit_message="Set Docker SDK and port in Space metadata")
    api.upload_folder(folder_path=str(root), repo_id=space_id,
                      repo_type="space", commit_message="Deploy experimental reference voice UI")
info = api.space_info(space_id)
card = getattr(info, "card_data", None)
sdk = card.get("sdk") if isinstance(card, dict) else getattr(card, "sdk", None)
if sdk != "docker":
    raise RuntimeError(f"Published Space SDK was {sdk!r}, expected 'docker'.")
print(f"VERIFIED_SDK={sdk}")
print(f"REFERENCE_SPACE=https://huggingface.co/spaces/{space_id}")
