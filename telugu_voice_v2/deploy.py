"""Publish a fresh Docker Space only after local Docker checks pass."""
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("Missing HF_TOKEN GitHub Actions secret.")
api = HfApi(token=token)
owner = api.whoami()["name"]
space = f"{owner}/telugu-voice-reference-v2"
source = Path(__file__).resolve().parent
api.create_repo(repo_id=space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

with tempfile.TemporaryDirectory() as d:
    dst = Path(d)
    for file in ["app.py", "Dockerfile", "requirements.txt"]:
        (dst/file).write_bytes((source/file).read_bytes())
    (dst/"README.md").write_bytes((source/"SPACE_README.md").read_bytes())
    metadata = (dst/"README.md").read_text(encoding="utf-8")
    assert metadata.startswith("---\n"), "Broken README frontmatter"
    assert "\nsdk: docker\n" in metadata, "Wrong SDK"
    assert "\napp_port: 7860\n" in metadata, "Wrong app port"
    api.upload_folder(folder_path=d, repo_id=space, repo_type="space",
                      commit_message="Clean Docker voice reference v2")
info = api.space_info(space)
card = info.card_data
sdk = card.get("sdk") if isinstance(card, dict) else getattr(card, "sdk", None)
if sdk != "docker":
    raise RuntimeError(f"Unexpected hosted SDK: {sdk!r}")
print(f"SPACE_URL=https://huggingface.co/spaces/{space}")
print("SPACE_SDK=docker")
print("Model inference is not validated by a deployment commit.")
