import ast
from pathlib import Path

app = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
ast.parse(app)
assert "PiperVoice.load" in app
assert "synthesize_wav" in app
assert "gr.Audio" in app
assert "gr.File" in app
print("Static checks passed; synthesis not run")
