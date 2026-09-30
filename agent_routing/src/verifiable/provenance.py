"""Freeze executable harness identity, not just user-supplied hyperparameters."""
from pathlib import Path
import hashlib
import importlib.metadata

from .protocol import PROTOCOL_VERSION


def harness_identity():
    root = Path(__file__).parent
    files = sorted([*root.glob("*.py"), *root.glob("*.jinja")])
    # The shared selector and I/O also affect labels and saved records.
    files += [root.parent / "manager/marginal_value.py", root.parent / "utils/io.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root.parent)).encode() + b"\0" + path.read_bytes())
    packages = {}
    # flash-linear-attention and causal-conv1d switch Qwen3.5's linear-attention kernels when importable.
    for name in ("torch", "transformers", "trl", "peft", "datasets", "math-verify", "numpy",
                 "flash-linear-attention", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"protocol_version": PROTOCOL_VERSION, "source_sha256": digest.hexdigest(), "packages": packages}
