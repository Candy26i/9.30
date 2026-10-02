"""Advisor LoRA copies that vLLM actually applies on the Qwen3.5 multimodal architecture.

The trap. The advisors were trained on ``Qwen3_5ForCausalLM`` (``load_text_causal_model``),
so their PEFT keys are ``base_model.model.model.layers.N.<module>.lora_{A,B}.weight``.
vLLM serves ``Qwen/Qwen3.5-9B`` as ``Qwen3_5ForConditionalGeneration`` (the paper logs:
``Resolved architecture: Qwen3_5ForConditionalGeneration``, vLLM 0.26.0, no override),
whose language-model modules are ``language_model.model.layers.N.<module>``. vLLM maps a
LoRA tensor name by stripping ``base_model.model.`` and applying the model's
``hf_to_vllm_mapper`` (``vllm/lora/utils.py::parse_fine_tuned_lora_name``); the
Qwen3-VL mapper only rewrites ``model.language_model.`` -> ``language_model.model.``
(``qwen3_vl.py``; ``Qwen3_5ForConditionalGeneration`` inherits it), so
``model.layers.N.*`` names match no module. The adapter "loads" (``Loaded new LoRA
adapter``), the request is served by the plain base, and nothing above debug level says so.

The fix used here keeps the paper's architecture and flags and renames the keys of a
served copy to the multimodal layout, ``base_model.model.model.language_model.layers.N.*``,
which the same mapper turns into ``language_model.model.layers.N.*``. Only the
safetensors header changes: the tensor data section is byte-identical, and
``verify_served_lora`` proves it against the pinned source (``advisors.check_server``
accepts a served root only if its weights are the pinned file or such a verified copy).
``preflight`` then checks behaviourally that the LoRA changes outputs and reproduces the
recorded paper outputs.

Standard library only (the vLLM venv and the experiment venv can both run it).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

SERVING_VERSION = "mcq_rsi_served_lora/1"
PROVENANCE = "mcq_rsi_served_lora.json"
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
TEXT_PREFIX = "base_model.model.model.layers."
MULTIMODAL_PREFIX = "base_model.model.model.language_model.layers."
MODES = ("multimodal", "as_is")  # as_is: the paper-era serving (keys unmatched on the multimodal model)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path) -> Tuple[Dict[str, Any], int]:
    """(header dict, data offset) of a safetensors file."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100 * 1024 * 1024:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        header = json.loads(f.read(n).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return header, 8 + n


def data_sha256(path) -> str:
    """sha256 of the tensor data section (everything after the header)."""
    _, offset = read_header(path)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rename_key(name: str) -> str:
    if name == "__metadata__":
        return name
    if name.startswith(MULTIMODAL_PREFIX):
        return name
    if not name.startswith(TEXT_PREFIX):
        raise ValueError(f"unexpected LoRA tensor name {name!r} (expected {TEXT_PREFIX}...)")
    return MULTIMODAL_PREFIX + name[len(TEXT_PREFIX):]


def renamed_header(header: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, entry in header.items():
        new = rename_key(name)
        if new in out:
            raise ValueError(f"rename collision on {new}")
        out[new] = entry
    return out


def _encode_header(header: Dict[str, Any]) -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)  # the safetensors writer aligns the data section to 8 bytes
    return struct.pack("<Q", len(raw)) + raw


def write_renamed(src, dest) -> Dict[str, Any]:
    """Copy ``src`` safetensors to ``dest`` with the multimodal key layout; tensor bytes untouched."""
    header, offset = read_header(src)
    new = renamed_header(header)
    dest = Path(dest)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as f:
            out.write(_encode_header(new))
            f.seek(offset)
            shutil.copyfileobj(f, out, 1 << 20)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {"n_tensors": sum(k != "__metadata__" for k in header),
            "renamed": sum(rename_key(k) != k for k in header)}


def prepare_served_lora(source_dir, dest_dir, mode: str = "multimodal", expected_sha256: str = "") -> Dict[str, Any]:
    """Write ``dest_dir`` (adapter_config.json + renamed weights + provenance) for vLLM ``--lora-modules``.

    ``mode="as_is"`` returns the source directory itself (paper-era serving, for diagnosis).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    source_dir, dest_dir = Path(source_dir), Path(dest_dir)
    src = source_dir / WEIGHTS
    source_sha = file_sha256(src)
    if expected_sha256 and source_sha != expected_sha256:
        raise ValueError(f"{src}: sha256 {source_sha} differs from the pinned {expected_sha256}")
    if mode == "as_is":
        return {"mode": mode, "served_root": str(source_dir), "source_sha256": source_sha}
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dir / CONFIG, dest_dir / CONFIG)
    counts = write_renamed(src, dest_dir / WEIGHTS)
    provenance = {
        "version": SERVING_VERSION, "mode": mode, "source_dir": str(source_dir.resolve()),
        "source_sha256": source_sha, "source_data_sha256": data_sha256(src),
        "served_sha256": file_sha256(dest_dir / WEIGHTS), "rename": {"from": TEXT_PREFIX, "to": MULTIMODAL_PREFIX},
        "adapter_config_sha256": file_sha256(dest_dir / CONFIG), **counts,
    }
    (dest_dir / PROVENANCE).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**provenance, "served_root": str(dest_dir)}


def verify_served_lora(served_root, pinned_sha256: str) -> Dict[str, Any]:
    """Raise unless ``served_root`` holds the pinned adapter's tensors under the multimodal key layout.

    Checks the provenance, that its source file still has the pinned sha256, that the
    served header is exactly the renamed source header (dtype, shape and offsets), and
    that both data sections hash identically, i.e. the served weights are bitwise the pinned ones.
    """
    root = Path(served_root)
    path = root / PROVENANCE
    if not path.is_file():
        raise ValueError(f"{root}: no {PROVENANCE} (not a prepared served LoRA)")
    prov = json.loads(path.read_text(encoding="utf-8"))
    if prov.get("version") != SERVING_VERSION or prov.get("mode") != "multimodal":
        raise ValueError(f"{path}: unknown served-LoRA provenance {prov.get('version')}/{prov.get('mode')}")
    src = Path(prov["source_dir"]) / WEIGHTS
    if not src.is_file() or file_sha256(src) != pinned_sha256:
        raise ValueError(f"{root}: source {src} is missing or is not the pinned adapter {pinned_sha256}")
    served = root / WEIGHTS
    if file_sha256(served) != prov["served_sha256"]:
        raise ValueError(f"{served}: sha256 differs from its provenance")
    src_header, _ = read_header(src)
    served_header, _ = read_header(served)
    if renamed_header(src_header) != served_header:
        raise ValueError(f"{served}: header is not the renamed pinned header")
    if data_sha256(served) != data_sha256(src):
        raise ValueError(f"{served}: tensor data differs from the pinned adapter")
    if (root / CONFIG).read_bytes() != (Path(prov["source_dir"]) / CONFIG).read_bytes():
        raise ValueError(f"{root}: adapter_config.json differs from the pinned adapter's")
    return {"mode": "multimodal", "source_dir": prov["source_dir"], "n_tensors": prov["n_tensors"]}


def key_layout(path) -> Dict[str, int]:
    """Counts of tensor-name prefixes (evidence printed by the start script)."""
    header, _ = read_header(path)
    out: Dict[str, int] = {}
    for name in header:
        if name == "__metadata__":
            continue
        prefix = MULTIMODAL_PREFIX if name.startswith(MULTIMODAL_PREFIX) else (
            TEXT_PREFIX if name.startswith(TEXT_PREFIX) else name.split(".layers.")[0])
        out[prefix] = out.get(prefix, 0) + 1
    return out


def lora_modules(entries: List[Tuple[str, str]]) -> List[str]:
    """``name=path`` arguments for ``vllm serve --lora-modules``."""
    return [f"{name}={path}" for name, path in entries]
