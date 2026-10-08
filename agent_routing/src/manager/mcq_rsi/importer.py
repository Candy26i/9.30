"""Fetch the pinned round-1 artifacts of one MCQ benchmark from HF.

Only the files named in the registry are requested: top-level adapter files
(never ``checkpoint-*``), single files of HF datasets, and named members of the
asset tarballs. Every file is checked against its registry digest before it is
moved into place, and ``import_manifest.json`` records what landed where.

Layout under ``<out>/<bench>/``::

    advisors/<kind>/...        extractor/reasoner/verifier LoRA (vLLM name <bench>_<kind>)
    round1/sft/...             S_1 manager adapter
    round1/labels.jsonl        static-arm label file S_1 was trained on
    round1/records.jsonl       round-1 counterfactual tree (collect_r1 roots)
    advisor_sft/<kind>.jsonl   advisor SFT data (prompt provenance, fresh-pool exclusion)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import benchmarks as registry
from .benchmarks import Adapter, Benchmark, HFFile, TarMember

IMPORT_VERSION = "mcq_rsi_import/1"
DEFAULT_OUT = "outputs/mcq_rsi/import"
PARTS = ("advisors", "manager", "data")
Downloader = Callable[..., str]


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, sha256: str = "", git_oid: str = "") -> str:
    if not (sha256 or git_oid):
        raise ValueError(f"{path}: no registry digest to check against")
    actual = _sha256_file(path)
    if sha256 and actual != sha256:
        raise ValueError(f"{path}: sha256 {actual} != {sha256}")
    if git_oid and git_blob_oid(path.read_bytes()) != git_oid:
        raise ValueError(f"{path}: git blob oid differs from {git_oid}")
    return actual


def _adapter_items(adapter: Adapter, dest: str, part: str) -> List[Dict[str, Any]]:
    items = []
    for f in adapter.hf_files():
        name = f.path.rsplit("/", 1)[-1]
        if "checkpoint-" in f.path or name not in registry.IMPORTABLE_FILES:
            raise ValueError(f"refusing to import {f.path}")
        items.append({"dest": f"{dest}/{name}", "source": f, "part": part})
    return items


def plan(bench: Benchmark, parts=PARTS) -> List[Dict[str, Any]]:
    if not parts or set(parts) - set(PARTS):
        raise ValueError(f"parts must be a non-empty subset of {PARTS}")
    items: List[Dict[str, Any]] = []
    for kind, adapter in bench.advisors:
        items += _adapter_items(adapter, f"advisors/{kind}", "advisors")
    items += _adapter_items(bench.round1_adapter, "round1/sft", "manager")
    data = [("round1/labels.jsonl", bench.round1_labels), ("round1/records.jsonl", bench.round1_records)]
    data += [(f"round1/{label}.jsonl", src) for label, src in bench.extra_sources]
    data += [(f"advisor_sft/{kind}.jsonl", member) for kind, member in bench.advisor_sft]
    items += [{"dest": dest, "source": src, "part": "data"} for dest, src in data]
    return [i for i in items if i["part"] in parts]


def describe(item: Dict[str, Any]) -> Dict[str, Any]:
    src = item["source"]
    out = {"dest": item["dest"], "part": item["part"], "uri": src.uri}
    if src.sha256:
        out["sha256"] = src.sha256
    if isinstance(src, HFFile) and src.git_oid:
        out["git_oid"] = src.git_oid
    if isinstance(src, TarMember):
        out["archive_sha256"] = src.archive.sha256
    return out


def _download(downloader: Downloader, f: HFFile, cache_dir: Optional[str]) -> Path:
    if "checkpoint-" in f.path:
        raise ValueError(f"refusing to download {f.path}")
    return Path(downloader(repo_id=f.repo_id, filename=f.path, repo_type=f.repo_type,
                           revision=f.revision, cache_dir=cache_dir))


def _commit(tmp: Path, dest: Path, sha256: str = "", git_oid: str = "") -> str:
    try:
        digest = _verify(tmp, sha256, git_oid)
    except ValueError:
        tmp.unlink()
        raise
    os.replace(tmp, dest)
    return digest


def _place(src_path: Path, dest: Path, sha256: str = "", git_oid: str = "") -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    shutil.copyfile(src_path, tmp)
    return _commit(tmp, dest, sha256, git_oid)


def _extract(archive: Path, members: List[Tuple[str, Path, str]]) -> Dict[str, str]:
    wanted = {name: (dest, sha) for name, dest, sha in members}
    done: Dict[str, str] = {}
    with tarfile.open(archive, "r:*") as tar:
        for info in tar:
            name = info.name[2:] if info.name.startswith("./") else info.name
            if name not in wanted or not info.isfile():
                continue
            dest, sha = wanted[name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            with tar.extractfile(info) as fin, open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            done[name] = _commit(tmp, dest, sha)
    missing = sorted(set(wanted) - set(done))
    if missing:
        raise ValueError(f"{archive}: members not found {missing}")
    return done


def run_import(
    bench: Benchmark,
    out: str,
    *,
    dry_run: bool = False,
    downloader: Optional[Downloader] = None,
    cache_dir: Optional[str] = None,
    parts=PARTS,
) -> Dict[str, Any]:
    registry.validate(bench)
    items = plan(bench, parts)
    root = Path(out) / bench.name
    if dry_run:
        return {"benchmark": bench.name, "dry_run": True, "root": str(root), "files": [describe(i) for i in items]}
    if downloader is None:
        from huggingface_hub import hf_hub_download as downloader
    records: List[Dict[str, Any]] = []
    archives: Dict[HFFile, List[Tuple[Dict[str, Any], Path]]] = {}
    for item in items:
        src, dest = item["source"], root / item["dest"]
        entry = describe(item)
        if dest.exists():
            try:
                entry["sha256"] = _verify(dest, src.sha256, getattr(src, "git_oid", ""))
                records.append({**entry, "status": "present"})
                continue
            except ValueError:
                dest.unlink()
        if isinstance(src, TarMember):
            archives.setdefault(src.archive, []).append((entry, dest))
            continue
        entry["sha256"] = _place(_download(downloader, src, cache_dir), dest, src.sha256, src.git_oid)
        records.append({**entry, "status": "downloaded"})
    for archive, pending in archives.items():
        local = _download(downloader, archive, cache_dir)
        _verify(local, archive.sha256)
        by_member = {e["uri"].split("#", 1)[1]: (e, d) for e, d in pending}
        digests = _extract(local, [(m, d, e["sha256"]) for m, (e, d) in by_member.items()])
        for member, (entry, _) in by_member.items():
            records.append({**entry, "sha256": digests[member], "status": "extracted"})
    if "advisors" in parts:
        for kind, _ in bench.advisors:
            check_adapter_config(root / f"advisors/{kind}/adapter_config.json")
    if "manager" in parts:
        check_adapter_config(root / "round1/sft/adapter_config.json")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "import_manifest.json"
    previous = json.loads(path.read_text(encoding="utf-8")).get("files", []) if path.exists() else []
    records += _carry_over(bench, root, previous, {r["dest"] for r in records})
    manifest = {
        "import_version": IMPORT_VERSION,
        "benchmark": bench.name,
        "base_model": registry.BASE_MODEL,
        "base_revision": registry.BASE_REVISION,
        "lora_names": {kind: bench.lora_name(kind) for kind, _ in bench.advisors},
        "files": sorted(records, key=lambda r: r["dest"]),
    }
    tmp = root / "import_manifest.json.part"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return manifest


def _carry_over(bench: Benchmark, root: Path, previous, fresh) -> List[Dict[str, Any]]:
    """Entries of earlier runs whose files still verify against the current registry."""
    planned = {i["dest"]: i for i in plan(bench)}
    kept = []
    for old in previous:
        item = planned.get(old["dest"])
        if old["dest"] in fresh or item is None or not (root / old["dest"]).is_file():
            continue
        src = item["source"]
        try:
            digest = _verify(root / old["dest"], src.sha256, getattr(src, "git_oid", ""))
        except ValueError:
            continue
        kept.append({**describe(item), "sha256": digest, "status": old["status"]})
    return kept


def check_adapter_config(path) -> Dict[str, Any]:
    """The paper LoRA shape: r16/alpha32 on q/k/v/o + MLP of Qwen/Qwen3.5-9B."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if cfg.get("base_model_name_or_path") != registry.BASE_MODEL or cfg.get("peft_type") != "LORA":
        raise ValueError(f"{path}: not a {registry.BASE_MODEL} LoRA")
    if cfg.get("r") != 16 or cfg.get("lora_alpha") != 32 or set(cfg.get("target_modules") or ()) != expected:
        raise ValueError(f"{path}: unexpected LoRA shape")
    return cfg


def default_paths(bench: Benchmark, root) -> Dict[str, Any]:
    base = Path(root) / bench.name
    return {
        "records": base / "round1/records.jsonl",
        "labels": base / "round1/labels.jsonl",
        # The records ``labels`` was balanced from (GPQA: the depth-1 collection, not ``records``).
        "label_records": base / ("round1/records.jsonl" if bench.label_records == "round1_records"
                                 else f"round1/{bench.label_records}.jsonl"),
        "advisor_sft": [base / f"advisor_sft/{kind}.jsonl" for kind, _ in bench.advisor_sft],
    }
