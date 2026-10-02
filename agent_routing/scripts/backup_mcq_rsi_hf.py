#!/usr/bin/env python3
"""Back up /workspace/mcq_rsi to a Hugging Face repo (runbook §15).

The pod may have no persistent volume, so this is the only copy that survives a stop:

    python scripts/backup_mcq_rsi_hf.py --work /workspace/mcq_rsi --repo MaliDDD/margent-mcq-rsi           # one pass
    python scripts/backup_mcq_rsi_hf.py ... --every-minutes 60                                              # loop (tmux)
    python scripts/backup_mcq_rsi_hf.py ... --dry-run                                                       # list only

Each pass:
1. mirrors ``runs/``, ``logs/``, ``advisor_cache/{preflight,locked_test}/`` and the import manifests into
   ``<work>/hf_backup_stage`` (only changed files are copied, files gone from the source are removed),
   leaving out what a restore does not need: per-step FA-GRPO weights and optimizer states (``step-*/``,
   ~370 MB per step; the stage's ``final/`` adapter, every ``step.json`` and ``metrics.jsonl`` are kept),
   trainer ``checkpoint-*``, ``incomplete-*`` / ``.final-*`` temporaries, ``*.part`` and lock files;
2. packs the advisor output cache (one small JSON per request) into ``advisor_cache.tar.gz`` when it changed;
3. uploads the staging folder with ``upload_large_folder`` (resumable; unchanged files are skipped).

Uploading from the staging copy, never the live tree, means no upload reads a file while a stage
writes or deletes it. A file copied mid-write is copied again (and re-uploaded) on the next pass, and
files removed locally stay in the repo. The repo is created private unless ``--public``. Credentials
are never handled here: log in once with ``HF_HOME=/workspace/hf-cache /workspace/mcq-venv/bin/hf auth login``.

Restore on a new pod (same paths; runbook §15):
    hf download <repo> --local-dir /workspace/mcq_rsi_restore
    rsync -a /workspace/mcq_rsi_restore/{runs,logs,advisor_cache} /workspace/mcq_rsi/
    tar -xzf /workspace/mcq_rsi_restore/advisor_cache.tar.gz -C /workspace/mcq_rsi
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

STAGE = "hf_backup_stage"
TREES = ("runs", "logs", "advisor_cache/preflight", "advisor_cache/locked_test")
EXTRA_GLOBS = ("import/*/import_manifest.json",)
EXCLUDE = (
    "*.lock", "*.part", "*/__pycache__/*", "*/.cache/*",
    "*/incomplete-*", "*/.final-*", "*/checkpoint-*/*",
    "*/step-*/adapter_model.safetensors", "*/step-*/optimizer.pt",
)
ADVISOR_TAR = "advisor_cache.tar.gz"
ADVISOR_SKIP = ("preflight", "locked_test")


def log(msg: str) -> None:
    print(f"[mcq-backup {time.strftime('%F %T')}] {msg}", flush=True)


def excluded(rel: str) -> bool:
    rel = "/" + rel  # every pattern starts with "*" (which also matches "/"), so "*/x" matches a top-level x too
    return any(fnmatch.fnmatchcase(rel, p) for p in EXCLUDE)


def source_files(work: Path) -> Dict[str, Path]:
    """Relative path -> source file of everything the backup keeps."""
    out: Dict[str, Path] = {}
    for tree in TREES:
        root = work / tree
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(work).as_posix()
            dirnames[:] = [d for d in dirnames if not excluded(f"{rel_dir}/{d}/x")]
            for name in filenames:
                rel = f"{rel_dir}/{name}"
                if not excluded(rel):
                    out[rel] = Path(dirpath) / name
    for pattern in EXTRA_GLOBS:
        for f in work.glob(pattern):
            if f.is_file():
                out[f.relative_to(work).as_posix()] = f
    return out


def mirror(work: Path, stage: Path, dry_run: bool = False) -> Dict[str, int]:
    """Copy new or changed files (size or mtime differ) into ``stage``; delete staged files the source no longer has."""
    wanted = source_files(work)
    counts = {"copied": 0, "unchanged": 0, "removed": 0, "bytes_copied": 0}
    for rel, src in sorted(wanted.items()):
        dest = stage / rel
        try:
            st = src.stat()
        except FileNotFoundError:  # deleted by a running stage since the walk
            continue
        if dest.is_file():
            dt = dest.stat()
            if dt.st_size == st.st_size and dt.st_mtime_ns == st.st_mtime_ns:  # copy2 keeps the ns mtime
                counts["unchanged"] += 1
                continue
        counts["copied"] += 1
        counts["bytes_copied"] += st.st_size
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.copying")
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dest)
        except FileNotFoundError:
            tmp.unlink(missing_ok=True)
            counts["copied"] -= 1
            counts["bytes_copied"] -= st.st_size
    for tree in TREES + ("import",):
        root = stage / tree
        if not root.is_dir():
            continue
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = f.relative_to(stage).as_posix()
            if rel not in wanted:
                counts["removed"] += 1
                if not dry_run:
                    f.unlink()
    return counts


def pack_advisor_cache(work: Path, stage: Path, dry_run: bool = False) -> Optional[str]:
    """Rewrite ``advisor_cache.tar.gz`` when any cached advisor output is newer than it; returns why, or None."""
    cache = work / "advisor_cache"
    if not cache.is_dir():
        return None
    dest = stage / ADVISOR_TAR
    files = [f for d in sorted(cache.iterdir()) if d.is_dir() and d.name not in ADVISOR_SKIP
             for f in d.rglob("*") if f.is_file() and not f.name.endswith(".part")]
    if not files:
        return None
    newest = max(f.stat().st_mtime for f in files)
    if dest.is_file() and dest.stat().st_mtime >= newest:
        return None
    reason = f"{len(files)} cached advisor files, newest {time.strftime('%F %T', time.localtime(newest))}"
    if dry_run:
        return reason
    tmp = dest.with_name(f".{ADVISOR_TAR}.writing")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        for f in files:
            tar.add(f, arcname=f.relative_to(work).as_posix(), recursive=False)
    os.replace(tmp, dest)
    return reason


def upload(stage: Path, repo: str, private: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo, folder_path=str(stage), repo_type="model",
                            ignore_patterns=["*.copying", "*.writing", ".cache/**"], print_report=False)


def require_login() -> str:
    from huggingface_hub import whoami
    try:
        return whoami()["name"]
    except Exception as e:  # noqa: BLE001 - reported with the login hint
        raise SystemExit(f"not logged in to Hugging Face ({type(e).__name__}); run once: "
                         f"HF_HOME={os.environ.get('HF_HOME', '/workspace/hf-cache')} "
                         f"{Path(sys.executable).parent / 'hf'} auth login") from None


def one_pass(work: Path, repo: str, private: bool, dry_run: bool) -> Dict[str, object]:
    stage = work / STAGE
    started = time.time()
    counts = mirror(work, stage, dry_run=dry_run)
    packed = pack_advisor_cache(work, stage, dry_run=dry_run)
    result: Dict[str, object] = {"repo": repo, "private": private, "dry_run": dry_run, **counts,
                                 "advisor_cache_packed": packed}
    if not dry_run:
        upload(stage, repo, private)
        result["seconds"] = round(time.time() - started, 1)
        result["finished"] = time.strftime("%F %T")
        (work / "logs").mkdir(parents=True, exist_ok=True)
        (work / "logs" / "hf_backup_last.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=os.environ.get("MCQ_WORK", "/workspace/mcq_rsi"))
    p.add_argument("--repo", default=os.environ.get("HF_BACKUP_REPO", "MaliDDD/margent-mcq-rsi"))
    p.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    p.add_argument("--every-minutes", type=float, default=0.0, help="loop with this period (0: one pass)")
    p.add_argument("--dry-run", action="store_true", help="list what would be copied, packed and removed")
    a = p.parse_args(argv)
    work = Path(a.work)
    if not work.is_dir():
        raise SystemExit(f"no work directory {work}")
    if not a.dry_run:
        log(f"logged in to Hugging Face as {require_login()}; backing up {work} -> {a.repo} "
            f"({'public' if a.public else 'private'})")
    while True:
        try:
            log(json.dumps(one_pass(work, a.repo, not a.public, a.dry_run)))
        except Exception as e:  # noqa: BLE001 - a loop survives transient network/hub errors
            if a.every_minutes <= 0:
                raise
            log(f"backup pass failed ({type(e).__name__}: {e}); retrying next period")
        if a.every_minutes <= 0:
            return 0
        time.sleep(a.every_minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
