#!/usr/bin/env python3
"""Retrain the MCQ advisor LoRAs with a validation split and best-epoch selection (subagents v2).

The paper-era advisors were trained for 10 epochs on 600 teacher (DeepSeek) examples per role with no
validation set; their train loss ends near 0.003 (memorised), and served correctly they do not beat the
base model in the same role (scripts/mcq_advisor_ab.py). v2 keeps the data, recipe and LoRA shape
(``src.subagents.train``: lr 2e-4, effective batch 8, r 16 / alpha 32, response-only loss) and changes:

- questions in the benchmark's ``dev``/``test`` pools are removed (the A/B and the locked test use them);
- 15% of the remaining questions (grouped by ``question_hash``) form a validation split;
- at most 3 epochs, validation loss after each, and the adapter of the epoch with the lowest
  validation loss is exported (early stopping by selection).

    # one queue per GPU (the advisor servers must be stopped)
    CUDA_VISIBLE_DEVICES=0 python scripts/train_subagents_v2.py train --jobs medqa:extractor,medqa:reasoner ...
    python scripts/train_subagents_v2.py serve-copies      # renamed copies for vLLM: <bench>_<kind>_v2
    python scripts/train_subagents_v2.py upload --repo-template MaliDDD/agent-routing-advisors-{bench}-9b-v2

Layout (``--out``, default /workspace/mcq_rsi/subagents_v2):
    <bench>/<kind>/data/{train,val}.jsonl, split.json   the split actually trained on
    <bench>/<kind>/run/                                 trainer output (checkpoint-* per epoch)
    <bench>/<kind>_adapter/                             the selected epoch's adapter + tokenizer
    <bench>/<kind>_report.json                          losses per epoch, selection, timings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KINDS = ("extractor", "reasoner", "verifier")
BENCHES = ("medqa", "mmlu_pro", "gpqa", "aqua")
HELD_OUT_POOLS = ("dev", "test", "test_paper")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# The paper-era advisors' LoRA targets (no linear-attention projections: vLLM 0.26 applies none to Qwen3.5's
# in_proj_*/out_proj, so an adapter trained on them would not be served as trained).
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt")


def _read_jsonl(path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def held_out_hashes(bench: str, import_dir: str) -> Dict[str, set]:
    """("id", example_id) and ("hash", question_hash) of every dev/test question (frozen split manifest;
    the advisor SFT rows use the same ``splits.identity`` hash)."""
    from src.manager.mcq_rsi import controller
    root = Path(__file__).resolve().parents[1]
    cfg = controller.load_config(json.loads((root / "configs" / f"mcq_rsi_{bench}.json").read_text()))
    cfg.update(import_dir=import_dir)
    pools = controller.Runtime(cfg).manifest()["pools"]
    return {pool: {("id", int(i["example_id"])) for i in pools[pool]} | {("hash", str(i["question_hash"])) for i in pools[pool]}
            for pool in HELD_OUT_POOLS if pool in pools}


def _held(row: Dict[str, Any], keys: set) -> bool:
    return ("id", int(row["example_id"])) in keys or ("hash", str(row.get("question_hash"))) in keys


def split_rows(rows: List[Dict[str, Any]], held_out: Dict[str, set], val_frac: float, seed: int):
    """Drop held-out questions; put ``val_frac`` of the remaining questions (by hash) in validation."""
    banned = set().union(*held_out.values()) if held_out else set()
    kept = [r for r in rows if not _held(r, banned)]
    dropped = {pool: sum(_held(r, keys) for r in rows) for pool, keys in held_out.items()}
    questions = sorted({str(r["question_hash"]) for r in kept})
    order = sorted(questions, key=lambda h: hashlib.sha1(f"{seed}:{h}".encode()).hexdigest())
    n_val = max(1, round(val_frac * len(order)))
    val_q = set(order[:n_val])
    train = [{**r, "split": "train"} for r in kept if str(r["question_hash"]) not in val_q]
    val = [{**r, "split": "dev"} for r in kept if str(r["question_hash"]) in val_q]
    return train, val, {"rows_in": len(rows), "dropped_held_out": dropped, "rows_kept": len(kept),
                        "questions": len(questions), "val_questions": len(val_q), "train_rows": len(train),
                        "val_rows": len(val), "val_frac": val_frac, "seed": seed}


def epoch_losses(run_dir: Path) -> Dict[str, Any]:
    """Validation and last train loss per epoch from the newest checkpoint's trainer_state.json."""
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        raise FileNotFoundError(f"{run_dir}: no checkpoint-*/trainer_state.json")
    log = json.loads(states[-1].read_text())["log_history"]
    evals = [{"epoch": round(x["epoch"], 3), "step": x["step"], "eval_loss": x["eval_loss"]} for x in log if "eval_loss" in x]
    return {"eval": evals, "train_loss_by_log": [(round(x["epoch"], 2), x["loss"]) for x in log if "loss" in x]}


def select_best(run_dir: Path) -> Dict[str, Any]:
    losses = epoch_losses(run_dir)
    if not losses["eval"]:
        raise RuntimeError(f"{run_dir}: no validation losses were logged")
    best = min(losses["eval"], key=lambda e: (e["eval_loss"], e["step"]))
    ckpt = run_dir / f"checkpoint-{best['step']}"
    if not (ckpt / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"{ckpt}: no adapter for the best epoch")
    return {**losses, "best": best, "best_checkpoint": str(ckpt)}


def _kernels() -> Dict[str, Optional[str]]:
    import importlib.metadata as md
    out = {}
    for dist in ("flash-linear-attention", "causal-conv1d", "torch", "transformers", "peft"):
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            out[dist] = None
    return out


def train_one(bench: str, kind: str, args) -> Dict[str, Any]:
    from src.manager.mcq_rsi import benchmarks as registry
    import src.subagents.train as subagent_train
    from src.subagents.train import SFTConfig, train_subagent_sft
    found = subagent_train._lora_target_modules
    if not getattr(found, "_v2", False):
        def seven(model, _found=found):
            return [m for m in _found(model) if m in TARGETS]
        seven._v2 = True
        subagent_train._lora_target_modules = seven
    root = Path(args.out) / bench / kind
    report_path = Path(args.out) / bench / f"{kind}_report.json"
    if report_path.is_file() and not args.force:
        print(f"[SUBAGENT_V2] {bench}/{kind}: done ({report_path})", flush=True)
        return json.loads(report_path.read_text())
    rows = _read_jsonl(Path(args.import_dir) / bench / "advisor_sft" / f"{kind}.jsonl")
    data = root / "data"
    if (data / "split.json").is_file():  # a resumed run keeps its split (the trainer checks the data sha)
        split = json.loads((data / "split.json").read_text())
    else:
        train, val, split = split_rows(rows, held_out_hashes(bench, args.import_dir), args.val_frac, args.seed)
        _write_jsonl(data / "train.jsonl", train)
        _write_jsonl(data / "val.jsonl", val)
        (data / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print(f"[SUBAGENT_V2] {bench}/{kind}: {split}", flush=True)
    cfg = SFTConfig(base_model=registry.BASE_MODEL, base_model_revision=registry.BASE_REVISION,
                    train_jsonl=str(data / "train.jsonl"), dev_jsonl=str(data / "val.jsonl"),
                    out_dir=str(root / "run"), seed=args.seed, max_seq_len=args.max_seq_len,
                    learning_rate=args.lr, num_train_epochs=args.epochs, per_device_batch_size=1,
                    gradient_accumulation_steps=8, lora_r=16, lora_alpha=32, lora_dropout=0.05)
    started = time.time()
    train_subagent_sft(cfg)
    sel = select_best(root / "run")
    dest = Path(args.out) / bench / f"{kind}_adapter"
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in ADAPTER_FILES:
        shutil.copy2(Path(sel["best_checkpoint"]) / name, tmp / name)
    for name in TOKENIZER_FILES:
        if (root / "run" / name).is_file():
            shutil.copy2(root / "run" / name, tmp / name)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    targets = sorted(json.loads((dest / "adapter_config.json").read_text())["target_modules"])
    if targets != sorted(TARGETS):
        raise RuntimeError(f"{dest}: LoRA targets {targets} are not the paper-era {sorted(TARGETS)}")
    sha = hashlib.sha256((dest / "adapter_model.safetensors").read_bytes()).hexdigest()
    report = {"bench": bench, "kind": kind, "split": split, "config": {k: getattr(cfg, k) for k in (
        "learning_rate", "num_train_epochs", "gradient_accumulation_steps", "lora_r", "lora_alpha", "lora_dropout",
        "max_seq_len", "seed", "base_model", "base_model_revision")}, "lora_targets": targets,
        "kernels": _kernels(), "teacher": rows[0].get("teacher_model"),
        **{k: sel[k] for k in ("eval", "best", "best_checkpoint", "train_loss_by_log")},
        "adapter_sha256": sha, "wall_seconds": time.time() - started}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[SUBAGENT_V2] {bench}/{kind}: eval losses {[round(e['eval_loss'], 4) for e in sel['eval']]} "
          f"-> epoch {sel['best']['epoch']} (step {sel['best']['step']}), sha {sha[:12]}", flush=True)
    return report


def cmd_train(args) -> int:
    jobs = [j.split(":") for j in args.jobs.split(",") if j]
    failed = []
    for bench, kind in jobs:
        try:
            train_one(bench, kind, args)
        except Exception as e:  # noqa: BLE001 - the queue continues; failures are listed at the end
            print(f"[SUBAGENT_V2] {bench}/{kind} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(f"{bench}:{kind}")
    print(f"[SUBAGENT_V2] queue done; failed: {failed}", flush=True)
    return 1 if failed else 0


def cmd_serve_copies(args) -> int:
    """Renamed (multimodal key) copies vLLM applies, named <bench>_<kind>_v2; prints the --lora-modules entries."""
    from src.manager.mcq_rsi.serving import prepare_served_lora
    entries = []
    for bench in args.benches.split(","):
        for kind in KINDS:
            src = Path(args.out) / bench / f"{kind}_adapter"
            if not (src / "adapter_model.safetensors").is_file():
                print(f"[SUBAGENT_V2] {bench}/{kind}: no adapter yet", flush=True)
                continue
            name = f"{bench}_{kind}_v2"
            prepare_served_lora(src, Path(args.served_dir) / name, mode="multimodal")
            entries.append(f"{name}={Path(args.served_dir) / name}")
    print(" ".join(entries))
    return 0


def cmd_upload(args) -> int:
    from huggingface_hub import HfApi
    api = HfApi()
    for bench in args.benches.split(","):
        folder = Path(args.out) / bench
        adapters = [k for k in KINDS if (folder / f"{k}_adapter" / "adapter_model.safetensors").is_file()]
        if not adapters:
            print(f"[SUBAGENT_V2] {bench}: nothing to upload", flush=True)
            continue
        repo = args.repo_template.format(bench=bench)
        api.create_repo(repo, repo_type="model", private=not args.public, exist_ok=True)
        allow = [f"{k}_adapter/*" for k in adapters] + [f"{k}_report.json" for k in adapters] + \
                [f"{k}/data/*" for k in adapters]
        api.upload_folder(repo_id=repo, folder_path=str(folder), allow_patterns=allow,
                          commit_message="subagents v2: validation split, best of <=3 epochs")
        print(f"[SUBAGENT_V2] uploaded {bench} {adapters} -> https://huggingface.co/{repo}", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--jobs", required=True, help="bench:kind,... (one queue; run one per GPU)")
    t.add_argument("--import-dir", default="/workspace/mcq_rsi/import")
    t.add_argument("--val-frac", type=float, default=0.15)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--max-seq-len", type=int, default=4096)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--force", action="store_true")
    s = sub.add_parser("serve-copies")
    s.add_argument("--benches", default=",".join(BENCHES))
    s.add_argument("--served-dir", default="/workspace/mcq_rsi/served_loras_v2")
    u = sub.add_parser("upload")
    u.add_argument("--benches", default=",".join(BENCHES))
    u.add_argument("--repo-template", default="MaliDDD/agent-routing-advisors-{bench}-9b-v2")
    u.add_argument("--public", action="store_true", help="create public repos (default: private)")
    for q in (t, s, u):
        q.add_argument("--out", default="/workspace/mcq_rsi/subagents_v2")
    a = p.parse_args(argv)
    return {"train": cmd_train, "serve-copies": cmd_serve_copies, "upload": cmd_upload}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
