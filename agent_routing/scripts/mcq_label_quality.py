#!/usr/bin/env python3
"""Label quality by round: SFT from the same start (S_1) on each round's recollected labels, then one dev eval each.

Isolates the data from the model: in a dynamic run S_k = SFT(S_{k-1}, labels_k), so a later round's dev score mixes the
labels with everything the model accumulated. Here every SFT starts from S_1 (the imported paper manager) with the
config's SFT settings (``sft.draft_supervision`` included), and every checkpoint is evaluated on the config's dev pool
with the cached advisors, exactly like a round's ``sft_dev`` stage. Optionally the round-1 labels (the static arm's
data) and S_1 itself are evaluated too.

    python scripts/mcq_label_quality.py --config configs/mcq_rsi_medqa_v3.json --out /workspace/mcq_rsi/runs/medqa_v3lq \\
        --labels r2=/workspace/mcq_rsi/labels/medqa/r2.jsonl r3=... r4=... r5=... --round1 --eval-s1

Each SFT and each eval runs in its own subprocess (``--step``), as the controller does, so GPU memory is released
between steps. Finished steps (``sft_report.json`` / ``mcq_rsi_eval.json`` present) are skipped on a rerun. Output:
``<out>/<tag>/{model,sft_report.json,dev/mcq_rsi_eval.json}`` and ``<out>/summary.{json,md}``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS = ("accuracy", "initial_draft_accuracy", "call_rate", "calls_per_example", "call_rate_given_draft_wrong",
           "correction_rate", "corruption_rate", "valid_answer_rate")


def _cfg(path):
    from src.manager.mcq_rsi import controller
    return controller.load_config(path)


def step_sft(cfg, labels: Path, out: Path) -> None:
    from src.manager.mcq_rsi import controller, sft
    rt = controller.Runtime(cfg)
    init = rt.import_path("S_1")
    if not (init / "adapter_config.json").is_file():
        raise FileNotFoundError(f"S_1 adapter not imported: {init}")
    report = sft.train_round_sft(labels, init, out, base_model=cfg["base_model"], base_revision=cfg["base_revision"] or None,
                                 bench=rt.bench.name, config=cfg["sft"])
    print(f"[labelq] sft {labels} -> {out / 'model'}: {report['labels']['rows']} rows", flush=True)


def step_eval(cfg, checkpoint: Path, out: Path) -> None:
    from src.manager.mcq_rsi import controller, evaluate
    rt = controller.Runtime(cfg)
    rows = rt.rows(cfg["dev_pool"])
    result = evaluate.evaluate(str(checkpoint), rows, rt.pool(), out, bench=rt.bench.name, base_model=cfg["base_model"],
                               base_revision=cfg["base_revision"] or None, speculative=bool(cfg["eval"]["speculative"]),
                               require_gate=False)
    m = result["metrics"]
    print(f"[labelq] eval {checkpoint} on {cfg['dev_pool']}: acc={m['accuracy']:.4f} calls={m['calls_per_example']:.3f} "
          f"gate={result.get('gate') or 'pass'}", flush=True)


def run_step(args, kind: str, *extra: str) -> None:
    cmd = [sys.executable, __file__, "--config", args.config, "--out", args.out, "--step", kind, *extra]
    if args.advisor_url:
        cmd += ["--advisor-url", args.advisor_url]
    print(f"[labelq] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def label_stats(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    kinds = {}
    for r in rows:
        kinds[str(r.get("decision_type") or "?")] = kinds.get(str(r.get("decision_type") or "?"), 0) + 1
    report = path.with_suffix(".report.json")
    extra = {}
    if report.is_file():
        rep = json.loads(report.read_text())
        extra = {k: rep.get(k) for k in ("n_examples", "n_rescued", "n_selected_commit_decisions", "n_selected_rescue_decisions",
                                         "direct_accuracy", "oracle_accuracy", "rho") if k in rep}
    return {"rows": len(rows), "decision_types": kinds, **extra}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run config (v3: configs/mcq_rsi_<bench>_v3.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="*", default=[], help="tag=path pairs, e.g. r2=/.../labels.jsonl")
    ap.add_argument("--round1", action="store_true", help="also SFT on the imported round-1 labels (static's data)")
    ap.add_argument("--eval-s1", action="store_true", help="also evaluate S_1 itself (no training)")
    ap.add_argument("--advisor-url", default=None)
    ap.add_argument("--step", choices=["sft", "eval"], default=None, help="(internal) one unit in a fresh process")
    ap.add_argument("--labels-file", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--step-out", default=None)
    args = ap.parse_args()

    cfg = _cfg(args.config)
    if args.advisor_url:
        cfg["advisor_url"] = args.advisor_url
    out = Path(args.out)
    if args.step == "sft":
        step_sft(cfg, Path(args.labels_file), Path(args.step_out)); return 0
    if args.step == "eval":
        step_eval(cfg, Path(args.checkpoint), Path(args.step_out)); return 0

    out.mkdir(parents=True, exist_ok=True)
    from src.manager.mcq_rsi import controller
    rt = controller.Runtime(cfg)
    units = []
    if args.eval_s1:
        units.append(("S_1", None))
    if args.round1:
        units.append(("r1", rt.import_path("labels")))
    for item in args.labels:
        tag, _, path = item.partition("=")
        if not tag or not path:
            raise SystemExit(f"--labels wants tag=path, got {item!r}")
        units.append((tag, Path(path)))
    (out / "units.json").write_text(json.dumps([{"tag": t, "labels": str(p) if p else None} for t, p in units], indent=2))

    summary = {"config": os.path.abspath(args.config), "bench": cfg["bench"], "dev_pool": cfg["dev_pool"],
               "sft": cfg["sft"], "init": str(rt.import_path("S_1")), "units": {}}
    for tag, labels in units:
        unit = out / tag
        unit.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if labels is None:
            checkpoint = rt.import_path("S_1")
        else:
            checkpoint = unit / "model"
            if not (unit / "sft_report.json").is_file():
                run_step(args, "sft", "--labels-file", str(labels), "--step-out", str(unit))
        dev = unit / "dev"
        if not (dev / "mcq_rsi_eval.json").is_file():
            run_step(args, "eval", "--checkpoint", str(checkpoint), "--step-out", str(dev))
        result = json.loads((dev / "mcq_rsi_eval.json").read_text())
        entry = {"labels": str(labels) if labels else None, "checkpoint": str(checkpoint),
                 "metrics": {k: result["metrics"].get(k) for k in METRICS}, "gate": result.get("gate") or [],
                 "per_advisor_calls": result["metrics"].get("per_advisor_calls"), "seconds": round(time.time() - started)}
        if labels is not None:
            entry["label_stats"] = label_stats(Path(labels))
            rep = unit / "sft_report.json"
            if rep.is_file():
                entry["sft_rows"] = json.loads(rep.read_text()).get("labels", {}).get("rows")
        summary["units"][tag] = entry
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# Label quality by round: {cfg['bench']} ({cfg['dev_pool']}, SFT from S_1, draft_supervision="
             f"{cfg['sft'].get('draft_supervision')})", "",
             "| labels | SFT rows | acc | draft acc | call rate | calls/ex | call rate (draft wrong) | correction | corruption | valid |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, e in summary["units"].items():
        m = e["metrics"]
        f = lambda k: "-" if m.get(k) is None else f"{m[k]:.3f}"
        lines.append(f"| {tag} | {e.get('sft_rows', '-')} | {f('accuracy')} | {f('initial_draft_accuracy')} | {f('call_rate')} | "
                     f"{f('calls_per_example')} | {f('call_rate_given_draft_wrong')} | {f('correction_rate')} | "
                     f"{f('corruption_rate')} | {f('valid_answer_rate')} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
