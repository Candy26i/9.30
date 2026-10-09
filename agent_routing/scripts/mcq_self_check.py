#!/usr/bin/env python3
"""Does the manager carry an "I am wrong" signal it could route on? Confidence and self-consistency diagnostics on dev.

For every dev question the manager's first turn is rendered exactly as the evaluator does (same system prompt, tools,
chat template). Three cheap signals are measured before any advisor is called:

- letter confidence: the next-token distribution after the forced prefix ``DRAFT_ANSWER_`` restricted to the choice
  keys (p_max, margin p1 - p2, entropy);
- self-consistency: ``--n-samples`` sampled drafts at ``--temperature``; agreement = share equal to the greedy draft;
- (reference) the greedy draft itself, checked against the free-policy eval's initial draft.

Each signal is scored as a detector of "greedy draft is wrong" (AUROC, no sklearn), and a confidence-thresholded router
is simulated with the forced-Verifier dev eval of the same checkpoint: call the Verifier when the signal says uncertain,
otherwise commit the draft. Its cost/accuracy curve is reported at the budgets of the learned policies.

    python scripts/mcq_self_check.py --config configs/mcq_rsi_medqa_v3.json --out /workspace/mcq_rsi/runs/medqa_v3sc \\
        --policy-eval .../r1/S1_dev/manager_tool_eval.jsonl --forced-verifier .../final/S_1/dev_forced_verifier/manager_forced_verifier.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def auroc(scores, labels) -> float:
    """Rank AUROC of ``scores`` for ``labels`` (True = positive); ties get mid-ranks."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = [r for r, (_, l) in zip(ranks, pairs) if l]
    npos, nneg = len(pos), n - len(pos)
    if not npos or not nneg:
        return float("nan")
    return (sum(pos) - npos * (npos + 1) / 2) / (npos * nneg)


def tb(v) -> bool:
    return v is True or str(v) == "True"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="manager adapter dir (default: the imported S_1)")
    ap.add_argument("--policy-eval", required=True, help="manager_tool_eval.jsonl of the free policy on dev (gold + drafts)")
    ap.add_argument("--forced-verifier", default=None, help="manager_forced_verifier.jsonl of the same checkpoint on dev")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--budgets", default="", help="comma list of call rates to report the simulated router at")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from src.manager.mcq_rsi import controller, evaluate, protocol
    from src.manager.prompt import parse_draft_answer
    from src.pipeline import stages

    cfg = controller.load_config(args.config)
    rt = controller.Runtime(cfg)
    rows = rt.rows(cfg["dev_pool"])
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = {int(r["example_id"]): r for r in (json.loads(l) for l in open(args.policy_eval, encoding="utf-8") if l.strip())}
    forced = {}
    if args.forced_verifier:
        forced = {int(r["example_id"]): r for r in (json.loads(l) for l in open(args.forced_verifier, encoding="utf-8") if l.strip())}
    checkpoint = Path(args.checkpoint) if args.checkpoint else rt.import_path("S_1")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ctx = evaluate.stage_context(rt.base(), out)
    tok, model = stages._load_manager_for_eval(ctx, str(checkpoint), device, dtype)
    tools = stages._manager_tool_schemas(evaluate.BINDING)
    records = []
    started = time.time()
    try:
        for idx, r in enumerate(rows):
            row = r if isinstance(r, dict) else r.to_dict()
            eid = int(row["example_id"])
            keys = list(row["choices"])
            messages = protocol.manager_messages(rt.bench, row)
            prompt = stages._render_manager_chat(tok, messages, tools)
            inputs = tok(prompt, return_tensors="pt").to(device)
            plen = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            greedy_text = tok.decode(gen[0, plen:], skip_special_tokens=True).strip()
            draft = parse_draft_answer(greedy_text, keys)
            # letter distribution after the forced prefix
            pre = tok(prompt + "DRAFT_ANSWER_", return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**pre).logits[0, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            key_lp = {}
            for k in keys:
                ids = tok.encode(k, add_special_tokens=False)
                key_lp[k] = float(logp[ids[0]]) if ids else float("-inf")
            m = max(key_lp.values())
            z = math.log(sum(math.exp(v - m) for v in key_lp.values())) + m
            p = {k: math.exp(v - z) for k, v in key_lp.items()}
            ps = sorted(p.values(), reverse=True)
            p_max, margin = ps[0], ps[0] - (ps[1] if len(ps) > 1 else 0.0)
            entropy = -sum(v * math.log(v) for v in p.values() if v > 0)
            argmax_key = max(p, key=p.get)
            # self-consistency
            with torch.no_grad():
                samp = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                                      num_return_sequences=args.n_samples, pad_token_id=tok.pad_token_id,
                                      eos_token_id=tok.eos_token_id)
            sdrafts = [parse_draft_answer(tok.decode(s[plen:], skip_special_tokens=True).strip(), keys) for s in samp]
            agree = sum(1 for d in sdrafts if d == draft) / max(1, len(sdrafts))
            votes = {}
            for d in sdrafts:
                if d is not None:
                    votes[d] = votes.get(d, 0) + 1
            majority = max(votes, key=votes.get) if votes else None
            pe = policy.get(eid) or {}
            gold = row.get("ground_truth") or pe.get("ground_truth")
            records.append({"example_id": eid, "gold": gold, "draft": draft, "draft_wrong": (draft != gold) if gold else None,
                            "policy_draft": pe.get("initial_draft"), "policy_correct": tb(pe.get("correct")) if pe else None,
                            "policy_calls": float(pe.get("tool_calls") or 0) if pe else None,
                            "p_max": p_max, "margin": margin, "entropy": entropy, "argmax_key": argmax_key,
                            "samples": sdrafts, "agreement": agree, "majority": majority,
                            "verifier_correct": tb(forced[eid]["correct"]) if eid in forced else None})
            if (idx + 1) % 25 == 0:
                print(f"[self-check] {idx + 1}/{len(rows)} ({time.time() - started:.0f}s)", flush=True)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    (out / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    scored = [x for x in records if x["draft_wrong"] is not None]
    n = len(scored)
    wrong = [x["draft_wrong"] for x in scored]
    signals = {"1-p_max": [1 - x["p_max"] for x in scored], "-margin": [-x["margin"] for x in scored],
               "entropy": [x["entropy"] for x in scored], "1-agreement": [1 - x["agreement"] for x in scored]}
    summary = {"config": str(Path(args.config).resolve()), "checkpoint": str(checkpoint), "n": n, "n_samples": args.n_samples,
               "temperature": args.temperature, "draft_accuracy": 1 - sum(wrong) / n,
               "draft_matches_policy_draft": sum(x["draft"] == x["policy_draft"] for x in scored) / n,
               "argmax_matches_draft": sum(x["argmax_key"] == x["draft"] for x in scored) / n,
               "majority_accuracy": sum(x["majority"] == x["gold"] for x in scored) / n,
               "auroc_draft_wrong": {k: auroc(v, wrong) for k, v in signals.items()},
               "mean_signal_wrong_vs_right": {k: (sum(s for s, w in zip(v, wrong) if w) / max(1, sum(wrong)),
                                                  sum(s for s, w in zip(v, wrong) if not w) / max(1, n - sum(wrong))) for k, v in signals.items()}}
    if forced:
        pol = [x for x in scored if x["policy_correct"] is not None]
        summary["policy_dev"] = {"accuracy": sum(x["policy_correct"] for x in pol) / len(pol),
                                 "calls": sum(x["policy_calls"] for x in pol) / len(pol)}
        summary["always_verifier"] = sum(bool(x["verifier_correct"]) for x in scored if x["verifier_correct"] is not None) / n
        curves = {}
        for name, s in signals.items():
            order = sorted(range(n), key=lambda i: -s[i])  # most uncertain first
            pts = []
            correct_commit = [not scored[i]["draft_wrong"] for i in range(n)]
            acc_all_commit = sum(correct_commit) / n
            cur = acc_all_commit
            pts.append((0.0, cur))
            for rank, i in enumerate(order, 1):
                cur += ((1 if scored[i]["verifier_correct"] else 0) - (1 if correct_commit[i] else 0)) / n
                pts.append((rank / n, cur))
            curves[name] = pts
            budgets = [float(b) for b in args.budgets.split(",") if b.strip()]
            at = {}
            for b in budgets:
                k = min(range(len(pts)), key=lambda j: abs(pts[j][0] - b))
                at[f"{b:.2f}"] = round(pts[k][1], 4)
            best = max(pts, key=lambda t: t[1])
            summary.setdefault("simulated_router", {})[name] = {"at_budget": at, "best": {"calls": round(best[0], 3), "accuracy": round(best[1], 4)},
                                                               "curve_every_10pct": [(round(c, 2), round(a, 4)) for c, a in pts[:: max(1, n // 10)]]}
        (out / "curves.json").write_text(json.dumps(curves))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# Self-check: {cfg['bench']} dev (n={n}), checkpoint {checkpoint}", "",
             f"draft accuracy {summary['draft_accuracy']:.3f} (matches policy draft {summary['draft_matches_policy_draft']:.3f}); "
             f"majority-of-{args.n_samples} accuracy {summary['majority_accuracy']:.3f}", "",
             "| signal | AUROC(draft wrong) | mean when wrong | mean when right |", "|---|---|---|---|"]
    for k in signals:
        w, rgt = summary["mean_signal_wrong_vs_right"][k]
        lines.append(f"| {k} | {summary['auroc_draft_wrong'][k]:.3f} | {w:.3f} | {rgt:.3f} |")
    if forced:
        lines += ["", f"policy on dev: acc {summary['policy_dev']['accuracy']:.3f} at calls {summary['policy_dev']['calls']:.2f}; "
                      f"always-Verifier {summary['always_verifier']:.3f}", "",
                  "| simulated router (call Verifier when uncertain) | " + " | ".join(f"acc @ calls {b}" for b in summary["simulated_router"]["1-p_max"]["at_budget"]) + " | best (calls, acc) |",
                  "|---|" + "---|" * len(summary["simulated_router"]["1-p_max"]["at_budget"]) + "---|"]
        for k, v in summary["simulated_router"].items():
            lines.append(f"| {k} | " + " | ".join(f"{a:.3f}" for a in v["at_budget"].values()) + f" | {v['best']['calls']}, {v['best']['accuracy']:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
