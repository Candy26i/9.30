#!/usr/bin/env python3
"""Token accounting of locked-test (or dev) trajectories: what each routing policy really costs per question.

For every record of a ``manager_tool_eval.jsonl`` the manager's turns are re-rendered exactly as the evaluator rendered
them (same system prompt, tool schemas and chat template) and the advisor calls are re-rendered with the advisors'
own prompts, so every token the two models saw or produced is counted:

- ``mgr_in``: prefill tokens of every manager generation step (the prompt, then the prompt + its own turn + the
  advisor's full output for the second step, …);
- ``mgr_out``: tokens the manager generated (draft, tool call, final answer);
- ``adv_in`` / ``adv_out``: prefill and generated tokens of every advisor call (full outputs from the advisor cache;
  the trajectory keeps only the first 2,000 characters);
- ``total`` = all four; ``decode`` = mgr_out + adv_out (the sequential part that dominates latency).

A ``draft_only`` row (commit the first draft, never call) is derived from the first turn of the first eval given.

    python scripts/mcq_token_cost.py --config configs/mcq_rsi_medqa_v3.json --pool test --out /workspace/tmp/tokcost/medqa_test \\
        --eval S_1=/path/S_1/test/manager_tool_eval.jsonl --eval dynamic_r3=/path/dynamic_sft/test/manager_tool_eval.jsonl

Writes ``<out>/<label>.jsonl`` (one line per question) and ``<out>/summary.json`` / ``summary.md`` (means per label).
CPU only (tokenizer + cache reads).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tb(v) -> bool:
    return v is True or str(v) == "True"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", action="append", default=[], help="label=path to manager_tool_eval.jsonl (repeatable)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from src.manager.mcq_rsi import controller, evaluate, protocol, prompts
    from src.manager.mcq_rsi.advisors import AdvisorRequest, CachedAdvisorPool
    from src.pipeline import stages

    cfg = controller.load_config(args.config)
    rt = controller.Runtime(cfg)
    rows = {int(r["example_id"]): r for r in rt.rows(args.pool)}
    tok = AutoTokenizer.from_pretrained(str(rt.import_path("S_1")), trust_remote_code=True)
    tools = stages._manager_tool_schemas(evaluate.BINDING)
    pool = CachedAdvisorPool(rt.bench.name, cfg["advisor_cache"], None, workers=1, mode=cfg["advisor_mode"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def ntok(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def render(messages, gen: bool) -> str:
        msgs = stages._normalize_tool_calls_for_template(messages)
        try:
            return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gen)

    def advisor_prompt_tokens(kind: str, row, candidate: str) -> int:
        msgs = prompts.build_advisor_messages(rt.bench.name, kind, row["question"], row.get("context") or "", dict(row["choices"]),
                                              candidate_answer=candidate if kind == "verifier" else "")
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return ntok(text)

    summary = {"config": str(Path(args.config).resolve()), "pool": args.pool, "labels": {}}
    first_eval = True
    for item in args.eval:
        label, _, path = item.partition("=")
        if not label or not path:
            raise SystemExit(f"--eval wants label=path, got {item!r}")
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if args.limit:
            recs = recs[: args.limit]
        per_q, draft_only = [], []
        missing_cache = 0
        for rec in recs:
            eid = int(rec["example_id"])
            row = rows.get(eid)
            if row is None:
                continue
            base = protocol.manager_messages(rt.bench, row)
            msgs = list(base)
            mgr_in = mgr_out = adv_in = adv_out = 0
            per_adv = {}
            pending_candidate = ""
            n_calls = 0
            for i, m in enumerate(rec.get("trajectory") or []):
                if m.get("role") == "assistant":
                    prefix_gen = ntok(render(msgs, True))
                    prefix_nogen = ntok(render(msgs, False))
                    asst = {"role": "assistant", "content": m.get("content") or ""}
                    if m.get("tool_call"):
                        asst["tool_calls"] = [{"type": "function", "function": {"name": m["tool_call"]["name"],
                                                                                 "arguments": m["tool_call"].get("arguments") or {}}}]
                        pending_candidate = str((m["tool_call"].get("arguments") or {}).get("current_draft") or "")
                    full_nogen = ntok(render(msgs + [asst], False))
                    header = prefix_gen - prefix_nogen
                    mgr_in += prefix_gen
                    mgr_out += max(0, full_nogen - prefix_nogen - header)
                    msgs.append(asst)
                    if i == 0:
                        d = m.get("content") or ""
                        d = d.split("\n")[0] if d else ""
                        draft_only.append({"example_id": eid, "mgr_in": prefix_gen, "mgr_out": ntok(d + "\n" + d.replace("DRAFT_", "")) + 1,
                                           "adv_in": 0, "adv_out": 0, "calls": 0})
                elif m.get("role") == "tool":
                    kind = str(m.get("name") or "")[:-5]
                    candidate = pending_candidate if kind == "verifier" else ""
                    full = pool.cached(AdvisorRequest.for_row(kind, row, candidate)) if kind in ("extractor", "reasoner", "verifier") else None
                    if full is None:
                        full = m.get("content") or ""
                        missing_cache += 1
                    a_in = advisor_prompt_tokens(kind, row, candidate) if kind in ("extractor", "reasoner", "verifier") else 0
                    a_out = ntok(full) + 1
                    adv_in += a_in
                    adv_out += a_out
                    per_adv[kind] = per_adv.get(kind, 0) + a_in + a_out
                    n_calls += 1
                    msgs.append({"role": "tool", "name": m.get("name"), "content": full})
            per_q.append({"example_id": eid, "correct": tb(rec.get("correct")), "calls": n_calls, "tools": rec.get("tool_names_called") or [],
                          "mgr_in": mgr_in, "mgr_out": mgr_out, "adv_in": adv_in, "adv_out": adv_out,
                          "total": mgr_in + mgr_out + adv_in + adv_out, "decode": mgr_out + adv_out, "per_advisor": per_adv})
        (out / f"{label}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in per_q))
        n = len(per_q)
        mean = lambda k: sum(x[k] for x in per_q) / n
        adv_tot = {}
        for x in per_q:
            for k, v in x["per_advisor"].items():
                adv_tot[k] = adv_tot.get(k, 0) + v
        summary["labels"][label] = {"n": n, "accuracy": sum(x["correct"] for x in per_q) / n, "calls": mean("calls"),
                                    "mgr_in": mean("mgr_in"), "mgr_out": mean("mgr_out"), "adv_in": mean("adv_in"), "adv_out": mean("adv_out"),
                                    "total": mean("total"), "decode": mean("decode"), "tokens_per_advisor_per_q": {k: v / n for k, v in adv_tot.items()},
                                    "advisor_outputs_missing_from_cache": missing_cache}
        if first_eval and draft_only:
            (out / "draft_only.jsonl").write_text("".join(json.dumps({**x, "total": x["mgr_in"] + x["mgr_out"], "decode": x["mgr_out"]}) + "\n" for x in draft_only))
            summary["labels"]["draft_only"] = {"n": len(draft_only), "accuracy": None, "calls": 0.0,
                                               "mgr_in": sum(x["mgr_in"] for x in draft_only) / len(draft_only),
                                               "mgr_out": sum(x["mgr_out"] for x in draft_only) / len(draft_only), "adv_in": 0.0, "adv_out": 0.0,
                                               "total": sum(x["mgr_in"] + x["mgr_out"] for x in draft_only) / len(draft_only),
                                               "decode": sum(x["mgr_out"] for x in draft_only) / len(draft_only), "tokens_per_advisor_per_q": {}}
            first_eval = False
        print(f"[tokcost] {label}: n={n} acc={summary['labels'][label]['accuracy']:.3f} calls={summary['labels'][label]['calls']:.3f} "
              f"total={summary['labels'][label]['total']:.0f} decode={summary['labels'][label]['decode']:.0f} missing_cache={missing_cache}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# Token cost per question: {cfg['bench']} / {args.pool}", "",
             "| policy | acc | calls | mgr in | mgr out | adv in | adv out | total | decode | per advisor (tokens/q) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, s in summary["labels"].items():
        acc = "-" if s["accuracy"] is None else f"{s['accuracy']*100:.1f}"
        adv = ", ".join(f"{k[:3]} {v:.0f}" for k, v in sorted(s["tokens_per_advisor_per_q"].items()))
        lines.append(f"| {lab} | {acc} | {s['calls']:.2f} | {s['mgr_in']:.0f} | {s['mgr_out']:.0f} | {s['adv_in']:.0f} | {s['adv_out']:.0f} | "
                     f"{s['total']:.0f} | {s['decode']:.0f} | {adv} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
