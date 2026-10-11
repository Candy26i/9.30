#!/usr/bin/env python3
"""Strict single-agent baseline: one model answers the locked test alone, no advisors.

Two prompt modes:
- ``plain``: a short system instruction ("answer with ANSWER_<letter>") and the question with its choices; meant for the
  untrained base model (no DRAFT/tool protocol at all).
- ``manager``: the manager's own system prompt and user message exactly as in the evaluator, rendered WITHOUT the tool
  schemas, so a trained manager answers in its DRAFT_ANSWER_X / ANSWER_X format but cannot call anyone.

Greedy decoding, batched; every record keeps the text, the parsed answer and the prompt / generated token counts, so the
result is comparable with ``mcq_token_cost.py`` (here total = prompt + generated, decode = generated).

    python scripts/mcq_single_agent.py --config configs/mcq_rsi_medqa_v3.json --pool test --mode plain \\
        --checkpoint <base snapshot dir> --out /workspace/mcq_rsi/runs/single_agent/base_plain/medqa/test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FINAL_RE = re.compile(r"(?<![A-Z_])ANSWER_([A-Za-z0-9]+)\b")
DRAFT_RE = re.compile(r"DRAFT_ANSWER_([A-Za-z0-9]+)\b")
LETTER_RE = re.compile(r"\b([A-J])\b")

PLAIN_SYSTEM = ("You answer multiple-choice questions. Think briefly if you need to, then end your reply with exactly one line "
                "of the form ANSWER_<letter>, where <letter> is the key of the correct choice.")


def parse_answer(text: str, keys) -> str | None:
    keys = list(keys)
    finals = [m.group(1) for m in FINAL_RE.finditer(text)]
    for cand in reversed(finals):
        if cand in keys:
            return cand
    drafts = [m.group(1) for m in DRAFT_RE.finditer(text)]
    for cand in reversed(drafts):
        if cand in keys:
            return cand
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    letters = [m.group(1) for m in LETTER_RE.finditer(tail)]
    for cand in reversed(letters):
        if cand in keys:
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool", default="test")
    ap.add_argument("--mode", choices=["plain", "manager"], required=True)
    ap.add_argument("--checkpoint", required=True, help="adapter directory, or a full model directory (the base snapshot)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from src.manager.mcq_rsi import controller, evaluate, protocol
    from src.pipeline import stages

    cfg = controller.load_config(args.config)
    rt = controller.Runtime(cfg)
    rows = rt.rows(args.pool)
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ctx = evaluate.stage_context(rt.base(), out)
    tok, model = stages._load_manager_for_eval(ctx, str(args.checkpoint), device, dtype)
    tok.padding_side = "left"

    def prompt_for(row) -> str:
        row = row if isinstance(row, dict) else row.to_dict()
        if args.mode == "manager":
            messages = protocol.manager_messages(rt.bench, row)
        else:
            choices = "\n".join(f"  {k}. {v}" for k, v in row["choices"].items())
            user = f"Question:\n{row['question']}\n\nChoices:\n{choices}"
            if row.get("context"):
                user = f"Context:\n{row['context']}\n\n" + user
            messages = [{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    records = []
    started = time.time()
    try:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i: i + args.batch_size]
            prompts = [prompt_for(r) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            plen = enc["input_ids"].shape[1]
            for j, r in enumerate(batch):
                row = r if isinstance(r, dict) else r.to_dict()
                out_ids = gen[j, plen:]
                text = tok.decode(out_ids, skip_special_tokens=True).strip()
                n_out = int((out_ids != tok.pad_token_id).sum()) if tok.pad_token_id is not None else int(out_ids.numel())
                n_in = int(enc["attention_mask"][j].sum())
                pred = parse_answer(text, row["choices"].keys())
                gold = row.get("ground_truth")
                records.append({"example_id": int(row["example_id"]), "ground_truth": gold, "pred": pred, "correct": pred == gold,
                                "valid_answer": pred is not None, "in_tokens": n_in, "out_tokens": n_out,
                                "hit_limit": n_out >= args.max_new_tokens, "text": text[:3000]})
            print(f"[single-agent] {min(i + args.batch_size, len(rows))}/{len(rows)} ({time.time() - started:.0f}s)", flush=True)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    (out / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    n = len(records)
    summary = {"config": str(Path(args.config).resolve()), "pool": args.pool, "mode": args.mode, "checkpoint": str(args.checkpoint),
               "n": n, "accuracy": sum(x["correct"] for x in records) / n, "valid_answer_rate": sum(x["valid_answer"] for x in records) / n,
               "accuracy_given_valid": (sum(x["correct"] for x in records if x["valid_answer"]) / max(1, sum(x["valid_answer"] for x in records))),
               "hit_limit_rate": sum(x["hit_limit"] for x in records) / n,
               "in_tokens": sum(x["in_tokens"] for x in records) / n, "out_tokens": sum(x["out_tokens"] for x in records) / n,
               "total_tokens": sum(x["in_tokens"] + x["out_tokens"] for x in records) / n, "max_new_tokens": args.max_new_tokens,
               "seconds": round(time.time() - started)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[single-agent] {cfg['bench']}/{args.pool} {args.mode}: acc={summary['accuracy']:.3f} valid={summary['valid_answer_rate']:.3f} "
          f"acc|valid={summary['accuracy_given_valid']:.3f} tokens in/out={summary['in_tokens']:.0f}/{summary['out_tokens']:.0f} "
          f"hit_limit={summary['hit_limit_rate']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
