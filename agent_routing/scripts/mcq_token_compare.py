#!/usr/bin/env python3
"""Paired comparisons of per-question token costs written by ``mcq_token_cost.py``.

    python scripts/mcq_token_compare.py --dir <out of mcq_token_cost.py> --pairs v3/S_1:v3/dynamic_sft v3/S_1:v3/static_sft ...

For each pair (same questions): mean total / decode / calls of both, the paired difference with a bootstrap two-sided p,
the relative change, and accuracy. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(path: Path):
    return {int(json.loads(l)["example_id"]): json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def boot_p(diffs, n_boot=10000, seed=0) -> float:
    rng = random.Random(seed)
    n = len(diffs)
    obs = sum(diffs) / n
    hits = 0
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        if (s - obs) * (1 if obs >= 0 else -1) <= -abs(obs):  # centred bootstrap: how often the centred mean is as extreme
            hits += 1
    return min(1.0, 2 * hits / n_boot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="labelA:labelB (labels are file stems in --dir)")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    d = Path(args.dir)
    print("| A | B | acc A → B | calls A → B | total tokens A → B (Δ%) | p | decode A → B (Δ%) | p |")
    print("|---|---|---|---|---|---|---|---|")
    for pair in args.pairs:
        a_lab, _, b_lab = pair.partition(":")
        try:
            A, B = load(d / f"{a_lab}.jsonl"), load(d / f"{b_lab}.jsonl")
        except FileNotFoundError as e:
            print(f"| {a_lab} | {b_lab} | missing ({e.filename}) |  |  |  |  |  |"); continue
        ids = sorted(set(A) & set(B))
        if not ids:
            print(f"| {a_lab} | {b_lab} | no common ids |  |  |  |  |  |"); continue
        n = len(ids)
        row = []
        for key in ("correct", "calls", "total", "decode"):
            ma = sum(float(A[i].get(key) or 0) for i in ids) / n
            mb = sum(float(B[i].get(key) or 0) for i in ids) / n
            row.append((ma, mb))
        tot_d = [float(B[i]["total"]) - float(A[i]["total"]) for i in ids]
        dec_d = [float(B[i]["decode"]) - float(A[i]["decode"]) for i in ids]
        p_tot, p_dec = boot_p(tot_d, args.n_boot), boot_p(dec_d, args.n_boot)
        (acc_a, acc_b), (c_a, c_b), (t_a, t_b), (d_a, d_b) = row
        acc = f"{acc_a*100:.1f} → {acc_b*100:.1f}" if A[ids[0]].get("correct") is not None and B[ids[0]].get("correct") is not None else "-"
        print(f"| {a_lab} | {b_lab} | {acc} | {c_a:.2f} → {c_b:.2f} | {t_a:.0f} → {t_b:.0f} ({(t_b/t_a-1)*100:+.0f}%) | {p_tot:.3f} | "
              f"{d_a:.0f} → {d_b:.0f} ({(d_b/d_a-1)*100:+.0f}%) | {p_dec:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
