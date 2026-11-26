"""fig_rebuttal_budget: accuracy against advisor-call budget on the locked tests, one panel per benchmark.

Every point and curve starts from the same manager-written candidate (checkpoint S_1 by default), with the same
decoding and forced-commit revision; only the rule that decides which questions get a call differs. Source:
docs/mcq_rsi_results/rebuttal/verifier_grid/analysis/<ckpt>_<bench>_test.json and _curve.csv
(scripts/mcq_verifier_grid_analyze.py). Nothing is typed by hand.

    python scripts/fig_rebuttal_budget.py [ckpt]      # ckpt: s1 (default), dyn3, dyn3g, base
"""
import csv
import json
import sys

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
import style

style.setup()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

CKPT = sys.argv[1] if len(sys.argv) > 1 else "s1"
SRC = style.ROOT.parent / "rebuttal" / "verifier_grid" / "analysis"
NAME = {"s1": "MARGENT S$_1$", "dyn3": "MARGENT RSI (dynamic), SFT only", "dyn3g": "MARGENT RSI (dynamic)", "base": "Base manager"}

benches = [b for b in style.BENCH_ORDER if (SRC / f"{CKPT}_{b}_test.json").is_file()]
if not benches:
    raise SystemExit(f"no analysis for {CKPT} in {SRC}")
fig, axes = plt.subplots(1, len(benches), figsize=(3.6 * len(benches) if len(benches) < 4 else 14.4, 3.7), squeeze=False)
summary = {}
for i, (bench, ax) in enumerate(zip(benches, axes[0])):
    res = json.load(open(SRC / f"{CKPT}_{bench}_test.json"))
    rules = res["rules"]
    curve = list(csv.DictReader(open(SRC / f"{CKPT}_{bench}_test_curve.csv")))
    x = [float(r["calls_per_question"]) for r in curve]
    conf = [100 * float(r["confidence_gated_accuracy"]) for r in curve]
    rand = [100 * float(r["random_verifier_expected_accuracy"]) for r in curve]
    # Verifier on a random subset (expected accuracy is linear in the budget) and on the least confident candidates
    ax.plot(x, rand, color=style.C["muted"], linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)
    ax.plot(x, conf, color=style.C["mid"], linewidth=1.9, zorder=3)
    learned, always, nocand = rules["learned"], rules["always_verifier"], rules.get("always_verifier_nocand")
    r = rules["random_verifier"]
    ax.errorbar([r["calls"]], [100 * r["accuracy"]], yerr=[[100 * (r["accuracy"] - r["ci95"][0])], [100 * (r["ci95"][1] - r["accuracy"])]],
                color=style.C["muted"], capsize=3, linewidth=1.2, zorder=3.5)
    ax.scatter([learned["calls"]], [100 * learned["accuracy"]], s=95, marker="D", facecolor=style.C["dark"],
               edgecolor=style.C["edge"], linewidth=0.8, zorder=5)
    ax.scatter([1.0], [100 * always["accuracy"]], s=85, marker="s", facecolor=style.C["light"], edgecolor=style.C["edge"],
               linewidth=0.8, zorder=5, clip_on=False)
    if nocand:
        ax.scatter([1.0], [100 * nocand["accuracy"]], s=85, marker="s", facecolor="white", edgecolor=style.C["edge"],
                   linewidth=1.0, zorder=4.5, clip_on=False)
    ax.scatter([0.0], [100 * rules["candidate"]["accuracy"]], s=70, marker="o", facecolor=style.C["unsolved"],
               edgecolor=style.C["edge"], linewidth=0.8, zorder=5, clip_on=False)
    ax.annotate(f"{100 * learned['accuracy']:.1f}", (learned["calls"], 100 * learned["accuracy"]), xytext=(-7, 7),
                textcoords="offset points", ha="right", va="bottom", fontsize=10, color=style.C["num"], zorder=6)
    # the two always-Verifier values sit at x = 1: the higher one is labelled above its marker, the lower one below
    ends = [(always["accuracy"], True)] + ([(nocand["accuracy"], False)] if nocand else [])
    top = max(v for v, _ in ends)
    for value, _ in ends:
        above = value == top or len(ends) == 1
        ax.annotate(f"{100 * value:.1f}", (1.0, 100 * value), xytext=(0, 8 if above else -8), textcoords="offset points",
                    ha="center", va="bottom" if above else "top", fontsize=10, color=style.C["num"], zorder=6)
    lo = min(conf + rand + [100 * rules["candidate"]["accuracy"]])
    hi = max(conf + [100 * always["accuracy"], 100 * (nocand or always)["accuracy"]])
    pad = (hi - lo) * 0.12
    ax.set_xlim(-0.03, 1.07)
    ax.set_ylim(lo - pad, hi + 1.6 * pad)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(5 if hi - lo <= 30 else 10))
    ax.set_title(f"{style.BENCH_LABEL[bench]} (n = {res['n']})", loc="left", fontsize=12, pad=10)
    ax.set_xlabel("Advisor calls per question")
    if i == 0:
        ax.set_ylabel("Accuracy (%)")
    summary[bench] = {"candidate": rules["candidate"]["accuracy"], "learned": (learned["calls"], learned["accuracy"]),
                      "always_verifier": always["accuracy"], "nocand": nocand and nocand["accuracy"],
                      "confidence_at_learned_budget": rules["confidence_matched"]["accuracy"],
                      "random_at_learned_budget": (r["accuracy"], r["ci95"])}

handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=style.C["unsolved"], markeredgecolor=style.C["edge"]),
    Line2D([0], [0], marker="D", linestyle="", markersize=8.5, markerfacecolor=style.C["dark"], markeredgecolor=style.C["edge"]),
    Line2D([0], [0], color=style.C["mid"], linewidth=1.9),
    Line2D([0], [0], color=style.C["muted"], linestyle=(0, (4, 3)), linewidth=1.3),
    Line2D([0], [0], marker="s", linestyle="", markersize=8.5, markerfacecolor=style.C["light"], markeredgecolor=style.C["edge"]),
    Line2D([0], [0], marker="s", linestyle="", markersize=8.5, markerfacecolor="white", markeredgecolor=style.C["edge"]),
]
labels = ["Candidate only", "Learned calling (first call)", "Verifier on the least confident candidates",
          "Verifier on a random subset (95% interval at the learned budget)", "Always Verifier, with the candidate",
          "Always Verifier, candidate removed"]
fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.15, wspace=0.2)
style.top_legend(fig, handles, labels, ncol=3, y=0.85, fontsize=10.5, handletextpad=0.5)
out = style.save(fig, f"fig_rebuttal_budget_{CKPT}")
print("\n".join(out))
for bench, s in summary.items():
    print(bench, json.dumps(s))
