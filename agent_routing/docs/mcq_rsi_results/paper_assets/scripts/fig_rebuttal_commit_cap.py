"""fig_rebuttal_commit_cap: the commit cap of the manager's training data, on the locked tests, one column per benchmark.

Managers trained on the same round-1 collection with the S_1 recipe; only the commit cap rho differs. Top row: the
accuracy of each manager's learned calling next to that manager's own equal-information references (always Verifier,
the Verifier on its least confident candidates at the same number of calls, random questions with the same calls per
advisor, candidate only). Bottom row: the learned policy's advisor calls per question. Source:
docs/mcq_rsi_results/rebuttal/verifier_grid/commit_cap_sweep.json (scripts/mcq_commit_cap_sweep.py). Nothing is typed
by hand.

    python scripts/fig_rebuttal_commit_cap.py
"""
import json
import sys

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
import style

style.setup()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

SRC = style.ROOT.parent / "rebuttal" / "verifier_grid" / "commit_cap_sweep.json"
tables = json.load(open(SRC))
benches = [b for b in style.BENCH_ORDER if tables.get(b)]
SERIES = (  # key, label, line and marker style (drawn in this order, the learned policy last)
    ("candidate", "Candidate only", dict(color=style.C["muted"], linewidth=1.0, linestyle=(0, (1, 2)), marker="o", markersize=6.5,
                                         markerfacecolor=style.C["unsolved"], markeredgecolor=style.C["edge"], markeredgewidth=0.8)),
    ("random_roles", "Random questions, same calls per advisor", dict(color=style.C["muted"], linewidth=1.2, linestyle=(0, (4, 3)), marker="x",
                                                                      markersize=6, markeredgewidth=1.3)),
    ("always_verifier", "Always Verifier, with the candidate", dict(color=style.C["light"], linewidth=1.6, marker="s", markersize=7.5,
                                                                    markerfacecolor=style.C["light"], markeredgecolor=style.C["edge"], markeredgewidth=0.8)),
    ("confidence_matched", "Verifier on the least confident candidates, same calls", dict(color=style.C["mid"], linewidth=1.6, marker="o", markersize=7,
                                                                                         markerfacecolor="white", markeredgecolor=style.C["mid"], markeredgewidth=1.6)),
    ("learned", "Learned calling (first call)", dict(color=style.C["dark"], linewidth=2.0, marker="D", markersize=7.5,
                                                    markerfacecolor=style.C["dark"], markeredgecolor=style.C["edge"], markeredgewidth=0.8)),
)
fig, axes = plt.subplots(2, len(benches), figsize=(14.4, 5.0), squeeze=False, sharex="col",
                         gridspec_kw={"height_ratios": [3.0, 1.1], "hspace": 0.12, "wspace": 0.2})
summary = {}
for i, bench in enumerate(benches):
    rows, top, bottom = tables[bench], axes[0][i], axes[1][i]
    xs = list(range(len(rows)))
    for z, (key, _, kw) in enumerate(SERIES):
        top.plot(xs, [100 * r[key] for r in rows], zorder=3 + z, clip_on=False, **kw)
    s1 = next((x for x, r in zip(xs, rows) if r["ckpt"] == "s1"), None)
    if s1 is not None:  # the paper's manager: a light band behind its column in both rows
        for ax in (top, bottom):
            ax.axvspan(s1 - 0.36, s1 + 0.36, color=style.C["unsolved"], alpha=0.45, linewidth=0, zorder=0)
    values = [100 * r[key] for r in rows for key, _, _ in SERIES]
    lo, hi = min(values), max(values)
    top.set_ylim(lo - 0.06 * (hi - lo), hi + 0.08 * (hi - lo))
    top.yaxis.set_major_locator(MultipleLocator(5 if hi - lo <= 30 else 10))
    top.set_title(f"{style.BENCH_LABEL[bench]} (n = {rows[0]['n']})", loc="left", fontsize=12, pad=10)
    top.tick_params(axis="x", length=0)
    bars = bottom.bar(xs, [r["calls"] for r in rows], width=0.5, color=style.C["dark"], edgecolor=style.C["edge"], linewidth=0.8, zorder=3)
    for x, r in zip(xs, rows):
        bottom.annotate(f"{r['calls']:.2f}", (x, r["calls"]), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, color=style.C["num"])
    bottom.set_ylim(0, 1.3)
    bottom.set_yticks([0, 0.5, 1.0])
    bottom.set_xlim(-0.55, len(rows) - 0.45)
    bottom.set_xticks(xs)
    bottom.set_xticklabels([("no cap" if r["cap"] == "no cap" else r["cap"]) + ("\n(S$_1$)" if r["ckpt"] == "s1" else "") for r in rows])
    bottom.set_xlabel("Commit cap $\\rho$ of the training data")
    if i == 0:
        top.set_ylabel("Accuracy (%)")
        bottom.set_ylabel("Calls per\nquestion")
    summary[bench] = [(r["cap"], r["calls"], r["learned"], r["confidence_matched"], r["always_verifier"]) for r in rows]

handles = [Line2D([0], [0], **kw) for _, _, kw in reversed(SERIES)]
labels = [label for _, label, _ in reversed(SERIES)]
fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.14)
style.top_legend(fig, handles, labels, ncol=3, y=0.865, fontsize=10.5, handletextpad=0.5)
out = style.save(fig, "fig_rebuttal_commit_cap")
print("\n".join(out))
for bench, s in summary.items():
    print(bench, json.dumps(s))
