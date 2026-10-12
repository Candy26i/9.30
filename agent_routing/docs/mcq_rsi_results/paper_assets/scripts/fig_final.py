"""Final paper figures (2026-10-11), no caption text inside the images (captions live in LaTeX):
  fig_final_value     main      (a) oracle accuracy by delegation depth, (b) net marginal value of one forced call per advisor
  fig_final_tokens    main      mean total tokens per question: base system vs S_1 vs RSI (dynamic), with the reduction vs base
  fig_final_dev       appendix  development operating points: selected MARGENT, force all, force Verifier, final MARGENT
  fig_final_outcomes  appendix  correction vs corruption mass at the high-call operating points
MARGENT numbers come from margent_paper_tables.json (the paper's Tables 1, 2, 4, 5, 8); token numbers from data.json."""
import json, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import style
style.setup()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PT = json.load(open(ROOT / "margent_paper_tables.json")); D = json.load(open(ROOT / "data.json"))
B = style.BENCH_ORDER; LAB = style.BENCH_LABEL; C = style.C
def r1(x): return str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
def bar_kw(face, hatch="", edge=None):
    return dict(facecolor=face, edgecolor=edge or C["edge"], hatch=hatch, linewidth=0.8)
out = {}

# ---------------------------------------------------------------- fig_final_value (main)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.30))
order = B[::-1]  # top-to-bottom = MedQA, MMLU-Pro, GPQA, AQuA-RAT
y = np.arange(len(order))
segs = [("Direct answer", C["light"], ""), ("+ First call", C["mid"], ""), ("+ Second call", C["tan"], "//"), ("Unsolved", C["unsolved"], "")]
for i, b in enumerate(order):
    n, a0, a1, ad, g1 = PT["table1_oracle_by_depth"][b]
    parts = [a0, a1 - a0, ad - a1, 100 - ad]
    left = 0
    for (name, face, hatch), w in zip(segs, parts):
        ax1.barh(i, w, left=left, height=0.62, **bar_kw(face, hatch, edge=C["axis"] if name == "Unsolved" else None), label=name if i == 0 else None)
        if name in ("Direct answer", "+ First call"):
            ax1.text(left + w / 2, i, r1(w), ha="center", va="center", fontsize=9, color="white" if name == "+ First call" else C["text"])
        left += w
    ax1.text(103, i, f"{r1(g1)}%", ha="left", va="center", fontsize=9.5, color=C["num"])
ax1.text(103, len(order) - 0.42, "1st-call share", ha="left", va="bottom", fontsize=8, color=C["muted"])
ax1.set_yticks(y, [LAB[b] for b in order]); ax1.set_xlim(0, 100); ax1.set_ylim(-0.6, len(order) - 0.4)
ax1.set_xticks([0, 25, 50, 75, 100]); ax1.set_xlabel("Questions (%)")
style.horizontal_axes(ax1)
ax1.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=9, handlelength=1.4, columnspacing=1.0)
style.panel_tag(ax1, "(a)", x=-0.27, y=1.02)
adv = [("Extractor", C["light"], "//"), ("Reasoner", C["slate"], ""), ("Verifier", C["mid"], "")]
h = 0.26
for j, (name, face, hatch) in enumerate(adv):
    vals = [PT["table2_net_marginal_value_pp"][b][j] for b in order]
    ypos = y + (1 - j) * h
    ax2.barh(ypos, vals, height=h * 0.95, **bar_kw(face, hatch), label=name)
    for yy, v in zip(ypos, vals):
        ax2.text(v + (0.6 if v >= 0 else -0.6), yy, f"{v:+.1f}".replace("-", "\u2212"), ha="left" if v >= 0 else "right", va="center", fontsize=8, color=C["num"] if name == "Verifier" else C["numgray"])
ax2.axvline(0, color=C["axis"], linewidth=0.9)
ax2.set_xlim(-10, 35); ax2.set_xticks([-10, 0, 10, 20, 30]); ax2.tick_params(axis="y", labelleft=False)
ax2.set_xlabel("Net marginal value of one call (pp)")
style.horizontal_axes(ax2)
ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=9, handlelength=1.4, columnspacing=1.0)
style.panel_tag(ax2, "(b)", x=-0.03, y=1.10)
out["fig_final_value"] = style.save(fig, "fig_final_value")

# ---------------------------------------------------------------- fig_final_tokens (main)
T = {b: D["tokens"][f"{b}:test"] for b in B}
def s1_tokens(b):
    return T[b]["v3g/S_1"]["total"] if "v3g/S_1" in T[b] else T[b]["v3/S_1"]["total"]
series = [("Base system", lambda b: T[b]["base/manager"]["total"], style.POLICY["base_system"]),
          ("S$_1$", s1_tokens, style.POLICY["S_1"]),
          ("RSI (dynamic)", lambda b: T[b]["v3g/dynamic"]["total"], style.POLICY["dynamic"])]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
x = np.arange(len(B)); w = 0.26
vals = {name: [f(b) for b in B] for name, f, _ in series}
for k, (name, f, pol) in enumerate(series):
    bars = ax.bar(x + (k - 1) * w, vals[name], w * 0.95, label=name, **bar_kw(pol["facecolor"], pol["hatch"]))
    for xi, (bb, v) in enumerate(zip(bars, vals[name])):
        base = vals["Base system"][xi]
        txt = f"{v:,.0f}" if k == 0 else f"{v:,.0f}\n$-${(1 - v / base) * 100:.0f}%"
        ax.text(bb.get_x() + bb.get_width() / 2, v + 180, txt, ha="center", va="bottom", fontsize=8, color=C["numgray"] if k == 0 else C["num"], linespacing=1.0)
ax.set_xticks(x, [LAB[b] for b in B]); ax.set_ylabel("Tokens per question"); ax.set_ylim(0, 13500)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
style.top_legend(ax, ncol=3, y=1.0, fontsize=10)
out["fig_final_tokens"] = style.save(fig, "fig_final_tokens")

# ---------------------------------------------------------------- fig_final_dev (appendix)
fig, (axa, axb) = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.12))
dev = [("Selected MARGENT", lambda b: PT["table5_selected_vs_force_all"][b][0:2], C["mid"], ""),
       ("Force all", lambda b: PT["table5_selected_vs_force_all"][b][2:4], C["tan"], "//"),
       ("Always Verifier", lambda b: PT["table4_low_call_vs_verifier"][b][2:4], C["light"], ""),
       ("Low-call MARGENT", lambda b: PT["table4_low_call_vs_verifier"][b][0:2], C["dark"], "")]
x = np.arange(len(B)); w = 0.2
for k, (name, f, face, hatch) in enumerate(dev):
    accs = [f(b)[0] for b in B]; calls = [f(b)[1] for b in B]
    ba = axa.bar(x + (k - 1.5) * w, accs, w * 0.95, label=name, **bar_kw(face, hatch))
    bc = axb.bar(x + (k - 1.5) * w, calls, w * 0.95, **bar_kw(face, hatch))
    for bb, v in zip(ba, accs):
        axa.text(bb.get_x() + bb.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=C["num"])
    for bb, v in zip(bc, calls):
        lab = f"{v:.2f}"
        axb.text(bb.get_x() + bb.get_width() / 2, v + 0.08, lab, ha="center", va="bottom", fontsize=7.5, color=C["num"])
axa.set_ylim(0, 100); axa.set_ylabel("Dev accuracy (%)"); axb.set_ylim(0, 3.6); axb.set_ylabel("Mean calls"); axb.set_yticks([0, 1, 2, 3])
axb.set_xticks(x, [LAB[b] for b in B])
style.top_legend(axa, ncol=4, y=1.0, fontsize=10)
style.panel_tag(axa, "(a)", x=-0.07, y=0.98); style.panel_tag(axb, "(b)", x=-0.07, y=0.95)
out["fig_final_dev"] = style.save(fig, "fig_final_dev")

# ---------------------------------------------------------------- fig_final_outcomes (appendix)
fig, ax = plt.subplots(figsize=(7.2, 2.5))
order = B[::-1]; y = np.arange(len(order))
for i, b in enumerate(order):
    calls, corr, corrupt = PT["table8_outcomes"][b]
    ax.barh(i, corr * 100, height=0.55, **bar_kw(C["mid"]), label="Correction" if i == 0 else None)
    ax.barh(i, -corrupt * 100, height=0.55, **bar_kw(C["red"], "//", edge="#7A3532"), label="Corruption" if i == 0 else None)
    ax.text(corr * 100 + 0.8, i, f"+{r1(corr * 100)}", ha="left", va="center", fontsize=8.5, color=C["num"])
    ax.text(-corrupt * 100 - 0.8, i, f"$-${r1(corrupt * 100)}", ha="right", va="center", fontsize=8.5, color=C["red"])
    ax.text(58, i, f"+{r1((corr - corrupt) * 100)}", ha="left", va="center", fontsize=9, color=C["num"])
ax.text(58, len(order) - 0.45, "Net (pp)", ha="left", va="bottom", fontsize=9, color=C["text"])
ax.spines["bottom"].set_bounds(-10, 50)
ax.axvline(0, color=C["axis"], linewidth=0.9)
ax.set_yticks(y, [LAB[b] for b in order]); ax.set_xlim(-10, 64); ax.set_xticks([-10, 0, 10, 20, 30, 40, 50]); ax.set_ylim(-0.6, len(order) - 0.3)
ax.set_xlabel("Signed outcome mass (% of all examples)")
style.horizontal_axes(ax)
style.top_legend(ax, ncol=2, y=1.0, fontsize=10)
out["fig_final_outcomes"] = style.save(fig, "fig_final_outcomes")

print(json.dumps({"tokens": vals}, indent=None))
for k, v in out.items(): print(k, v[0])
