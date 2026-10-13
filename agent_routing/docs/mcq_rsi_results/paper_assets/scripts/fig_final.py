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
from matplotlib.patches import Rectangle
fig = plt.figure(figsize=(7.0, 2.5))
# two panels of equal width; the narrow gap between them holds panel (a)'s first-call-share column
AX_Y, AX_H, AX_W = 0.18, 0.62, 0.385
ax1 = fig.add_axes([0.118, AX_Y, AX_W, AX_H]); ax2 = fig.add_axes([0.600, AX_Y, AX_W, AX_H], sharey=ax1)
order = B[::-1]  # top-to-bottom = MedQA, MMLU-Pro, GPQA, AQuA-RAT
y = np.arange(len(order))
SEP = dict(edgecolor="white", linewidth=1.1)
segs = [("Direct", C["light"], ""), ("+1st call", C["green"], ""), ("+2nd call", C["lgreen"], "////"), ("Unsolved", C["unsolved"], "")]
for i, b in enumerate(order):
    n, a0, a1, ad, g1 = PT["table1_oracle_by_depth"][b]
    left = 0
    for (name, face, hatch), w in zip(segs, [a0, a1 - a0, ad - a1, 100 - ad]):
        ax1.barh(i, w, left=left, height=0.58, facecolor=face, hatch=hatch, zorder=2, **SEP)
        if name == "Direct":
            ax1.text(left + w / 2, i, r1(w), ha="center", va="center", fontsize=9, color=C["text"], zorder=3)
        elif name == "+1st call":
            ax1.text(left + w / 2, i, "+" + r1(w), ha="center", va="center", fontsize=9, color="white", zorder=3)
        left += w
    ax1.text(1.025, i, f"{r1(g1)}%", transform=ax1.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, color=C["num"])
ax1.text(1.025, len(order) - 0.47, "1st-call\nshare", transform=ax1.get_yaxis_transform(), ha="left", va="bottom", fontsize=7.5, color=C["text"], linespacing=0.95)
ax1.set_xlim(0, 100); ax1.set_ylim(-0.55, len(order) - 0.45)
ax1.set_yticks(y, [LAB[b] for b in order]); ax1.set_xticks([0, 25, 50, 75, 100])
ax1.set_xlabel("Questions (%)", labelpad=3)
for sp in ("top", "right", "left"):
    ax1.spines[sp].set_visible(False)
ax1.spines["bottom"].set_color(C["axis"]); ax1.tick_params(axis="y", length=0, pad=4); ax1.tick_params(axis="x", length=3)
ax1.grid(False)
h1 = [Patch(facecolor=f, hatch=h, edgecolor=C["edge"] if n != "Unsolved" else C["axis"], linewidth=0.6, label=n) for n, f, h in segs]
ax1.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.065, 1.03), ncol=4, frameon=False, fontsize=8,
           handlelength=1.1, handleheight=0.85, handletextpad=0.3, columnspacing=0.6, borderaxespad=0)
ax1.text(0.0, 1.085, "(a)", transform=ax1.transAxes, ha="left", va="center", fontsize=10.5)
# (b) net marginal value of one forced call: three thin bars per benchmark around a zero line
adv = [("Extractor", C["lgreen"], ""), ("Reasoner", C["green"], ""), ("Verifier", C["blue"], "")]
h = 0.19
for j, (name, face, hatch) in enumerate(adv):
    vals = [PT["table2_net_marginal_value_pp"][b][j] for b in order]
    ypos = y + (1 - j) * (h + 0.025)
    ax2.barh(ypos, vals, height=h, facecolor=face, hatch=hatch, edgecolor=C["edge"], linewidth=0.5, zorder=2, label=name)
    for yy, v in zip(ypos, vals):
        ax2.text(v + (0.7 if v >= 0 else -0.7), yy, f"{v:+.1f}".replace("-", "−"), ha="left" if v >= 0 else "right", va="center",
                 fontsize=7.5, color=C["num"], zorder=3)
ax2.axvline(0, color=C["text"], linewidth=0.8, zorder=3)
ax2.set_xlim(-8, 33); ax2.set_xticks([0, 10, 20, 30])
ax2.set_xlabel("Net value of one call (pp)", labelpad=3)
for sp in ("top", "right", "left"):
    ax2.spines[sp].set_visible(False)
ax2.spines["bottom"].set_color(C["axis"]); ax2.tick_params(axis="y", length=0, labelleft=False); ax2.tick_params(axis="x", length=3)
ax2.grid(True, axis="x", color=C["grid"], linewidth=0.7); ax2.grid(False, axis="y"); ax2.set_axisbelow(True)
h2 = [Patch(facecolor=f, hatch=hh, edgecolor=C["edge"], linewidth=0.6, label=n) for n, f, hh in adv]
ax2.legend(handles=h2, loc="lower left", bbox_to_anchor=(0.07, 1.03), ncol=3, frameon=False, fontsize=8,
           handlelength=1.2, handleheight=0.85, handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
ax2.text(0.0, 1.085, "(b)", transform=ax2.transAxes, ha="left", va="center", fontsize=10.5)
# light separators between benchmarks across both panels
for k in range(len(order) - 1):
    for ax in (ax1, ax2):
        ax.axhline(k + 0.5, color=C["grid"], linewidth=0.6, zorder=0)
out["fig_final_value"] = style.save(fig, "fig_final_value")

# ---------------------------------------------------------------- fig_final_tokens (main)
T = {b: D["tokens"][f"{b}:test"] for b in B}
def s1_tokens(b):
    return T[b]["v3g/S_1"]["total"] if "v3g/S_1" in T[b] else T[b]["v3/S_1"]["total"]
series = [("Base system", lambda b: T[b]["base/manager"]["total"], style.POLICY["base_system"]),
          ("MARGENT S$_1$", s1_tokens, style.POLICY["S_1"]),
          ("MARGENT RSI (dynamic)", lambda b: T[b]["v3g/dynamic"]["total"], style.POLICY["dynamic"])]
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
    ax.barh(i, corr * 100, height=0.55, **bar_kw(C["green"]), label="Correction" if i == 0 else None)
    ax.barh(i, -corrupt * 100, height=0.55, **bar_kw(C["red"]), label="Corruption" if i == 0 else None)
    ax.text(corr * 100 + 0.8, i, f"+{r1(corr * 100)}", ha="left", va="center", fontsize=8.5, color=C["num"])
    ax.text(-corrupt * 100 - 0.8, i, f"$-${r1(corrupt * 100)}", ha="right", va="center", fontsize=8.5, color=C["num"])
    ax.text(58, i, f"+{r1((corr - corrupt) * 100)}", ha="left", va="center", fontsize=9, color=C["num"])
ax.text(58, len(order) - 0.45, "Net (pp)", ha="left", va="bottom", fontsize=9, color=C["text"])
ax.spines["bottom"].set_bounds(-10, 50)
ax.axvline(0, color=C["axis"], linewidth=0.9)
ax.set_yticks(y, [LAB[b] for b in order]); ax.set_xlim(-10, 64); ax.set_xticks([-10, 0, 10, 20, 30, 40, 50]); ax.set_ylim(-0.6, len(order) - 0.3)
ax.set_xlabel("Signed outcome mass (% of all examples)")
style.horizontal_axes(ax)
style.top_legend(ax, ncol=2, y=1.0, fontsize=10)
out["fig_final_outcomes"] = style.save(fig, "fig_final_outcomes")

# ---------------------------------------------------------------- fig_final_calls (main): accuracy bars + tool-call line
SERIES_C = [("Base only", "base_only", lambda b: D["single_agent"][f"{b}:test"]["base_nocot_f"]["acc"], lambda b: 0.0),
            ("Base system", "base_system", lambda b: D["base_system"][f"{b}:test"]["acc"], lambda b: D["base_system"][f"{b}:test"]["calls"]),
            (style.POLICY["S_1"]["label"], "S_1", lambda b: D["locked"][f"{b}:test"]["v3g/S_1"]["acc"], lambda b: D["locked"][f"{b}:test"]["v3g/S_1"]["calls"]),
            (style.POLICY["dynamic"]["label"], "dynamic", lambda b: D["locked"][f"{b}:test"]["v3g/dynamic"]["acc"], lambda b: D["locked"][f"{b}:test"]["v3g/dynamic"]["calls"]),
            (style.POLICY["success"]["label"], "success", lambda b: D["locked"][f"{b}:test"]["v3g/success"]["acc"], lambda b: D["locked"][f"{b}:test"]["v3g/success"]["calls"])]
LINE_C = C["line"]
from decimal import Decimal as _Dec
def _exact(v, n, scale=100.0):  # value recomputed from its integer count (stored rates are rounded)
    return round(v * n / scale) * scale / n
def _n(b): return D["meta"]["n_test"][b]
import matplotlib.patheffects as _pe
BAR_TOP = {}  # (benchmark, series index) -> (bar height, fill), for label colours
CALL_TOP = 8.0  # right axis 0..8 keeps every call point below the bar tops
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax2 = ax.twinx()
x = np.arange(len(B)); k = len(SERIES_C); w = 0.16
handles = []
for j, (name, key, fa, fc) in enumerate(SERIES_C):
    pol = style.POLICY[key]
    xs = x + (j - (k - 1) / 2) * w
    accs = [_exact(fa(b), _n(b)) for b in B]
    calls_j = [_exact(fc(b), _n(b), 1.0) for b in B]
    for b, v in zip(B, accs):
        BAR_TOP[(b, j)] = (v, pol["facecolor"])
    bars = ax.bar(xs, accs, w * 0.94, facecolor=pol["facecolor"], edgecolor=C["edge"], hatch=pol["hatch"], linewidth=0.7, zorder=2, label=name)
    for bb, v, c in zip(bars, accs, calls_j):
        pct = c / CALL_TOP * 100  # where the call marker sits, in accuracy units
        clash = (pct - 3) < v + 7 and (pct + 8) > v + 1  # marker or its label would cover the accuracy label
        if clash:
            ax.text(bb.get_x() + bb.get_width() / 2, v - 1.2, r1(v), ha="center", va="top", fontsize=7.6, color=style.ink_on(pol["facecolor"]), zorder=4,
                    path_effects=[_pe.Stroke(linewidth=2.2, foreground=pol["facecolor"]), _pe.Normal()])
        else:
            ax.text(bb.get_x() + bb.get_width() / 2, v + 1.2, r1(v), ha="center", va="bottom", fontsize=7.6, color=C["num"], zorder=4)
for bi, b in enumerate(B):  # one line per benchmark through the five policies
    xs = [x[bi] + (j - (k - 1) / 2) * w for j in range(k)]
    cs = [_exact(fc(b), _n(b), 1.0) for _, _, _, fc in SERIES_C]
    import matplotlib.patheffects as _pe
    ax2.plot(xs, cs, color=LINE_C, lw=1.2, marker="o", ms=3.8, mfc="white", mec=LINE_C, mew=1.0, zorder=5,
             path_effects=[_pe.Stroke(linewidth=2.8, foreground="white"), _pe.Normal()])
    for j, (xx, c) in enumerate(zip(xs, cs)):
        top, face = BAR_TOP[(b, j)]
        below = j > 0 and cs[j - 1] > c and c >= 0.55  # the line arrives from above: put the number under the marker
        yy, va = (c - 0.16, "top") if below else (c + 0.16, "bottom")
        xt, ha = xx, "center"
        inside = (yy + (0 if below else 0.3)) / CALL_TOP * 100 + 3.5 < top  # the number sits on the bar: pick a colour that reads on it
        ax2.text(xt, yy, f"{c:.2f}" if c else "0", ha=ha, va=va, fontsize=6.8, zorder=6,
                 color=style.ink_on(face) if inside else LINE_C,
                 path_effects=[_pe.Stroke(linewidth=2.2, foreground=face), _pe.Normal()] if inside else None)
ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, CALL_TOP); ax2.set_yticks([0, 1, 2, 3]); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_bounds(0, 3)
ax2.spines["right"].set_color(C["axis"]); ax2.tick_params(axis="y", length=3, colors=C["num"], labelsize=10); ax2.grid(False)
ax2.set_ylabel("Tool calls per question", color=C["num"]); ax2.yaxis.set_label_coords(1.055, 0.25)
for sp in ("top",):
    ax2.spines[sp].set_visible(False)
ax.set_xticks(x, [LAB[b] for b in B]); ax.tick_params(axis="x", length=0)
from matplotlib.lines import Line2D as _L2
h, l = ax.get_legend_handles_labels()
h.append(_L2([], [], color=LINE_C, lw=1.2, marker="o", ms=3.8, mfc="white", mec=LINE_C)); l.append("Tool calls (right)")
ax.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9.5, handlelength=1.6, columnspacing=1.4)
out["fig_final_calls"] = style.save(fig, "fig_final_calls")

print(json.dumps({"tokens": vals}, indent=None))
for k, v in out.items(): print(k, v[0])
