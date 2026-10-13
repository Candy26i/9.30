"""fig_label_quality + tab_label_quality: dev accuracy / calls of a manager SFT'd from S_1 on one round's labels only,
per benchmark, with the experiment-D pairs (labels collected by S_k vs G_k after GRPO) overlaid at chain round r_{k+1}.
All numbers come from data.json (label_quality, grpo_data_quality)."""
import sys, json
PA = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + ""
sys.path.insert(0, PA)
import style; style.setup()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from pathlib import Path

C = style.C
d = json.load(open(style.ROOT / "data.json"))
LQ, GQ = d["label_quality"], d["grpo_data_quality"]
ROUNDS = ["S_1", "r1", "r2", "r3", "r4", "r5"]
TICK = {"S_1": "S$_1$", "r1": "r1", "r2": "r2", "r3": "r3", "r4": "r4", "r5": "r5"}
N_DEV = d["meta"]["n_dev"]

def d_round(label):
    """Experiment-D label 'S2'/'G2' -> the chain round whose labels S_k collected (r_{k+1}); matches sft_rows in data."""
    return f"r{int(label[1:]) + 1}"

# ------------------------------------------------------------------ figure ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
fig.subplots_adjust(top=0.83, bottom=0.10, left=0.09, right=0.93, hspace=0.62, wspace=0.34)
XS, XG = 0.40, 0.62   # the S_k / G_k pair sits just right of its chain round
MINUS = "\u2212"

for ax, b, tag in zip(axes.flat, style.BENCH_ORDER, "abcd"):
    rows = LQ[b]
    xs = [ROUNDS.index(r["labels"]) for r in rows]
    acc = [r["acc"] for r in rows]
    calls = [r["calls"] for r in rows]
    s1 = acc[xs.index(0)]
    dq = GQ.get(b, [])

    # calls per question: light-blue bars on a secondary axis, kept in the lower third of the panel
    ax2 = ax.twinx()
    ax2.grid(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.bar(xs, calls, width=0.56, facecolor=C["light"], edgecolor=C["edge"], linewidth=0.6, zorder=1)
    ax2.set_ylim(0, 2.7)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.tick_params(axis="y", colors=C["numgray"], length=0, labelsize=9)
    for x, c in zip(xs, calls):
        ax2.text(x, c + 0.035, f"{c:.2f}", ha="center", va="bottom", fontsize=7, color=C["numgray"],
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)   # draw the accuracy axis above the bars

    # y-range: accuracy data in the upper half of the panel, bars below; accuracy ticks only where the data lives
    vals = acc + [r["acc"] for r in dq]
    lo, hi = min(vals), max(vals); span = max(hi - lo, 4.0)
    ax.set_ylim(lo - 1.35 * span, hi + 0.55 * span)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_yticks([t for t in ax.get_yticks() if lo - 0.35 * span <= t <= ax.get_ylim()[1]])
    R = ax.get_ylim()[1] - ax.get_ylim()[0]; off = R * 0.03

    # S_1 reference level and the accuracy line
    ax.axhline(s1, color=C["mid"], ls="--", lw=1.0, zorder=3)
    ax.plot(xs, acc, color=C["dark"], marker="o", ms=5, lw=1.6, zorder=5)
    for x, a in zip(xs, acc):
        ax.text(x, a + off, f"{a:.1f}", ha="center", va="bottom", fontsize=8, color=C["num"], zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

    # experiment D: hollow = S_k, filled = G_k, just right of chain round r_{k+1}; one delta label per pair
    pairs = {}
    for r in dq:
        pairs.setdefault(int(r["labels"][1:]), {})[r["labels"][0]] = r
    for k, pr in sorted(pairs.items()):
        x = ROUNDS.index(d_round(f"S{k}"))
        sa, ga = pr["S"]["acc"], pr["G"]["acc"]
        ax.plot(x + XS, sa, marker="D", ms=4.5, mfc="white", mec=C["mid"], mew=1.2, ls="none", zorder=6)
        ax.plot(x + XG, ga, marker="D", ms=4.5, mfc=C["mid"], mec=C["mid"], ls="none", zorder=6)
        dl = ga - sa
        txt = ("+" if dl > 0 else MINUS if dl < 0 else "") + f"{abs(dl):.1f}"
        if dl >= 0:
            ax.text(x + (XS + XG) / 2, max(sa, ga) + off, txt, ha="center", va="bottom", fontsize=7.5, color=C["mid"], zorder=7)
        else:
            ax.text(x + XG + 0.10, ga, txt, ha="left", va="center", fontsize=7.5, color=C["mid"], zorder=7)

    ax.set_xlim(-0.6, 5.6)
    ax.set_xticks(xs, [TICK[ROUNDS[x]] for x in xs])
    if len(xs) < len(ROUNDS):
        pass  # GPQA labels stop at r3: said in the caption, not inside the image
    ax.set_title(f"{style.BENCH_LABEL[b]}  (dev, n = {N_DEV[b]})", fontsize=11.5, pad=8)
    style.panel_tag(ax, f"({tag})", x=-0.08, y=1.03)
    if tag in "ac":
        ax.set_ylabel("Dev accuracy (%)")
    if tag in "bd":
        ax2.set_ylabel("Calls / question", color=C["numgray"], fontsize=10)
        ax2.yaxis.set_label_coords(1.13, 0.20)

# legend fills column-wise: row 1 = accuracy line, S_1 level, calls; row 2 = the two experiment-D markers
handles = [Line2D([], [], color=C["dark"], marker="o", ms=5, lw=1.6, label="Dev accuracy"),
           Line2D([], [], marker="D", ms=4.5, mfc="white", mec=C["mid"], mew=1.2, ls="none", label="Labels by S$_k$ (before GRPO)"),
           Line2D([], [], color=C["mid"], ls="--", lw=1.0, label="MARGENT S$_1$ level"),
           Line2D([], [], marker="D", ms=4.5, mfc=C["mid"], mec=C["mid"], ls="none", label="Labels by G$_k$ (after GRPO)"),
           Patch(facecolor=C["light"], edgecolor=C["edge"], label="Calls / question (right)")]
style.top_legend(fig, handles, [h.get_label() for h in handles], ncol=3, y=0.905)
style.caption(fig, ["Dev pools (MedQA/MMLU-Pro 200, GPQA 100, AQuA 254). Each point: SFT from S$_1$ on one round's labels only.",
                    "Hollow/filled pairs: labels collected by S$_k$ vs G$_k$ (after GRPO), placed beside chain round r$_{k+1}$ (built from S$_k$'s labels).",
                    "Small number next to a pair: G$_k$ $-$ S$_k$ accuracy. Bars: advisor calls per question (right axis).",
                    "Rounds: r1 = the static arm's data; r$_k$ = labels re-collected by S$_{k-1}$. GPQA labels stop at r3."],
              y=-0.11, x=0.05, fontsize=9)
print(style.save(fig, "fig_label_quality"))

# ------------------------------------------------------------------ table -------------------------------------------------------------------
TAB = Path(PA) / "tab"; TAB.mkdir(parents=True, exist_ok=True)
BENCH = style.BENCH_ORDER
best = {b: max(r["acc"] for r in LQ[b]) for b in BENCH}
by_round = {b: {r["labels"]: r for r in LQ[b]} for b in BENCH}
by_d = {b: {r["labels"]: r for r in GQ.get(b, [])} for b in BENCH}
d_ks = sorted({int(r["labels"][1:]) for b in BENCH for r in GQ.get(b, [])})

def tex_num(v, fmt, bold=False, signed=False):
    if v is None:
        return "--"
    s = fmt.format(abs(v)) if signed else fmt.format(v)
    if signed:
        s = ("$-$" if v < 0 else "+" if v > 0 else "") + s
    return f"\\textbf{{{s}}}" if bold else s

def md_num(v, fmt, bold=False, signed=False):
    if v is None:
        return "--"
    s = fmt.format(abs(v)) if signed else fmt.format(v)
    if signed:
        s = ("−" if v < 0 else "+" if v > 0 else "") + s
    return f"**{s}**" if bold else s

def row_cells(num, lookup, label, bold_best=False, signed=False):
    cells = []
    for b in BENCH:
        r = lookup[b].get(label)
        if r is None:
            cells += [num(None, ""), num(None, "")]
        elif signed:
            cells += [num(r["acc"], "{:.1f}", signed=True), num(r["calls"], "{:.2f}", signed=True)]
        else:
            cells += [num(r["acc"], "{:.1f}", bold=bold_best and abs(r["acc"] - best[b]) < 1e-9), num(r["calls"], "{:.2f}")]
    return cells

def deltas(k):
    out = {}
    for b in BENCH:
        s, g = by_d[b].get(f"S{k}"), by_d[b].get(f"G{k}")
        if s and g:
            out[b] = {"acc": g["acc"] - s["acc"], "calls": g["calls"] - s["calls"]}
    return out

caption = ("Label quality by round on the development pools. Each entry is a manager SFT'd from S$_1$ on that round's labels alone: "
           "dev accuracy (\\%) and advisor calls per question. Dev pools: MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254. "
           "r1 is the static arm's data; r$_k$ is re-collected by S$_{k-1}$; GPQA labels stop at r3. "
           "Experiment D: labels collected on the same questions by S$_k$ (before GRPO) and by G$_k$ (after GRPO), both trained from S$_1$; "
           "S$_k$'s labels are the ones used at chain round r$_{k+1}$. $\\Delta$ is G$_k$ minus S$_k$. "
           "Best chain-round accuracy per benchmark in bold; MedQA has no experiment-D runs.")

tex = []
tex.append("\\begin{table*}[t]")
tex.append("\\centering")
tex.append(f"\\caption{{{caption}}}")
tex.append("\\label{tab:label_quality}")
tex.append("\\small")
tex.append("\\setlength{\\tabcolsep}{5pt}")
tex.append("\\begin{tabular}{l rr rr rr rr}")
tex.append("\\toprule")
tex.append(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{style.BENCH_LABEL[b]}}}" for b in BENCH) + " \\\\")
tex.append(" ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(BENCH))))
tex.append("Labels & " + " & ".join(["Acc. & Calls"] * len(BENCH)) + " \\\\")
tex.append("\\midrule")
tex.append("\\multicolumn{9}{l}{\\textit{Chain rounds (one round's labels, SFT from S$_1$)}} \\\\")
for rd in ROUNDS:
    tex.append(f"{TICK[rd]} & " + " & ".join(row_cells(tex_num, by_round, rd, bold_best=True)) + " \\\\")
tex.append("\\midrule")
tex.append("\\multicolumn{9}{l}{\\textit{Experiment D: labels collected by S$_k$ (before GRPO) vs G$_k$ (after GRPO)}} \\\\")
for k in d_ks:
    rd = d_round(f"S{k}")
    tex.append(f"by S$_{k}$ ({rd}) & " + " & ".join(row_cells(tex_num, by_d, f"S{k}")) + " \\\\")
    tex.append(f"by G$_{k}$ ({rd}) & " + " & ".join(row_cells(tex_num, by_d, f"G{k}")) + " \\\\")
    dl = deltas(k)
    tex.append(f"\\quad $\\Delta$ (G$_{k}$ $-$ S$_{k}$) & " + " & ".join(row_cells(tex_num, {b: {"d": dl[b]} if b in dl else {} for b in BENCH}, "d", signed=True)) + " \\\\")
tex.append("\\bottomrule")
tex.append("\\end{tabular}")
tex.append("\\end{table*}")
(TAB / "tab_label_quality.tex").write_text("\n".join(tex) + "\n")

md_caption = (caption.replace("S$_1$", "S_1").replace("r$_k$", "r_k").replace("S$_{k-1}$", "S_{k-1}").replace("S$_k$", "S_k").replace("G$_k$", "G_k")
              .replace("r$_{k+1}$", "r_{k+1}").replace("$\\Delta$", "Δ").replace("\\%", "%"))
md = []
md.append("**Table: tab_label_quality.** " + md_caption)
md.append("")
md.append("| Labels | " + " | ".join(f"{style.BENCH_LABEL[b]} Acc. | {style.BENCH_LABEL[b]} Calls" for b in BENCH) + " |")
md.append("|---|" + "---:|" * (2 * len(BENCH)))
md.append("| *Chain rounds (one round's labels, SFT from S_1)* |" + " |" * (2 * len(BENCH)))
for rd in ROUNDS:
    md.append(f"| {rd} | " + " | ".join(row_cells(md_num, by_round, rd, bold_best=True)) + " |")
md.append("| *Experiment D: labels collected by S_k (before GRPO) vs G_k (after GRPO)* |" + " |" * (2 * len(BENCH)))
for k in d_ks:
    rd = d_round(f"S{k}")
    md.append(f"| by S_{k} ({rd}) | " + " | ".join(row_cells(md_num, by_d, f"S{k}")) + " |")
    md.append(f"| by G_{k} ({rd}) | " + " | ".join(row_cells(md_num, by_d, f"G{k}")) + " |")
    dl = deltas(k)
    md.append(f"| Δ (G_{k} − S_{k}) | " + " | ".join(row_cells(md_num, {b: {"d": dl[b]} if b in dl else {} for b in BENCH}, "d", signed=True)) + " |")
(TAB / "tab_label_quality.md").write_text("\n".join(md) + "\n")
print(TAB / "tab_label_quality.tex"); print(TAB / "tab_label_quality.md")
