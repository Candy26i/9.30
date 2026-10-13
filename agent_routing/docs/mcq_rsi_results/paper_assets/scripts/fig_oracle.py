"""fig_oracle + tab_oracle: dev-pool oracle headroom per checkpoint (S_1, round 3, round 5).

Figure: 4 panels (one per benchmark), x = checkpoint, y = dev accuracy. Oracle = dashed navy line with hollow
markers, policy = solid navy line with filled markers, draft-only and always-Verifier as horizontal reference lines
with small labels at the right end. Table: the same numbers (LaTeX booktabs + Markdown twin).
All numbers come from data.json (data.oracle_dev)."""
import sys, json
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + ""
sys.path.insert(0, ROOT)
import style; style.setup()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

C = style.C
d = json.load(open(Path(ROOT) / "data.json"))
O = d["oracle_dev"]
BENCH = style.BENCH_ORDER
CK = ["S_1", "r3", "r5"]
CK_LABEL = {"S_1": "MARGENT S$_1$", "r3": "Round 3", "r5": "Round 5"}
CK_SHORT = {"S_1": "S$_1$", "r3": "R3", "r5": "R5"}
X = {"S_1": 0, "r3": 1, "r5": 2}

# ---- sanity: the policy checkpoints are the dynamic arm of the no-GRPO (v3) chain at rounds 3 / 5 -------------
for b in BENCH:
    traj = d["dev_trajectory"][b]["v3"]["dynamic"]
    for ck, r in (("r3", "3"), ("r5", "5")):
        if ck in O[b]["policy"]:
            assert abs(O[b]["policy"][ck] - traj[r]["acc"]) < 0.06, (b, ck, O[b]["policy"][ck], traj[r]["acc"])
    assert abs(O[b]["policy"]["S_1"] - traj["1"]["acc"]) < 0.06
    assert O[b]["n"] == d["meta"]["n_dev"][b]

# ================================================================= FIGURE =====================================
import numpy as np
from matplotlib.transforms import Bbox

DRAFT_C = "#B4B4B4"
MS = 5.5          # marker size (pt)
VF = 8.5          # value-label font size
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey="row")
fig.subplots_adjust(left=0.075, right=0.985, top=0.865, bottom=0.225, wspace=0.10, hspace=0.50)
TAGS = ["(a)", "(b)", "(c)", "(d)"]
ROW_YLIM = {0: (50, 100), 1: (35, 90)}
ROW_YTICKS = {0: [60, 80, 100], 1: [40, 60, 80]}
LABEL_X = 2.32   # right end: reference-line labels start here (no markers beyond x = 2)
lbb = dict(facecolor="white", edgecolor="none", pad=0.6)

def _overlap(a, b):
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))

def _pad(bb, p):
    return Bbox([[bb.x0 - p, bb.y0 - p], [bb.x1 + p, bb.y1 + p]])

def place_policy_labels(ax, pts_policy, pts_oracle, ref_ys, extra_texts):
    """Put a value label next to each policy point: below by default, otherwise above / diagonal, picking the first
    candidate that does not collide with the reference lines, markers, line segments or other labels."""
    fig.canvas.draw(); R = fig.canvas.get_renderer(); px = fig.dpi / 72.0
    disp = lambda x, y: ax.transData.transform((x, y))
    axbb = ax.get_window_extent(R)
    obs = []
    for y in ref_ys:                                   # horizontal reference lines (+/- 1 pt)
        Y = disp(0, y)[1]; obs.append(Bbox([[axbb.x0, Y - 1.0 * px], [axbb.x1, Y + 1.0 * px]]))
    r = (MS / 2 + 1.0) * px
    for (x, y) in pts_policy + pts_oracle:             # markers
        X, Y = disp(x, y); obs.append(Bbox([[X - r, Y - r], [X + r, Y + r]]))
    for pts in (pts_policy, pts_oracle):               # line segments, sampled
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0, 1, 40):
                X, Y = disp(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
                obs.append(Bbox([[X - 1.2 * px, Y - 1.2 * px], [X + 1.2 * px, Y + 1.2 * px]]))
    for t in extra_texts:                              # oracle labels, reference-line labels
        obs.append(_pad(t.get_window_extent(R), 1.5 * px))
    cands = [((0, -4.5), "center", "top"), ((0, 4.5), "center", "bottom"), ((5, -3.5), "left", "top"),
             ((-5, -3.5), "right", "top"), ((5, 3.5), "left", "bottom"), ((-5, 3.5), "right", "bottom")]
    for (x, v) in pts_policy:
        best = None
        for off, ha, va in cands:
            t = ax.annotate(f"{v:.1f}", (x, v), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=VF,
                            color=C["num"], bbox=lbb, zorder=5)
            bb = _pad(t.get_window_extent(R), 0.6 * px)
            score = sum(_overlap(bb, o) for o in obs)
            if not (axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1):
                score += 1e6
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, t, bb)
            else:
                t.remove()
            if score == 0:
                break
        obs.append(_pad(best[2], 1.0 * px))

for i, (b, ax) in enumerate(zip(BENCH, axes.flat)):
    o = O[b]; row = i // 2
    cks = [ck for ck in CK if ck in o["policy"]]
    xs = [X[ck] for ck in cks]
    pol = [o["policy"][ck] for ck in cks]
    orc = [o["oracle"][ck] for ck in cks]

    ax.set_xlim(-0.35, 3.55); ax.set_ylim(*ROW_YLIM[row])
    ax.set_xticks([X[ck] for ck in CK], [CK_LABEL[ck] for ck in CK])
    ax.set_yticks(ROW_YTICKS[row])
    ax.tick_params(axis="x", labelsize=10)

    # reference lines (span the axes) with small labels at the right end
    ax.axhline(o["draft_only"], color=DRAFT_C, linestyle=(0, (1.2, 2.2)), linewidth=1.3, zorder=1)
    ax.axhline(o["always_verifier"], color=C["mid"], linestyle=(0, (4, 2.5)), linewidth=1.3, zorder=1)
    bb = dict(facecolor="white", edgecolor="none", pad=1.2)
    ref_texts = [
        ax.text(LABEL_X, o["draft_only"], f"Draft only  {o['draft_only']:.1f}", ha="left", va="center", fontsize=8,
                color=C["muted"], bbox=bb, zorder=3),
        ax.text(LABEL_X, o["always_verifier"], f"Always Verifier  {o['always_verifier']:.1f}", ha="left", va="center",
                fontsize=8, color=C["mid"], bbox=bb, zorder=3)]
    for _t in ref_texts:  # the legend names the reference lines; no inline text in the image
        _t.set_visible(False)

    # oracle (dashed, hollow) and policy (solid, filled)
    ax.plot(xs, orc, color=C["dark"], linestyle=(0, (4, 2.5)), linewidth=1.5, marker="o", markersize=MS,
            markerfacecolor="white", markeredgecolor=C["dark"], markeredgewidth=1.2, zorder=4)
    ax.plot(xs, pol, color=C["dark"], linestyle="-", linewidth=1.5, marker="o", markersize=MS,
            markerfacecolor=C["dark"], markeredgecolor=C["dark"], zorder=4)
    orc_texts = [ax.annotate(f"{v:.1f}", (x, v), xytext=(0, 4.5), textcoords="offset points", ha="center",
                             va="bottom", fontsize=VF, color=C["num"], bbox=lbb, zorder=5) for x, v in zip(xs, orc)]
    place_policy_labels(ax, list(zip(xs, pol)), list(zip(xs, orc)), [o["draft_only"], o["always_verifier"]],
                        ref_texts + orc_texts)

    style.panel_tag(ax, TAGS[i], x=-0.03, y=1.075)
    ax.text(0.0, 1.075, style.BENCH_LABEL[b] + f"  (n = {o['n']})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11.5, color=C["text"])
    if i >= 2:
        ax.set_xlabel("Manager checkpoint", fontsize=10.5, labelpad=4)

for ax in axes[:, 0]:
    ax.set_ylabel("Dev accuracy (%)")

handles = [
    Line2D([], [], color=C["dark"], linestyle=(0, (4, 2.5)), linewidth=1.5, marker="o", markersize=MS,
           markerfacecolor="white", markeredgecolor=C["dark"], markeredgewidth=1.2),
    Line2D([], [], color=C["dark"], linestyle="-", linewidth=1.5, marker="o", markersize=MS,
           markerfacecolor=C["dark"], markeredgecolor=C["dark"]),
    Line2D([], [], color=C["mid"], linestyle=(0, (4, 2.5)), linewidth=1.3),
    Line2D([], [], color=DRAFT_C, linestyle=(0, (1.2, 2.2)), linewidth=1.3),
]
labels = ["Oracle (best of direct / one call)", "Policy", "Always Verifier", "Draft only"]
style.top_legend(fig, handles=handles, labels=labels, ncol=4, y=0.915)

x0 = axes[1, 0].get_position().x0
style.caption(fig, [
    "Dev pools (MedQA/MMLU-Pro 200, GPQA 100, AQuA 254). Policy: dynamic arm of the no-GRPO chain.",
    "Oracle = per-question best of {answer directly, one advisor call} from forced evaluations;",
    "the advisors are frozen, so the oracle does not move. GPQA has no round-5 checkpoint.",
], y=0.02, x=x0, fontsize=9.5)
print(style.save(fig, "fig_oracle"))

# ================================================================= TABLE ======================================
def f1(v):
    return "--" if v is None else f"{v:.1f}"

def fd(v):
    return "--" if v is None else f"{v:+.1f}"

rows = []
for b in BENCH:
    o = O[b]
    pol = {ck: o["policy"].get(ck) for ck in CK}
    orc = {ck: o["oracle"].get(ck) for ck in CK}
    gap = {ck: (orc[ck] - pol[ck]) if pol[ck] is not None else None for ck in CK}
    best = max(v for v in pol.values() if v is not None)
    rows.append(dict(name=style.BENCH_LABEL[b], n=o["n"], draft=o["draft_only"], av=o["always_verifier"],
                     pol=pol, orc=orc, gap=gap, best=best))

TAB = Path(ROOT) / "tab"; TAB.mkdir(exist_ok=True)

# ---- LaTeX --------------------------------------------------------------------------------------------------
cap = ("Oracle headroom on the development pools. Dev accuracy (\\%) of the manager policy at three checkpoints "
       "(S$_1$, round 3, round 5) against the per-question oracle, the best of \\{answer directly, one advisor call\\} "
       "taken from the forced evaluations of the same checkpoint. Draft only answers every question directly; "
       "Always Verifier calls the Verifier on every question. Dev pools: MedQA 200, MMLU-Pro 200, GPQA 100, AQuA 254. "
       "R3/R5 are rounds 3 and 5 of the dynamic arm without GRPO; GPQA has no round 5. "
       "The advisors are frozen, so the oracle does not move. Bold marks the best policy checkpoint per benchmark.")
L = []
L.append("\\begin{table*}[t]")
L.append("\\centering")
L.append("\\small")
L.append("\\setlength{\\tabcolsep}{4.5pt}")
L.append("\\caption{" + cap + "}")
L.append("\\label{tab:oracle}")
L.append("\\begin{tabular}{l r r r rrr rrr rrr}")
L.append("\\toprule")
L.append(" & & Draft & Always & \\multicolumn{3}{c}{Policy} & \\multicolumn{3}{c}{Oracle} & "
         "\\multicolumn{3}{c}{Oracle $-$ policy} \\\\")
L.append("\\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}")
L.append("Benchmark & $n$ & only & Verifier & " + " & ".join(CK_SHORT[ck] for ck in CK) + " & "
         + " & ".join(CK_SHORT[ck] for ck in CK) + " & " + " & ".join(CK_SHORT[ck] for ck in CK) + " \\\\")
L.append("\\midrule")
for r in rows:
    pol_cells = []
    for ck in CK:
        v = r["pol"][ck]
        s = f1(v)
        if v is not None and abs(v - r["best"]) < 1e-9:
            s = "\\textbf{" + s + "}"
        pol_cells.append(s)
    orc_cells = [f1(r["orc"][ck]) for ck in CK]
    gap_cells = [fd(r["gap"][ck]) for ck in CK]
    L.append(f"{r['name']} & {r['n']} & {f1(r['draft'])} & {f1(r['av'])} & " + " & ".join(pol_cells) + " & "
             + " & ".join(orc_cells) + " & " + " & ".join(gap_cells) + " \\\\")
L.append("\\bottomrule")
L.append("\\end{tabular}")
L.append("\\end{table*}")
(TAB / "tab_oracle.tex").write_text("\n".join(L) + "\n")

# ---- Markdown twin ------------------------------------------------------------------------------------------
M = []
M.append("**Table: Oracle headroom on the development pools.** Dev accuracy (%) of the manager policy at three "
         "checkpoints (S_1, round 3, round 5) against the per-question oracle, the best of {answer directly, one advisor "
         "call} taken from the forced evaluations of the same checkpoint. Draft only answers every question directly; "
         "Always Verifier calls the Verifier on every question. Dev pools: MedQA 200, MMLU-Pro 200, GPQA 100, AQuA 254. "
         "R3/R5 are rounds 3 and 5 of the dynamic arm without GRPO; GPQA has no round 5. The advisors are frozen, so the "
         "oracle does not move. Bold marks the best policy checkpoint per benchmark.")
M.append("")
hdr = (["Benchmark", "n", "Draft only", "Always Verifier"] + [f"Policy {c}" for c in ["S_1", "R3", "R5"]]
       + [f"Oracle {c}" for c in ["S_1", "R3", "R5"]] + [f"Oracle − policy {c}" for c in ["S_1", "R3", "R5"]])
M.append("| " + " | ".join(hdr) + " |")
M.append("|:--|" + "--:|" * (len(hdr) - 1))
for r in rows:
    pol_cells = []
    for ck in CK:
        v = r["pol"][ck]
        s = "—" if v is None else f"{v:.1f}"
        if v is not None and abs(v - r["best"]) < 1e-9:
            s = f"**{s}**"
        pol_cells.append(s)
    orc_cells = ["—" if r["orc"][ck] is None else f"{r['orc'][ck]:.1f}" for ck in CK]
    gap_cells = ["—" if r["gap"][ck] is None else f"{r['gap'][ck]:+.1f}" for ck in CK]
    M.append("| " + " | ".join([r["name"], str(r["n"]), f"{r['draft']:.1f}", f"{r['av']:.1f}"] + pol_cells + orc_cells
                               + gap_cells) + " |")
(TAB / "tab_oracle.md").write_text("\n".join(M) + "\n")
print(TAB / "tab_oracle.tex"); print(TAB / "tab_oracle.md")

# ---- key numbers for the verifier ---------------------------------------------------------------------------
for r in rows:
    print(r["name"], "n=", r["n"], "draft", r["draft"], "AV", r["av"], "policy", r["pol"], "oracle", r["orc"],
          "gap", {k: (None if v is None else round(v, 1)) for k, v in r["gap"].items()})
