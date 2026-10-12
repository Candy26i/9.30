"""fig_selfcheck + tab_selfcheck: can the manager's own draft confidence replace the learned call policy?

Figure (7.2 in, two panels side by side):
  (a) horizontal grouped bars (like reference 3.jpg): per benchmark, AUROC of three draft-confidence signals
      (1 - p_max, entropy, 1 - agreement) for predicting "the draft is wrong"; vertical line at 0.5 = no signal.
  (b) vertical grouped bars: S_1's dev accuracy vs. a confidence-threshold router (1 - p_max, threshold chosen so the
      Verifier call rate equals S_1's) vs. always calling the Verifier.
Table: AUROC of all four signals, router accuracy at S_1's call budget for all four signals, S_1 and always-Verifier.
All numbers come from data.json (data.self_check); nothing is typed by hand."""
import json
import sys
from pathlib import Path

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + ""
sys.path.insert(0, ROOT)
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = style.C
d = json.load(open(Path(ROOT) / "data.json"))
SC = d["self_check"]
B = style.BENCH_ORDER
L = style.BENCH_LABEL
N_DEV = d["meta"]["n_dev"]

# ---- sanity: the self-check block is S_1 on the dev pools --------------------------------------------------------
for b in B:
    s = SC[b]
    assert s["n"] == N_DEV[b], (b, s["n"], N_DEV[b])
    r1 = d["dev_trajectory"][b]["v3"]["dynamic"]["1"]              # round 1 of the v3 chain = S_1 on dev
    assert abs(s["policy_acc"] - r1["acc"]) < 0.06 and abs(s["policy_calls"] - r1["calls"]) < 0.006, (b, s, r1)
    assert abs(s["always_verifier"] - d["oracle_dev"][b]["always_verifier"]) < 0.06, b

SIGNALS = ["1-p_max", "-margin", "entropy", "1-agreement"]           # order in data.json / table
SIG_TEX = {"1-p_max": "$1-p_{\\max}$", "-margin": "$-$margin", "entropy": "Entropy", "1-agreement": "$1-$agreement"}
SIG_MD = {"1-p_max": "1 − p_max", "-margin": "−margin", "entropy": "Entropy", "1-agreement": "1 − agreement"}
FIG_SIGNALS = ["1-p_max", "entropy", "1-agreement"]                   # panel (a)
SIG_FILL = {"1-p_max": C["dark"], "entropy": C["mid"], "1-agreement": C["slate"]}
ROUTER = "1-p_max"                                                    # panel (b)

auroc = {s: [SC[b]["auroc"][s] for b in B] for s in SIGNALS}
router_acc = {s: [SC[b]["router"][s]["acc_at_policy_budget"] for b in B] for s in SIGNALS}
s1_acc = [SC[b]["policy_acc"] for b in B]
s1_calls = [SC[b]["policy_calls"] for b in B]
av_acc = [SC[b]["always_verifier"] for b in B]


def pol(key, **over):
    p = dict(style.POLICY[key]); p.pop("linestyle", None); p.pop("label", None); p.update(over)
    return p


def text_width_in(fig, txt, fontsize):
    t = fig.text(0, 0, txt, fontsize=fontsize); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width / fig.dpi


def wrap_lines(fig, paragraphs, fontsize, max_in):
    """Greedy word wrap on measured widths (mathtext-aware) so no caption line is wider than the figure."""
    out = []
    for para in paragraphs:
        line = ""
        for wd in para.split(" "):
            cand = (line + " " + wd).strip()
            if line and text_width_in(fig, cand, fontsize) > max_in:
                out.append(line); line = wd
            else:
                line = cand
        if line:
            out.append(line)
    return out


def group_labels(ax, xs, heights, texts, fontsize, color, base=0.01):
    """Value labels above the bars of one group. The outer labels always sit on their bars (two bar pitches apart they never
    touch); the middle label is lifted just enough to clear both neighbours when it would overlap them."""
    lo, hi = ax.get_ylim()
    ax_h_pt = ax.get_window_extent(ax.figure.canvas.get_renderer()).height * 72 / ax.figure.dpi
    need = fontsize * 1.2 / ax_h_pt                      # one label height, as a fraction of the axis
    frac = lambda v: (v - lo) / (hi - lo) + base
    tod = lambda f: lo + f * (hi - lo)
    fs = [frac(h) for h in heights]
    mid = len(xs) // 2
    for j in range(len(xs)):
        f = fs[j]
        if j == mid:
            for nb in (j - 1, j + 1):
                if 0 <= nb < len(xs) and abs(f - fs[nb]) < need:
                    f = max(f, fs[nb] + need)
        ax.text(xs[j], tod(f), texts[j], ha="center", va="bottom", fontsize=fontsize, color=color, zorder=5)


# ================================================================= FIGURE =====================================
FIG_W = 7.2
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, 4.05), gridspec_kw=dict(width_ratios=[1.0, 1.08], wspace=0.45))
LEFT, RIGHT = 0.115, 0.975
fig.subplots_adjust(left=LEFT, right=RIGHT, top=0.815, bottom=0.30)

# ---- (a) AUROC, horizontal grouped bars, MedQA on top ------------------------------------------------------------
style.horizontal_axes(ax_a)
y = np.arange(len(B))[::-1]                       # MedQA at the top
k = len(FIG_SIGNALS); h = 0.24
handles_a = []
for i, s in enumerate(FIG_SIGNALS):
    yi = y + ((k - 1) / 2 - i) * h                # first signal on top within each group
    bars = ax_a.barh(yi, auroc[s], h, facecolor=SIG_FILL[s], edgecolor=C["edge"], linewidth=0.8, zorder=3)
    for bar, v in zip(bars, auroc[s]):
        ax_a.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", ha="left", va="center", fontsize=8.5, color=C["num"], zorder=4)
    handles_a.append(style.policy_patch("base_only", label=SIG_TEX[s], facecolor=SIG_FILL[s], edgecolor=C["edge"], hatch=""))
ax_a.axvline(0.5, color=C["axis"], linewidth=1.0, linestyle=(0, (4, 2.5)), zorder=2)
pass  # 0.5 = no signal: said in the caption
ax_a.set_xlim(0, 0.88); ax_a.set_xticks([0, 0.25, 0.5, 0.75]); ax_a.set_xticklabels(["0", "0.25", "0.5", "0.75"])
ax_a.set_ylim(-0.55, len(B) - 0.45 + 0.12)
ax_a.set_yticks(y, [L[b] for b in B]); ax_a.tick_params(axis="y", labelsize=10.5)
ax_a.set_xlabel("AUROC (draft is wrong)", labelpad=4)
ax_a.spines["left"].set_color(C["axis"]); ax_a.spines["bottom"].set_color(C["axis"])
style.panel_tag(ax_a, "(a)", x=-0.22, y=1.0)
style.top_legend(ax_a, handles=handles_a, labels=[hh.get_label() for hh in handles_a], ncol=3, y=1.11, fontsize=9.5)

# ---- (b) accuracy at S_1's call budget, vertical grouped bars -----------------------------------------------------
SERIES_B = [
    ("S$_1$", pol("S_1"), s1_acc),
    ("Router", pol("base_system"), router_acc[ROUTER]),
    ("Always Verifier", pol("verifier"), av_acc),
]
x = np.arange(len(B)); kb = len(SERIES_B); w = 0.27
handles_b = []
ax_b.set_ylim(0, 100); ax_b.set_yticks([0, 25, 50, 75, 100])
for i, (lab, st, vals) in enumerate(SERIES_B):
    xi = x + (i - (kb - 1) / 2) * w
    ax_b.bar(xi, vals, w, **st, zorder=3)
    handles_b.append(style.policy_patch("base_only", label=lab, **st))
for bi in range(len(B)):                           # value labels per group, lifted where neighbours would touch
    xs = [x[bi] + (i - (kb - 1) / 2) * w for i in range(kb)]
    hs = [vals[bi] for _, _, vals in SERIES_B]
    group_labels(ax_b, xs, hs, [f"{v:.1f}" for v in hs], fontsize=8.3, color=C["num"])
ax_b.set_xlim(-0.55, len(B) - 0.45)
ax_b.set_xticks(x, [f"{L[b]}\n{c:.2f} calls" for b, c in zip(B, s1_calls)]); ax_b.tick_params(axis="x", length=0, labelsize=9.5)
ax_b.set_ylabel("Dev accuracy (%)")
style.panel_tag(ax_b, "(b)", x=-0.17, y=1.0)
style.top_legend(ax_b, handles=handles_b, labels=[hh.get_label() for hh in handles_b], ncol=3, y=1.11, fontsize=9.5)

CAP_FS = 9.0
x0 = LEFT - 0.075
cap_lines = wrap_lines(fig, [
    f"S$_1$ on the dev pools (MedQA/MMLU-Pro {N_DEV['medqa']}, GPQA {N_DEV['gpqa']}, AQuA {N_DEV['aqua']}). "
    "Router: call the Verifier when the draft-letter confidence is below a threshold chosen to match S$_1$'s call rate.",
    "(a) AUROC of each signal for predicting that the draft is wrong; 0.5 = no signal. "
    "(b) Dev accuracy at S$_1$'s call rate (shown under each benchmark); the router uses $1-p_{\\max}$.",
], CAP_FS, (RIGHT - x0) * FIG_W)
print("caption lines:", len(cap_lines))
style.caption(fig, cap_lines, y=0.005, x=x0, fontsize=CAP_FS)
# ---- overhang check: nothing should stick out of the 7.2 in canvas (bbox_inches="tight" would widen the figure) ----
fig.canvas.draw(); rend = fig.canvas.get_renderer(); W, H = fig.get_size_inches()
arts = list(fig.texts) + [t for ax in (ax_a, ax_b) for t in ax.texts] + [ax.get_legend() for ax in (ax_a, ax_b)]
arts += [lab for ax in (ax_a, ax_b) for lab in ax.get_xticklabels() + ax.get_yticklabels()] + [ax.xaxis.label for ax in (ax_a, ax_b)] + [ax.yaxis.label for ax in (ax_a, ax_b)]
for a in arts:
    if a is None or not a.get_visible() or (hasattr(a, "get_text") and not a.get_text()):
        continue
    bb = a.get_window_extent(rend); x0i, x1i = bb.x0 / fig.dpi, bb.x1 / fig.dpi; y0i, y1i = bb.y0 / fig.dpi, bb.y1 / fig.dpi
    if x0i < 0 or x1i > W or y0i < 0 or y1i > H:
        txt = a.get_text() if hasattr(a, "get_text") else type(a).__name__
        print(f"OVERHANG {txt[:40]!r}: x {x0i:.2f}..{x1i:.2f} in, y {y0i:.2f}..{y1i:.2f} in (canvas {W} x {H})")
print(style.save(fig, "fig_selfcheck"))

# ================================================================= TABLE ======================================
TAB = Path(ROOT) / "tab"; TAB.mkdir(exist_ok=True)


def f3(v):
    return f"{v:.3f}"


def f1(v):
    return f"{v:.1f}"


rows = []
for bi, b in enumerate(B):
    s = SC[b]
    at_budget = {sig: router_acc[sig][bi] for sig in SIGNALS}
    cand = dict(at_budget); cand["S_1"] = s1_acc[bi]              # same call budget: routers and S_1
    best = max(cand.values())
    rows.append(dict(name=L[b], n=s["n"], auroc={sig: auroc[sig][bi] for sig in SIGNALS}, at_budget=at_budget,
                     s1=s1_acc[bi], calls=s1_calls[bi], av=av_acc[bi], best={kk for kk, v in cand.items() if abs(v - best) < 1e-9}))

n_txt = f"MedQA {N_DEV['medqa']}, MMLU-Pro {N_DEV['mmlu_pro']}, GPQA {N_DEV['gpqa']}, AQuA-RAT {N_DEV['aqua']}"
cap_tex = ("Self-check signals of the S$_1$ manager on the development pools (" + n_txt + "). "
           "AUROC: how well each draft-confidence signal ($1-p_{\\max}$, $-$margin, entropy, $1-$agreement; "
           "higher means less confident in the draft letter) predicts that the draft is wrong; 0.5 is no signal. "
           "Router: call the Verifier when the signal exceeds a threshold chosen so that the call rate matches S$_1$'s "
           "(Calls, advisor calls per question); accuracy in \\%. "
           "S$_1$ is the learned policy at that call rate; Always Verifier calls the Verifier on every question (1.00 calls). "
           "Bold: best accuracy at the matched call budget (routers and S$_1$); Always Verifier uses a larger budget and is not bolded.")

tex = []
tex.append("% Generated by scripts/fig_selfcheck.py from data.json -- do not edit by hand.  Requires \\usepackage{booktabs}.")
tex.append("% ~6.4in wide at \\footnotesize (fits a 6.5-7in text width); for a 5.5in text width wrap the tabular in \\resizebox{\\textwidth}{!}{...} or use \\scriptsize.")
tex.append("\\begin{table*}[t]")
tex.append("\\centering")
tex.append("\\footnotesize")
tex.append("\\setlength{\\tabcolsep}{3.5pt}")
tex.append("\\renewcommand{\\arraystretch}{1.1}")
tex.append("\\caption{" + cap_tex + "}")
tex.append("\\label{tab:selfcheck}")
tex.append("\\begin{tabular}{l r rrrr rrrr r r r}")
tex.append("\\toprule")
tex.append(" & & \\multicolumn{4}{c}{AUROC (draft is wrong)} & \\multicolumn{4}{c}{Router acc.\\ at S$_1$'s call rate} & "
           "\\multicolumn{2}{c}{S$_1$} & Always \\\\")
tex.append("\\cmidrule(lr){3-6} \\cmidrule(lr){7-10} \\cmidrule(lr){11-12}")
tex.append("Benchmark & $n$ & " + " & ".join(SIG_TEX[s] for s in SIGNALS) + " & " + " & ".join(SIG_TEX[s] for s in SIGNALS)
           + " & Acc. & Calls & Verifier \\\\")
tex.append("\\midrule")
for r in rows:
    cells = [r["name"], str(r["n"])] + [f3(r["auroc"][s]) for s in SIGNALS]
    for s in SIGNALS:
        v = f1(r["at_budget"][s]); cells.append("\\textbf{" + v + "}" if s in r["best"] else v)
    v = f1(r["s1"]); cells.append("\\textbf{" + v + "}" if "S_1" in r["best"] else v)
    cells += [f"{r['calls']:.2f}", f1(r["av"])]
    tex.append(" & ".join(cells) + " \\\\")
tex.append("\\bottomrule")
tex.append("\\end{tabular}")
tex.append("\\end{table*}")
(TAB / "tab_selfcheck.tex").write_text("\n".join(tex) + "\n")

# ---- Markdown twin ------------------------------------------------------------------------------------------
cap_md = (cap_tex.replace("S$_1$", "S_1").replace("$p_{\\max}$", "p_max").replace("\\%", "%").replace("\\ ", " "))
md = ["**Table: tab_selfcheck.** " + cap_md, ""]
hdr = (["Benchmark", "n"] + [f"AUROC {SIG_MD[s]}" for s in SIGNALS] + [f"Router {SIG_MD[s]}" for s in SIGNALS]
       + ["S_1 acc.", "S_1 calls", "Always Verifier"])
md.append("| " + " | ".join(hdr) + " |")
md.append("|:--|" + "--:|" * (len(hdr) - 1))
for r in rows:
    cells = [r["name"], str(r["n"])] + [f3(r["auroc"][s]) for s in SIGNALS]
    for s in SIGNALS:
        v = f1(r["at_budget"][s]); cells.append(f"**{v}**" if s in r["best"] else v)
    v = f1(r["s1"]); cells.append(f"**{v}**" if "S_1" in r["best"] else v)
    cells += [f"{r['calls']:.2f}", f1(r["av"])]
    md.append("| " + " | ".join(cells) + " |")
(TAB / "tab_selfcheck.md").write_text("\n".join(md) + "\n")
print(TAB / "tab_selfcheck.tex"); print(TAB / "tab_selfcheck.md")

# ---- key numbers for the verifier ---------------------------------------------------------------------------
for r in rows:
    print(r["name"], "n=", r["n"], "AUROC", {s: r["auroc"][s] for s in SIGNALS}, "router@budget", {s: round(r["at_budget"][s], 1) for s in SIGNALS},
          "S1", r["s1"], "calls", r["calls"], "AV", r["av"], "best", sorted(r["best"]))
