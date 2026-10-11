"""fig_single_agent + tab_single_agent: single-model baselines vs. the multi-agent systems on the locked tests.

Figure (7.2 in, two panels like reference 2.jpg): (a) accuracy, (b) total tokens per question (log axis) for the same bars.
Table: every single-agent variant per benchmark (accuracy, valid, forced, hit-limit, tokens) plus the multi-agent systems as
reference rows; pending cells are written as "pending". All numbers come from data.json.
"""
import json
import os
import re
import sys

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + ""
sys.path.insert(0, ROOT)
import style  # noqa: E402

style.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

d = json.load(open(f"{ROOT}/data.json"))
B = style.BENCH_ORDER
N = d["meta"]["n_test"]
POOL = "test"
L = style.BENCH_LABEL


# ----------------------------------------------------------------------------------------------------------------------
# accessors
def sa(b, variant):
    return d["single_agent"][f"{b}:{POOL}"].get(variant)


def locked(b, key):
    return d["locked"][f"{b}:{POOL}"].get(key)


def tokens(b, labels):
    """First available token entry among `labels`; returns (total, label used) or (None, None)."""
    t = d["tokens"][f"{b}:{POOL}"]
    for lab in labels:
        if lab in t and t[lab].get("total") is not None:
            return t[lab]["total"], lab
    return None, None


def pol(key, **over):
    p = dict(style.POLICY[key]); p.pop("linestyle", None); p.pop("label", None); p.update(over)
    return p


def join(names):
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else (names[0] if names else "")


S1_TOK = ["v3g/S_1", "v3/S_1", "main/S_1"]            # same S_1 checkpoint; v3g re-measurement first
SUC_TOK = ["v3g/success", "abl/success", "main/success"]  # v3g where measured, else the v1 success runs (same always-call policy)

# The seven bars of the figure, in order. Each entry: legend label, bar style, accuracy getter, token getter.
SERIES = [
    dict(label="Base, letter only", style=pol("base_only"),
         acc=lambda b: (sa(b, "base_nocot_f") or {}).get("acc"), tok=lambda b: ((sa(b, "base_nocot_f") or {}).get("total_tok"), "single_agent/base_nocot_f")),
    dict(label="S$_1$, letter only", style=pol("S_1", hatch="xx"),
         acc=lambda b: (sa(b, "s1_nocot_f") or {}).get("acc"), tok=lambda b: ((sa(b, "s1_nocot_f") or {}).get("total_tok"), "single_agent/s1_nocot_f")),
    dict(label="S$_1$ draft, no tools", style=pol("S_1", hatch="//"),
         acc=lambda b: (sa(b, "s1_manager_notools") or {}).get("acc"), tok=lambda b: ((sa(b, "s1_manager_notools") or {}).get("total_tok"), "single_agent/s1_manager_notools")),
    dict(label="S$_1$ + advisors", style=pol("S_1"),
         acc=lambda b: locked(b, "v3g/S_1")["acc"], tok=lambda b: tokens(b, S1_TOK)),
    dict(label="RSI (dynamic)", style=pol("dynamic"),
         acc=lambda b: locked(b, "v3g/dynamic")["acc"], tok=lambda b: tokens(b, ["v3g/dynamic"])),
    dict(label="RSI (success)", style=pol("success"),
         acc=lambda b: locked(b, "v3g/success")["acc"], tok=lambda b: tokens(b, SUC_TOK)),
    dict(label="Base system", style=pol("base_system"),
         acc=lambda b: d["base_system"][f"{b}:{POOL}"]["acc"], tok=lambda b: tokens(b, ["base/manager"])),
]

ACC = {s["label"]: [s["acc"](b) for b in B] for s in SERIES}
TOK = {s["label"]: [s["tok"](b)[0] for b in B] for s in SERIES}
TOK_SRC = {s["label"]: [s["tok"](b)[1] for b in B] for s in SERIES}
s1_fallback = [L[b] for b, lab in zip(B, TOK_SRC["S$_1$ + advisors"]) if lab != "v3g/S_1"]
suc_fallback = [L[b] for b, lab in zip(B, TOK_SRC["RSI (success)"]) if lab != "v3g/success"]


# ----------------------------------------------------------------------------------------------------------------------
# figure helpers
def text_width_pt(fig, s, fontsize, **kw):
    t = fig.text(0, 0, s, fontsize=fontsize, **kw); bb = t.get_window_extent(fig.canvas.get_renderer()); t.remove()
    return bb.width * 72 / fig.dpi


def fit_fontsize(fig, samples, max_pt, start=7.5, floor=5.5):
    """Largest font size <= start at which every sample string is narrower than max_pt."""
    fs = start
    while fs > floor and max(text_width_pt(fig, s, fs) for s in samples) > max_pt:
        fs -= 0.1
    return round(fs, 1)


def staggered_labels(ax, xs, heights, texts, fontsize, color, log=False, base=0.012, stagger=True, rotation=0):
    """Value labels above bars. Labels are narrower than the bar pitch, so neighbours never overlap; a label is lifted by
    one label height only when it would sit at (nearly) the same height as its left neighbour's label, so runs of equal
    bars read as 74.6 / 74.4 / 73.8 instead of a single string of digits (two levels only, so a label stays by its bar)."""
    lo, hi = ax.get_ylim()
    if log:
        lo, hi = np.log10(lo), np.log10(hi)
        frac = lambda v: (np.log10(v) - lo) / (hi - lo)
        tod = lambda f: 10 ** (lo + f * (hi - lo))
    else:
        frac = lambda v: (v - lo) / (hi - lo)
        tod = lambda f: lo + f * (hi - lo)
    ax_h_pt = ax.get_window_extent(ax.figure.canvas.get_renderer()).height * 72 / ax.figure.dpi
    step = fontsize * 1.25 / ax_h_pt  # one label height (plus a little air) in axes fraction
    prev = None
    for x, h, t in zip(xs, heights, texts):
        if h is None:
            prev = None
            continue
        f0 = frac(h) + base
        f = f0
        if prev is not None and stagger:
            for lvl in (0, 1):
                f = f0 + lvl * step
                if abs(f - prev) >= 0.5 * step:
                    break
        ax.text(x, tod(f), t, ha="center", va="bottom", fontsize=fontsize, color=color, rotation=rotation)
        prev = f


def wrap_lines(fig, paragraphs, fontsize, max_pt):
    """Greedy word wrap using measured text widths (mathtext-aware)."""
    out = []
    for para in paragraphs:
        words = para.split(" "); line = ""
        for wd in words:
            cand = (line + " " + wd).strip()
            if line and text_width_pt(fig, cand, fontsize) > max_pt:
                out.append(line); line = wd
            else:
                line = cand
        if line:
            out.append(line)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# figure
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.2, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[3.0, 1.6], hspace=0.2))
LEFT, RIGHT = 0.085, 0.995
fig.subplots_adjust(left=LEFT, right=RIGHT, top=0.86, bottom=0.11)
x = np.arange(len(B)); k = len(SERIES); w = 0.112
XLIM = (-0.55, len(B) - 0.45)
handles = []
for i, s in enumerate(SERIES):
    xi = x + (i - (k - 1) / 2) * w
    acc = ACC[s["label"]]; tok = TOK[s["label"]]
    keep = [j for j, v in enumerate(acc) if v is not None]
    ax_a.bar(xi[keep], [acc[j] for j in keep], w, **s["style"])
    keep_t = [j for j, v in enumerate(tok) if v is not None]
    ax_b.bar(xi[keep_t], [tok[j] for j in keep_t], w, **s["style"])
    handles.append(style.policy_patch("base_only", label=s["label"], **s["style"]))

ax_a.set_xlim(*XLIM)
pitch_pt = w * ax_a.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 72 / (XLIM[1] - XLIM[0])
fs_a = fit_fontsize(fig, [f"{v:.1f}" for vals in ACC.values() for v in vals if v is not None], pitch_pt * 0.98)
fs_b = fit_fontsize(fig, ["0"], pitch_pt, start=7.0)  # rotated labels only need their height to fit
print(f"bar pitch {pitch_pt:.1f} pt; label font (a) {fs_a} pt, (b) {fs_b} pt")

# panel (a): accuracy
ax_a.set_ylim(0, 100); ax_a.set_yticks([0, 25, 50, 75, 100]); ax_a.set_ylabel("Accuracy (%)")
for bi in range(len(B)):
    xs = [x[bi] + (i - (k - 1) / 2) * w for i in range(k)]
    hs = [ACC[s["label"]][bi] for s in SERIES]
    staggered_labels(ax_a, xs, hs, [f"{v:.1f}" if v is not None else "" for v in hs], fontsize=fs_a, color=style.C["num"], base=0.012)
ax_a.tick_params(axis="x", length=0)

# panel (b): tokens per question, log axis, rotated gray labels
ax_b.set_yscale("log"); ax_b.set_ylim(100, 60000); ax_b.set_yticks([100, 1000, 10000])
ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}")); ax_b.yaxis.set_minor_locator(NullLocator())
ax_b.set_ylabel("Tokens / question", labelpad=6)
for bi in range(len(B)):
    xs = [x[bi] + (i - (k - 1) / 2) * w for i in range(k)]
    hs = [TOK[s["label"]][bi] for s in SERIES]
    staggered_labels(ax_b, xs, hs, [f"{v:,.0f}" if v is not None else "" for v in hs], fontsize=fs_b, color=style.C["numgray"], log=True, base=0.025, stagger=False, rotation=90)
ax_b.set_xticks(x, [L[b] for b in B]); ax_b.tick_params(axis="x", length=0, labelsize=11.5)

style.panel_tag(ax_a, "(a)", x=-0.055, y=1.015); style.panel_tag(ax_b, "(b)", x=-0.055, y=1.015)
style.top_legend(ax_a, handles=handles, labels=[h.get_label() for h in handles], ncol=4, y=1.05, fontsize=10)

cap_pars = [  # first line on its own (pool and n), the rest flows as one wrapped paragraph
    "Locked tests: " + ", ".join(f"{L[b]} {N[b]}" + (" (test)" if b == "mmlu_pro" else "") for b in B) + ". (a) accuracy; (b) total tokens per question, log axis.",
    "Letter-only = model must answer with the letter (16-token budget, forced choice, 100% parseable); base system = untrained manager with tools. "
    "Tokens = prefill + output; S$_1$ + advisors on " + join(s1_fallback) + " from the v3-chain run of the same S$_1$; RSI (success) on " + join(suc_fallback) + " from the v1 success runs (same labelling rule, call rate 0.93-1.00).",
]
CAP_FS = 8.6
cap_lines = wrap_lines(fig, cap_pars, CAP_FS, (RIGHT - LEFT) * 7.2 * 72 * 0.995)
style.caption(fig, cap_lines, y=-0.03 - 0.035 * (len(cap_lines) - 1), x=LEFT, fontsize=CAP_FS)  # style.caption stacks lines 0.035 apart
print("caption lines:", len(cap_lines))
out = style.save(fig, "fig_single_agent"); print(out)


# ----------------------------------------------------------------------------------------------------------------------
# table
VARIANT_ROWS = [  # (variant key, display name)
    ("base_nocot_f", "Base, letter only"),
    ("s1_nocot_f", "S$_1$, letter only"),
    ("dyn3_nocot_f", "Dyn.\\ r3 (no GRPO), letter only"),
    ("s1_manager_notools", "S$_1$, no tools"),
    ("dyn3_manager_notools", "Dyn.\\ r3 (no GRPO), no tools"),
]
DESC = d["single_agent_variants"]


def budget_of(variant):
    """Token budget: from the run when present on any benchmark, else parsed from the variant description."""
    for b in B:
        r = sa(b, variant)
        if r and r.get("max_new_tokens"):
            return r["max_new_tokens"]
    m = re.search(r"(\d+)[- ]token", DESC[variant])
    return int(m.group(1)) if m else None


SYSTEM_ROWS = [  # display name, accuracy, valid, tokens (total, source label)
    ("S$_1$ + advisors", lambda b: locked(b, "v3g/S_1")["acc"], lambda b: locked(b, "v3g/S_1")["valid"], lambda b: tokens(b, S1_TOK)),
    ("RSI dynamic + GRPO (r3)", lambda b: locked(b, "v3g/dynamic")["acc"], lambda b: locked(b, "v3g/dynamic")["valid"], lambda b: tokens(b, ["v3g/dynamic"])),
    ("RSI success (r3)", lambda b: locked(b, "v3g/success")["acc"], lambda b: locked(b, "v3g/success")["valid"], lambda b: tokens(b, SUC_TOK)),
    ("Base system (untrained manager)", lambda b: d["base_system"][f"{b}:{POOL}"]["acc"], lambda b: d["base_system"][f"{b}:{POOL}"]["valid"], lambda b: tokens(b, ["base/manager"])),
]

pct = lambda v: f"{100 * v:.1f}"
acc1 = lambda v: f"{v:.1f}"
tok0 = lambda v: f"{v:,.0f}"

# cells[b][row_name] = (5 strings, raw accuracy); rows are the same for every benchmark
ROWS = [("single", v, name, budget_of(v)) for v, name in VARIANT_ROWS] + [("system", name, name, None) for name, *_ in SYSTEM_ROWS]
CELLS = {b: {} for b in B}
pending_variants = {}
for b in B:
    for v, name in VARIANT_ROWS:
        r = sa(b, v)
        if r is None:
            CELLS[b][v] = (["n/r"] * 5, None); pending_variants.setdefault(v, []).append(L[b])
        else:
            CELLS[b][v] = ([acc1(r["acc"]), pct(r["valid"]), pct(r["forced"]), pct(r["hit_limit"]), tok0(r["total_tok"])], r["acc"])
    for name, fa, fv, ft in SYSTEM_ROWS:
        t, src = ft(b)
        CELLS[b][name] = ([acc1(fa(b)), pct(fv(b)), "--", "--", tok0(t) if t is not None else "pending"], fa(b))

# bold = best accuracy per benchmark within each block (single model / multi-agent)
BOLD = {b: set() for b in B}
for b in B:
    for grp in ("single", "system"):
        keys = [k for g, k, _, _ in ROWS if g == grp]
        best = max((CELLS[b][k][1] for k in keys if CELLS[b][k][1] is not None), default=None)
        BOLD[b] |= {k for k in keys if CELLS[b][k][1] is not None and CELLS[b][k][1] == best}

pend_names = [L[b] for b in B if any(L[b] in v for v in pending_variants.values())]
caption = ("Single-model baselines and the multi-agent systems on the locked tests ("
           + ", ".join(f"{L[b]} {N[b]}" + (" (test pool)" if b == "mmlu_pro" else "") for b in B) + "). "
           "Acc. and Valid are percentages; Budget = generation budget in tokens; Tokens = prefill + output per question (for systems: manager and advisors). "
           "Letter-only rows must answer with the choice letter only (no reasoning); a forced choice over the option letters backs up unparsed outputs and was never needed. "
           "Valid = share of questions with a parseable answer. "
           "No-tools rows run the manager prompt with the tools removed (DRAFT then ANSWER, no advisor calls). "
           "Multi-agent rows are the v3+GRPO round-3 systems. "
           "Where the v3+GRPO run has no token count, S$_1$ + advisors uses the v3-chain run of the same S$_1$ checkpoint (" + join(s1_fallback) + ") "
           "and RSI success uses the v1 success runs, which use the same labelling rule and call an advisor on 93--100\\% of questions (" + join(suc_fallback) + "). "
           "Bold: best accuracy per benchmark within each block.")

METRICS = ["Acc.", "Valid", "Tokens"]
KEEP = [0, 1, 4]  # of (acc, valid, forced, limit, tokens)
HALVES = [B]
tex = ["\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
       f"\\caption{{{caption}}}", "\\label{tab:single_agent}",
       "\\begin{tabular}{l r " + " ".join(["rrr"] * len(B)) + "}", "\\toprule"]
md_name = lambda s_: s_.replace("S$_1$", "S₁").replace("\\ ", " ")
md = [f"**Table: tab_single_agent.** {md_name(caption).replace(chr(92) + '%', '%').replace('93--100', '93–100')}", ""]
for hi, half in enumerate(HALVES):
    if hi:
        tex.append("\\midrule")
    tex.append(" & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{L[b]} ($n$ = {N[b]})}}" for b in half) + " \\\\")
    tex.append(" ".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(half))))
    tex.append("Variant & Budget & " + " & ".join(METRICS * len(half)) + " \\\\")
    tex.append("\\midrule")
    md.append("| Variant | Budget | " + " | ".join(f"{L[b]} {m}" for b in half for m in METRICS) + " |")
    md.append("|:---|---:|" + "---:|" * (3 * len(half)))
    for grp, head in (("single", "Single model"), ("system", "Multi-agent systems")):
        tex.append(f"\\multicolumn{{{2 + 3 * len(half)}}}{{l}}{{\\textit{{{head}}}}} \\\\")
        md.append(f"| *{head}* |" + " |" * (1 + 3 * len(half)))
        for g, key, name, bud in ROWS:
            if g != grp:
                continue
            budget_s = f"{bud:,}" if bud else "--"
            tc, mc = [], []
            for b in half:
                c = [CELLS[b][key][0][i] for i in KEEP]; m = list(c)
                if key in BOLD[b]:
                    c[0] = f"\\textbf{{{c[0]}}}"; m[0] = f"**{m[0]}**"
                tc += c; mc += m
            tex.append("\\quad " + name + " & " + budget_s + " & " + " & ".join(tc) + " \\\\")
            md.append("| " + md_name(name) + " | " + budget_s + " | " + " | ".join(mc) + " |")
    md.append("")
tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

os.makedirs(f"{ROOT}/tab", exist_ok=True)
open(f"{ROOT}/tab/tab_single_agent.tex", "w").write("\n".join(tex) + "\n")
open(f"{ROOT}/tab/tab_single_agent.md", "w").write("\n".join(md) + "\n")
print(f"{ROOT}/tab/tab_single_agent.tex"); print(f"{ROOT}/tab/tab_single_agent.md")
print("TABLE", json.dumps({b: {k: CELLS[b][k][0] for k in CELLS[b]} for b in B}))

# key numbers for the verifier
print("ACC", json.dumps(ACC))
print("TOK", json.dumps(TOK))
print("TOK_SRC", json.dumps(TOK_SRC))
