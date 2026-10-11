#!/usr/bin/env python3
"""tab_main: main-results table of the MARGENT multi-round RSI paper (v3+GRPO, keys v3g/*) on the locked test pools,
plus the appendix twin tab_main_testpaper on MMLU-Pro test_paper.  Every number comes from data.json; nothing is typed
by hand.  Writes tab/tab_main.tex (both tables, booktabs) and tab/tab_main.md (Markdown twin with the same content)."""
import json
import sys
from pathlib import Path

ROOT = Path(str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "")
sys.path.insert(0, str(ROOT))
import style  # noqa: E402  (BENCH_ORDER / BENCH_LABEL / POLICY labels shared with the figures)

D = json.load(open(ROOT / "data.json"))
TAB = ROOT / "tab"
TAB.mkdir(parents=True, exist_ok=True)

POLICIES = ["S_1", "dynamic", "static", "success"]
LABEL_TEX = {p: style.POLICY[p]["label"] for p in POLICIES}              # "S$_1$", "RSI (dynamic)", ...
LABEL_MD = {p: style.POLICY[p]["label"].replace("S$_1$", "S₁") for p in POLICIES}
GRPO_SHORT = {"medqa": "MedQA", "mmlu_pro": "MMLU", "gpqa": "GPQA", "aqua": "AQuA"}
MAIN_BENCHES = [(b, f"{b}:test") for b in style.BENCH_ORDER]
APP_BENCHES = [("mmlu_pro", "mmlu_pro:test_paper")]
N_TEST = D["meta"]["n_test"]


# ----------------------------------------------------------------------------- data access
def tokens(key, pol):
    """Total tokens per question for the v3g run; S_1 falls back to the v3 re-evaluation of the same checkpoint.
    Returns (value or None, fallback_used)."""
    t = D["tokens"][key]
    k = f"v3g/{pol}"
    if k in t and t[k].get("total") is not None:
        return t[k]["total"], False
    if pol == "S_1" and "v3/S_1" in t and t["v3/S_1"].get("total") is not None:
        return t["v3/S_1"]["total"], True
    return None, False


def cell(key, pol):
    L = D["locked"][key][f"v3g/{pol}"]
    acc, calls = L["acc"], L["calls"]
    tok, fb = tokens(key, pol)
    delta = None
    if pol != "S_1":
        pr = D["pairs"][key][f"v3g/{pol} - v3g/S_1"]
        base = D["locked"][key]["v3g/S_1"]["acc"]
        assert abs(pr["d_acc"] - (acc - base)) < 0.06, (key, pol, pr["d_acc"], acc - base)
        delta = (pr["d_acc"], pr["p"])
    return dict(acc=acc, calls=calls, tok=tok, tok_fallback=fb, delta=delta, n=L["n"])


def grpo(bench, pol):
    pa = D["grpo"][bench]["per_arm"][pol]
    return f"{pa['accepted']}/{pa['total']}"


# ----------------------------------------------------------------------------- formatting
def f_acc(x):
    return f"{x:.1f}"


def f_calls(x):
    return f"{x:.2f}"


def f_tok(x):
    return f"{round(x):,}"


def f_p(p, tex):
    if round(p, 3) < 0.001:
        return "$<$0.001" if tex else "<0.001"
    return f"{p:.3f}"


def f_delta(d, tex):
    sign = "+" if d >= 0 else "-"
    body = f"{abs(d):.1f}"
    if tex:
        return f"${sign}{body}$"
    return ("+" if d >= 0 else "−") + body


def f_delta_p(delta, tex):
    d, p = delta
    return f"{f_delta(d, tex)} ({f_p(p, tex)})"


# ----------------------------------------------------------------------------- assemble cells
def build(benches):
    cells = {(key, p): cell(key, p) for _, key in benches for p in POLICIES}
    best = {key: max(cells[(key, p)]["acc"] for p in POLICIES) for _, key in benches}
    return cells, best


def pool_label(b, key, tex=False):
    return style.BENCH_LABEL[b] + ((" test\\_paper" if tex else " test_paper") if key.endswith(":test_paper") else "")


def missing_tokens_sentence(cells, benches, tex):
    """'Total tokens were not measured (--) for RSI (static) on MMLU-Pro and AQuA-RAT, and for RSI (success) on ...'
    Policies that miss the same benchmarks are grouped into one clause."""
    groups = {}  # tuple of missing bench labels -> [policy labels]
    for p in POLICIES:
        bs = tuple(pool_label(b, key, tex) for b, key in benches if cells[(key, p)]["tok"] is None)
        if bs:
            groups.setdefault(bs, []).append(LABEL_TEX[p] if tex else LABEL_MD[p])
    if not groups:
        return ""
    parts = [f"for {join_names(pols)} on {join_names(list(bs))}" for bs, pols in groups.items()]
    dash = "--" if tex else "\u2013"
    return f"Total tokens were not measured ({dash}) " + ", and ".join(parts) + "."


def fallback_sentence(cells, benches, tex):
    bs = [pool_label(b, key, tex) for b, key in benches if cells[(key, "S_1")]["tok_fallback"]]
    if not bs:
        return ""
    s1 = "S$_1$" if tex else "S₁"
    dag = "$\\dagger$" if tex else "†"
    return (f"{dag}: {s1} tokens on {join_names(bs)} were not re-measured in the v3+GRPO run; the value is from the "
            f"no-GRPO (v3) re-evaluation of the same {s1} checkpoint.")


def join_names(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def caption_main(cells, benches, tex):
    s1 = "S$_1$" if tex else "S₁"
    pools = ", ".join(f"{style.BENCH_LABEL[b]} {N_TEST[b]}" + (" (test)" if b == "mmlu_pro" else "") for b, _ in benches)
    delta = "$\\Delta$" if tex else "Δ"
    pp = "$p$" if tex else "p"
    lines = [
        "Main results of multi-round RSI (v3+GRPO, round 3) on the locked test pools.",
        f"Locked tests: {pools}.",
        ("Acc.: accuracy (\\%); " if tex else "Acc.: accuracy (%); ") + "Calls: advisor calls per question; Tok.: total tokens per question (manager and advisors, prefill and decode).",
        f"{delta} rows: accuracy change vs.\\ {s1} in points with the two-sided paired-bootstrap {pp} (10k resamples)." if tex else
        f"{delta} rows: accuracy change vs. {s1} in points with the two-sided paired-bootstrap {pp} (10k resamples).",
        "Bold: best accuracy per benchmark; RSI (success) calls an advisor on nearly every question (call rate at least 0.99).",
        "GRPO accepted: FA-GRPO stages accepted by the development gate out of three rounds; a rejected stage keeps that round's SFT manager.",
        missing_tokens_sentence(cells, benches, tex),
        fallback_sentence(cells, benches, tex),
    ]
    return " ".join(l for l in lines if l)


def caption_app(cells, benches, tex):
    s1 = "S$_1$" if tex else "S₁"
    delta = "$\\Delta$" if tex else "Δ"
    pp = "$p$" if tex else "p"
    n = N_TEST["mmlu_pro"]
    lines = [
        f"MMLU-Pro test\\_paper pool ({n} questions; v3+GRPO, round 3)." if tex else f"MMLU-Pro test_paper pool ({n} questions; v3+GRPO, round 3).",
        "This pool overlaps the training pools and is reported for completeness only; the locked MMLU-Pro result (test pool) is in " + ("Table~\\ref{tab:main}." if tex else "Table 1."),
        "Columns as in the main table; GRPO acceptances are the MMLU-Pro runs (the GRPO gate uses the development pool, not this one).",
        f"{delta} rows: accuracy change vs.\\ {s1} with the paired-bootstrap {pp}." if tex else f"{delta} rows: accuracy change vs. {s1} with the paired-bootstrap {pp}.",
        missing_tokens_sentence(cells, benches, tex),
        fallback_sentence(cells, benches, tex),
    ]
    return " ".join(l for l in lines if l)


# ----------------------------------------------------------------------------- LaTeX
def tex_table(benches, cells, best, grpo_benches, star, label, caption):
    env = "table*" if star else "table"
    ng = len(grpo_benches)
    ncol = 1 + 3 * len(benches) + ng
    colspec = "l" + "rrr" * len(benches) + "c" * ng
    out = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}", "\\renewcommand{\\arraystretch}{1.1}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    h1 = [""] + [f"\\multicolumn{{3}}{{c}}{{{style.BENCH_LABEL[b]}}}" for b, _ in benches]
    h1.append(f"\\multicolumn{{{ng}}}{{c}}{{GRPO accepted}}")
    out.append(" & ".join(h1) + " \\\\")
    cm, c = [], 2
    for _ in benches:
        cm.append(f"\\cmidrule(lr){{{c}-{c + 2}}}")
        c += 3
    cm.append(f"\\cmidrule(lr){{{c}-{c + ng - 1}}}")
    out.append(" ".join(cm))
    h2 = ["Policy"] + ["Acc.", "Calls", "Tok."] * len(benches) + [GRPO_SHORT[b] if ng > 1 else "of 3 rounds" for b in grpo_benches]
    out.append(" & ".join(h2) + " \\\\")
    out.append("\\midrule")
    for i, p in enumerate(POLICIES):
        row = [LABEL_TEX[p]]
        for b, key in benches:
            cd = cells[(key, p)]
            acc = f_acc(cd["acc"])
            if cd["acc"] == best[key]:
                acc = f"\\textbf{{{acc}}}"
            tok = "--" if cd["tok"] is None else f_tok(cd["tok"]) + ("$^{\\dagger}$" if cd["tok_fallback"] else "")
            row += [acc, f_calls(cd["calls"]), tok]
        row += ["--" if p == "S_1" else grpo(b, p) for b in grpo_benches]
        assert len(row) == ncol
        out.append(" & ".join(row) + " \\\\")
        if p != "S_1":
            drow = ["\\quad $\\Delta$ vs S$_1$ ($p$)"]
            for b, key in benches:
                drow.append(f"\\multicolumn{{3}}{{c}}{{{f_delta_p(cells[(key, p)]['delta'], tex=True)}}}")
            drow += [""] * ng
            assert 1 + 3 * (len(drow) - 1 - ng) + ng == ncol
            out.append(" & ".join(drow) + " \\\\")
        if i < len(POLICIES) - 1:
            out.append("\\addlinespace")
    out += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(out)


# ----------------------------------------------------------------------------- Markdown
def md_table(benches, cells, best, grpo_benches, title, caption):
    ng = len(grpo_benches)
    hdr = ["Policy"]
    for b, _ in benches:
        hdr += [f"{style.BENCH_LABEL[b]} Acc.", f"{style.BENCH_LABEL[b]} Calls", f"{style.BENCH_LABEL[b]} Tok."]
    hdr += [f"GRPO accepted ({style.BENCH_LABEL[b]})" for b in grpo_benches]
    out = [f"**{title}** {caption}", "", "| " + " | ".join(hdr) + " |", "|:--|" + "--:|" * (len(hdr) - 1)]
    for p in POLICIES:
        row = [LABEL_MD[p]]
        for b, key in benches:
            cd = cells[(key, p)]
            acc = f_acc(cd["acc"])
            if cd["acc"] == best[key]:
                acc = f"**{acc}**"
            tok = "–" if cd["tok"] is None else f_tok(cd["tok"]) + ("†" if cd["tok_fallback"] else "")
            row += [acc, f_calls(cd["calls"]), tok]
        row += ["–" if p == "S_1" else grpo(b, p) for b in grpo_benches]
        out.append("| " + " | ".join(row) + " |")
        if p != "S_1":
            drow = ["&nbsp;&nbsp;Δ vs S₁ (p)"]
            for b, key in benches:
                drow += [f_delta_p(cells[(key, p)]["delta"], tex=False), "", ""]
            drow += [""] * ng
            out.append("| " + " | ".join(drow) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    cells_m, best_m = build(MAIN_BENCHES)
    cells_a, best_a = build(APP_BENCHES)
    grpo_m = [b for b, _ in MAIN_BENCHES]
    grpo_a = [b for b, _ in APP_BENCHES]

    tex = "\n".join([
        "% Generated by scripts/tab_main.py from data.json -- do not edit by hand.  Requires \\usepackage{booktabs}.",
        "% tab_main: main results (v3+GRPO, locked tests).  tab_main_testpaper: appendix twin on MMLU-Pro test_paper.",
        "% tab_main is ~6.3in wide at \\footnotesize (fits a 6.5-7in text width); for a 5.5in text width switch to \\scriptsize or wrap the tabular in \\resizebox{\\textwidth}{!}{...}.",
        "",
        tex_table(MAIN_BENCHES, cells_m, best_m, grpo_m, star=True, label="tab:main",
                  caption=caption_main(cells_m, MAIN_BENCHES, tex=True)),
        "",
        "",
        tex_table(APP_BENCHES, cells_a, best_a, grpo_a, star=False, label="tab:main_testpaper",
                  caption=caption_app(cells_a, APP_BENCHES, tex=True)),
        "",
    ])
    (TAB / "tab_main.tex").write_text(tex)

    md = "\n".join([
        "<!-- Generated by scripts/tab_main.py from data.json; Markdown twin of tab_main.tex -->",
        "",
        "## tab_main",
        "",
        md_table(MAIN_BENCHES, cells_m, best_m, grpo_m, "Table 1.", caption_main(cells_m, MAIN_BENCHES, tex=False)),
        "",
        "## tab_main_testpaper (appendix)",
        "",
        md_table(APP_BENCHES, cells_a, best_a, grpo_a, "Table A1.", caption_app(cells_a, APP_BENCHES, tex=False)),
        "",
    ])
    (TAB / "tab_main.md").write_text(md)

    # compact summary for the verifier
    for name, benches, cells in (("tab_main", MAIN_BENCHES, cells_m), ("tab_main_testpaper", APP_BENCHES, cells_a)):
        print(f"== {name}")
        for b, key in benches:
            for p in POLICIES:
                cd = cells[(key, p)]
                tok = "-" if cd["tok"] is None else f_tok(cd["tok"]) + ("(v3/S_1 fallback)" if cd["tok_fallback"] else "")
                d = "" if cd["delta"] is None else f"  d={f_delta_p(cd['delta'], tex=False)}"
                g = "-" if p == "S_1" else grpo(b, p)
                print(f"  {key:22s} {p:8s} acc={f_acc(cd['acc']):>5s} calls={f_calls(cd['calls'])} tok={tok:>24s} grpo={g}{d}")
    print("wrote", TAB / "tab_main.tex", "and", TAB / "tab_main.md")
