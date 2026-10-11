"""tab_grpo_ablation: locked-test accuracy (mean calls) of the dynamic and static RSI arms with and without FA-GRPO,
paired-bootstrap deltas, FA-GRPO acceptance counts per arm and a rejection-reason summary line per benchmark.
Writes tab/tab_grpo_ablation.tex (booktabs) and tab/tab_grpo_ablation.md. All numbers come from data.json."""
import sys, json
from pathlib import Path
ROOT = Path(str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "")
sys.path.insert(0, str(ROOT))
import style

d = json.load(open(ROOT / "data.json"))
TAB = ROOT / "tab"; TAB.mkdir(parents=True, exist_ok=True)

ROWS = [("medqa", "test"), ("mmlu_pro", "test"), ("mmlu_pro", "test_paper"), ("gpqa", "test"), ("aqua", "test")]
ARMS = [("dynamic", "v3/dynamic_sft", "v3g/dynamic", "v3g/dynamic - v3/dynamic_sft"),
        ("static", "v3/static_sft", "v3g/static", "v3g/static - v3/static_sft")]
GRPO_ARMS = ["dynamic", "static", "success"]


def row_name(b, pool, latex):
    name = style.BENCH_LABEL[b]
    if pool != "test" or len(d["meta"]["pools"][b]) > 1:
        name += f" ({pool.replace('_', chr(92) + '_') if latex else pool})"
    return name


def fmt_p(p, latex):
    if p < 0.001:
        return "$<$0.001" if latex else "<0.001"
    return f"{p:.3f}"


def fmt_delta(dacc, p, latex):
    s = f"{dacc:+.1f}"
    return (f"${s}$" if latex else s) + f" ({fmt_p(p, latex)})"


def classify(reason):
    r = reason.lower()
    if "uninformative" in r:
        return "uninformative"
    if r.startswith("accuracy"):
        return "accuracy"
    return "gate"


def rejection_summary(b):
    g = d["grpo"][b]
    dec = g["decisions"]
    assert len(dec) == g["total"] and sum(x["accepted"] for x in dec) == g["accepted_total"], b
    rejected = [x for x in dec if not x["accepted"]]
    counts = {c: sum(c in {classify(r) for r in x["reasons"]} for x in rejected) for c in ("uninformative", "accuracy", "gate")}
    label = {"uninformative": "uninformative stage", "accuracy": "accuracy gate", "gate": "dev gate"}
    parts = [f"{label[c]} {n}" for c, n in counts.items() if n]
    txt = f"{g['accepted_total']} of {g['total']} FA-GRPO stage decisions accepted"
    txt += f"; {len(rejected)} rejected (" + ", ".join(parts) + ")." if rejected else "; none rejected."
    return txt, counts, len(rejected)


# ---------------------------------------------------------------- collect cells
table = []     # list of dicts, one per row
for b, pool in ROWS:
    key = f"{b}:{pool}"
    L, P = d["locked"][key], d["pairs"][key]
    cells = {"bench": b, "pool": pool, "n": L["v3g/dynamic"]["n"]}
    accs = {}
    for arm, k_sft, k_grpo, pair in ARMS:
        for tag, k in (("sft", k_sft), ("grpo", k_grpo)):
            cells[f"{arm}_{tag}_acc"] = L[k]["acc"]; cells[f"{arm}_{tag}_calls"] = L[k]["calls"]
            accs[f"{arm}_{tag}"] = L[k]["acc"]
        cells[f"{arm}_dacc"] = P[pair]["d_acc"]; cells[f"{arm}_p"] = P[pair]["p"]
        assert abs(P[pair]["d_acc"] - (L[k_grpo]["acc"] - L[k_sft]["acc"])) < 0.051, (key, arm)
    best = max(accs.values())
    cells["best"] = {k for k, v in accs.items() if abs(v - best) < 1e-9}
    if pool == "test":
        cells["accepted"] = {a: (d["grpo"][b]["per_arm"][a]["accepted"], d["grpo"][b]["per_arm"][a]["total"]) for a in GRPO_ARMS}
        cells["summary"] = rejection_summary(b)
    table.append(cells)


def acc_cell(c, arm, tag, latex):
    a = f"{c[f'{arm}_{tag}_acc']:.1f}"
    if f"{arm}_{tag}" in c["best"]:
        a = f"\\textbf{{{a}}}" if latex else f"**{a}**"
    return f"{a} ({c[f'{arm}_{tag}_calls']:.2f})"


n = d["meta"]["n_test"]
caption = ("FA-GRPO ablation on the locked tests (MedQA {medqa}, MMLU-Pro {mmlu_pro} test, GPQA {gpqa}, AQuA-RAT {aqua}). "
           "Cells: accuracy in \\% (mean advisor calls per question) of the round-3 manager trained with SFT only (v3 chain) "
           "or with SFT + FA-GRPO in every round (v3+GRPO). $\\Delta$ = with minus without GRPO in accuracy points, "
           "$p$ from a two-sided paired bootstrap. Bold: best accuracy in the row. Accepted: FA-GRPO stage decisions "
           "accepted out of three rounds per arm (round 1 is shared by the three arms except on MMLU-Pro, which has one run per arm); "
           "the MMLU-Pro test\\_paper row evaluates the same managers on the appendix pool, which overlaps the training pools. "
           "Rejection reasons may co-occur; the accuracy gate is dev accuracy below S$_k$ minus 1 point, the dev gate the valid-answer-rate check."
           ).format(**n)

# ---------------------------------------------------------------- LaTeX
tex = []
tex.append("\\begin{table*}[t]")
tex.append("\\centering\\footnotesize")
tex.append("\\caption{" + caption + "}")
tex.append("\\label{tab:grpo_ablation}")
tex.append("\\setlength{\\tabcolsep}{4pt}")
tex.append("\\begin{tabular}{l ccc ccc ccc}")
tex.append("\\toprule")
tex.append(" & \\multicolumn{3}{c}{RSI (dynamic)} & \\multicolumn{3}{c}{RSI (static)} & \\multicolumn{3}{c}{FA-GRPO accepted} \\\\")
tex.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}")
tex.append("Benchmark & SFT only & + FA-GRPO & $\\Delta$ ($p$) & SFT only & + FA-GRPO & $\\Delta$ ($p$) & Dyn. & Static & Succ. \\\\")
tex.append("\\midrule")
for c in table:
    cols = [row_name(c["bench"], c["pool"], True)]
    for arm, *_ in ARMS:
        cols += [acc_cell(c, arm, "sft", True), acc_cell(c, arm, "grpo", True), fmt_delta(c[f"{arm}_dacc"], c[f"{arm}_p"], True)]
    if "accepted" in c:
        cols += [f"{a}/{t}" for a, t in (c["accepted"][x] for x in GRPO_ARMS)]
    else:
        cols += ["--", "--", "--"]
    tex.append(" & ".join(cols) + " \\\\")
    if "summary" in c:
        tex.append("\\multicolumn{10}{l}{\\quad\\footnotesize\\textit{" + c["summary"][0] + "}} \\\\")
tex.append("\\bottomrule")
tex.append("\\end{tabular}")
tex.append("\\end{table*}")
(TAB / "tab_grpo_ablation.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------- Markdown twin
md_caption = (caption.replace("\\%", "%").replace("$\\Delta$", "Δ").replace("$p$", "p").replace("test\\_paper", "test_paper")
              .replace("S$_k$", "S_k"))
md = ["**Table: FA-GRPO ablation.** " + md_caption, ""]
hdr = ["Benchmark", "Dynamic, SFT only", "Dynamic, + FA-GRPO", "Dynamic Δ (p)", "Static, SFT only", "Static, + FA-GRPO", "Static Δ (p)",
       "Accepted dyn.", "Accepted static", "Accepted succ."]
md.append("| " + " | ".join(hdr) + " |")
md.append("|" + "|".join([":--"] + [":--:"] * (len(hdr) - 1)) + "|")
for c in table:
    cols = [row_name(c["bench"], c["pool"], False)]
    for arm, *_ in ARMS:
        cols += [acc_cell(c, arm, "sft", False), acc_cell(c, arm, "grpo", False), fmt_delta(c[f"{arm}_dacc"], c[f"{arm}_p"], False)]
    if "accepted" in c:
        cols += [f"{a}/{t}" for a, t in (c["accepted"][x] for x in GRPO_ARMS)]
    else:
        cols += ["--", "--", "--"]
    md.append("| " + " | ".join(cols) + " |")
md.append("")
md.append("Rejection reasons per benchmark:")
md.append("")
for c in table:
    if "summary" in c:
        md.append(f"- {style.BENCH_LABEL[c['bench']]}: {c['summary'][0]}")
(TAB / "tab_grpo_ablation.md").write_text("\n".join(md) + "\n")

print((TAB / "tab_grpo_ablation.tex"), (TAB / "tab_grpo_ablation.md"))
for c in table:
    print(row_name(c["bench"], c["pool"], False), {k: v for k, v in c.items() if k not in ("best", "summary", "bench", "pool")},
          c.get("summary", ("", None, None))[1:])
