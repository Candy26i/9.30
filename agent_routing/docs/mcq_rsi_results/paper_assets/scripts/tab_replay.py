"""tab_replay: matched-budget replay (appendix). For every policy on the development pools: the number of first calls it
fixed per advisor (E/R/V), its replay accuracy, the 95% interval of 2,000 random reassignments of the same per-advisor
budget, p = fraction of random assignments >= the policy, and whether the policy sits above the interval.
Archived rows come from data.replay_v3 (S_1 and the round-3 no-GRPO arms); page-sourced rows (data.replay_notes: the
v3+GRPO MMLU-Pro dynamic policy and the round-5 no-GRPO policies) are added with a dagger. Writes tab/tab_replay.tex
(booktabs) and tab/tab_replay.md. All numbers come from data.json."""
import sys, json
from pathlib import Path
ROOT = Path(str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "")
sys.path.insert(0, str(ROOT))
import style

d = json.load(open(ROOT / "data.json"))
TAB = ROOT / "tab"; TAB.mkdir(parents=True, exist_ok=True)

N_RANDOM = 2000
ROLES = ["extractor", "reasoner", "verifier"]
# display name per archived label / page-sourced key suffix
POLICY_NAME = {
    "S_1":             ("S$_1$",                          "S_1"),
    "dynamic_sft":     ("RSI (dynamic), no GRPO, R3",     "RSI (dynamic), no GRPO, R3"),
    "static_sft":      ("RSI (static), no GRPO, R3",      "RSI (static), no GRPO, R3"),
    "v3g/dynamic":     ("RSI (dynamic), +GRPO, R3",       "RSI (dynamic), +GRPO, R3"),
    "v3r5/dynamic_sft": ("RSI (dynamic), no GRPO, R5",    "RSI (dynamic), no GRPO, R5"),
    "v3r5/static_sft":  ("RSI (static), no GRPO, R5",     "RSI (static), no GRPO, R5"),
}
PAGE_ORDER = ["v3g/dynamic", "v3r5/dynamic_sft", "v3r5/static_sft"]   # order of dagger rows inside a benchmark


def parse_note_key(k):
    """'mmlu_pro v3g/dynamic (dev)' -> ('mmlu_pro', 'v3g/dynamic')"""
    parts = k.split()
    return parts[0], parts[1]


# ---------------------------------------------------------------- collect rows
rows = []   # dicts: bench, policy, n, roles (or None), acc, lo, hi, p, above, dagger
for b in style.BENCH_ORDER:
    for r in d["replay_v3"][b]:
        roles = {k: r["roles"].get(k, 0) for k in ROLES}
        assert set(r["roles"]) <= set(ROLES), (b, r["roles"])
        above = r["replay_acc"] > r["random_hi"]
        assert above == r["above"], (b, r["label"])      # the stored flag must match the numbers
        assert r["n"] == d["meta"]["n_dev"][b], (b, r["label"], r["n"])
        rows.append(dict(bench=b, policy=r["label"], n=r["n"], roles=roles, acc=r["replay_acc"], lo=r["random_lo"],
                         hi=r["random_hi"], p=r["p"], above=above, dagger=False))
    notes = {parse_note_key(k)[1]: v for k, v in d["replay_notes"].items() if parse_note_key(k)[0] == b}
    assert set(notes) <= set(PAGE_ORDER), (b, notes)
    for key in PAGE_ORDER:
        if key not in notes:
            continue
        v = notes[key]
        acc, lo, hi = v.get("replay_acc"), v.get("random_lo"), v.get("random_hi")
        if acc is not None:
            above = acc > hi
            if "above" in v:
                assert above == v["above"], (b, key)
        else:
            above = v["above"]
        rows.append(dict(bench=b, policy=key, n=d["meta"]["n_dev"][b], roles=None, acc=acc, lo=lo, hi=hi, p=v["p"],
                         above=above, dagger=True))


# ---------------------------------------------------------------- formatting
def fmt_p(p, latex):
    if isinstance(p, str):                      # "<0.001" from the results page
        assert p.startswith("<"), p
        return ("$<$" if latex else "<") + p[1:]
    if p < 0.001:
        return "$<$0.001" if latex else "<0.001"
    return f"{p:.3f}"


def fmt_acc(r, latex):
    if r["acc"] is None:
        return "--"
    s = f"{r['acc']:.1f}"
    if r["above"]:
        s = f"\\textbf{{{s}}}" if latex else f"**{s}**"
    return s


def fmt_interval(r):
    if r["lo"] is None:
        return "--"
    return f"[{r['lo']:.1f}, {r['hi']:.1f}]"


def fmt_roles(r):
    if r["roles"] is None:
        return "--"
    return " / ".join(str(r["roles"][k]) for k in ROLES)


def fmt_policy(r, latex):
    name = POLICY_NAME[r["policy"]][0 if latex else 1]
    if r["dagger"]:
        name += "$^{\\dagger}$" if latex else " †"
    return name


def fmt_above(r):
    return "yes" if r["above"] else "no"


nd = d["meta"]["n_dev"]
caption = (
    "Matched-budget replay on the development pools (MedQA {medqa}, MMLU-Pro {mmlu_pro}, GPQA {gpqa}, AQuA-RAT {aqua}). "
    "For each policy, the questions on which it made its first advisor call are rescored with the forced-call result of that "
    "advisor on the same question, and the other questions are scored with the direct answer; this gives the replay accuracy (\\%). "
    "Calls (E/R/V): the number of first calls the policy made to the Extractor, Reasoner and Verifier. "
    "The same number of calls per advisor is then reassigned to uniformly random questions {nr} times; "
    "Random 95\\% is the 2.5--97.5 percentile interval of the resulting accuracies, $p$ the fraction of the {nr} random "
    "assignments whose accuracy is at least the policy's replay accuracy, and Above whether the replay accuracy exceeds the "
    "upper end of the interval. Bold: replay accuracy above the interval. "
    "R3/R5: round 3 or 5 of the chain; no GRPO: the v3 chain trained with SFT only; +GRPO: the main v3+GRPO chain. "
    "$\\dagger$: taken from the results page because the per-question files of these runs were not archived, "
    "so their per-advisor call counts are unavailable (--), and for the AQuA-RAT round-5 policies only $p$ and the above/below outcome were recorded."
).format(nr=f"{N_RANDOM:,}", **nd)

# ---------------------------------------------------------------- LaTeX
tex = ["% Generated by scripts/tab_replay.py from data.json -- do not edit by hand.  Requires \\usepackage{booktabs}.",
       "% tab_replay: matched-budget replay (appendix), development pools.",
       "",
       "\\begin{table*}[t]", "\\centering", "\\small",
       "\\caption{" + caption + "}", "\\label{tab:replay}",
       "\\setlength{\\tabcolsep}{5pt}", "\\renewcommand{\\arraystretch}{1.08}",
       "\\begin{tabular}{l l r c r c r c}", "\\toprule",
       "Benchmark & Policy & $n$ & Calls (E/R/V) & Replay acc. & Random 95\\% & $p$ & Above \\\\",
       "\\midrule"]
prev = None
for r in rows:
    if prev is not None and r["bench"] != prev:
        tex.append("\\addlinespace")
    bench = style.BENCH_LABEL[r["bench"]] if r["bench"] != prev else ""
    tex.append(" & ".join([bench, fmt_policy(r, True), str(r["n"]), fmt_roles(r), fmt_acc(r, True), fmt_interval(r),
                           fmt_p(r["p"], True), fmt_above(r)]) + " \\\\")
    prev = r["bench"]
tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
(TAB / "tab_replay.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------- Markdown twin
md_caption = (caption.replace("\\%", "%").replace("$p$", "p").replace("$\\dagger$", "†").replace("--", "—")
              .replace("2.5—97.5", "2.5–97.5").replace("(—)", "(—)"))
md = ["**Table: Matched-budget replay (appendix).** " + md_caption, ""]
hdr = ["Benchmark", "Policy", "n", "Calls (E/R/V)", "Replay acc.", "Random 95%", "p", "Above"]
md.append("| " + " | ".join(hdr) + " |")
md.append("|:--|:--|--:|:--:|--:|:--:|--:|:--:|")
for r in rows:
    md.append("| " + " | ".join([style.BENCH_LABEL[r["bench"]], fmt_policy(r, False), str(r["n"]), fmt_roles(r),
                                 fmt_acc(r, False), fmt_interval(r), fmt_p(r["p"], False), fmt_above(r)]).replace(" -- ", " — ") + " |")
(TAB / "tab_replay.md").write_text("\n".join(md) + "\n")

print(TAB / "tab_replay.tex", TAB / "tab_replay.md")
for r in rows:
    print(f"{style.BENCH_LABEL[r['bench']]:9s} {fmt_policy(r, False):32s} n={r['n']:<4d} E/R/V={fmt_roles(r):14s} "
          f"acc={('--' if r['acc'] is None else f'{r['acc']:.1f}'):5s} rand={fmt_interval(r):14s} p={fmt_p(r['p'], False):7s} above={fmt_above(r)}")
