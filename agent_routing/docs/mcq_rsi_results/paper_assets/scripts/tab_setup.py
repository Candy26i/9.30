"""tab_setup: two-column appendix table (component / setting) generated from data.setup.

Writes tab/tab_setup.tex (booktabs) and tab/tab_setup.md. Every number comes from data.json except the GPQA
max_seq_len override (GPQA_MAX_SEQ_LEN below), which the asset brief specifies and the project config
agent_routing/configs/mcq_rsi_gpqa_v3.json (sft.max_seq_len = 16384) corroborates; data.setup only carries the
default 4096.
"""
import json
import sys
from decimal import Decimal

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "")
import style  # noqa: E402  (BENCH_ORDER / BENCH_LABEL; no plotting here)

ROOT = style.ROOT
TAB_DIR = ROOT / "tab"
S = json.load(open(ROOT / "data.json"))["setup"]

GPQA_MAX_SEQ_LEN = 16384  # asset brief + agent_routing/configs/mcq_rsi_gpqa_v3.json; not in data.json

# ----------------------------------------------------------------------------------------------------------- helpers
# free-text strings in data.setup contain tokens that must become math / verbatim in LaTeX; protect them, escape the rest
MATH = [
    ("enable_thinking=False", r"\texttt{enable\_thinking=False}", "enable_thinking=False"),
    ("G_{k-1}", r"G$_{k-1}$", "G_{k-1}"),
    ("S_1", r"S$_1$", "S_1"),
    ("S_k", r"S$_k$", "S_k"),
    ("G_k", r"G$_k$", "G_k"),
    ("J in (0.02, 0.98)", r"$J \in (0.02, 0.98)$", "J ∈ (0.02, 0.98)"),
    ("KL_dec", r"KL$_{\mathrm{dec}}$", "KL_dec"),
    (">=", r"$\geq$", "≥"),
    ("<=", r"$\leq$", "≤"),
    (":=", r"$:=$", ":="),
    (" x ", r" $\times$ ", " × "),
    (" - ", r" $-$ ", " − "),
]


def tex(s):
    s = str(s)
    for i, (raw, _, _) in enumerate(MATH):
        s = s.replace(raw, f"\x00{i}\x00")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    for i, (_, latex, _) in enumerate(MATH):
        s = s.replace(f"\x00{i}\x00", latex)
    return s


def md(s):
    s = str(s)
    for raw, _, plain in MATH:
        s = s.replace(raw, plain)
    return s.replace("|", "\\|")


def sci(v):
    """1e-05 -> ('$1\\times10^{-5}$', '1e-5')."""
    m, e = f"{v:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$", f"{m}e{int(e)}"


def num(v):
    """Plain number as stored (no spurious trailing zeros)."""
    d = Decimal(str(v))
    return f"{d.normalize():f}" if d != d.to_integral() else f"{int(d)}"


def per_bench(dct, fmt=num):
    return ", ".join(f"{style.BENCH_LABEL[b]} {fmt(dct[b])}" for b in style.BENCH_ORDER)


# ------------------------------------------------------------------------------------------------------------- rows
# each row: (component, setting) with the setting written once in a neutral form; tex()/md() handle the markup.
c, sft, g, gates, pools, arms, ver = (S["collect"], S["sft"], S["grpo"], S["gates"], S["pools"], S["arms"], S["versions"])
lr_sft_tex, lr_sft_md = sci(sft["learning_rate"])
lr_g_tex, lr_g_md = sci(g["learning_rate"])

groups = [
    ("Models", [
        ("Base model", S["base_model"]),
        ("Advisors", S["advisors"]),
        ("Manager", S["manager"]),
    ]),
    ("Rounds and data collection", [
        ("Rounds", f"{S['rounds']} (each: collect, SFT, FA-GRPO); rounds {S['extra_rounds']}"),
        ("Questions collected per round", f"{c['questions_per_round']} (GPQA {c['gpqa']})"),
        ("Max depth", f"{c['max_depth']}"),
        ("Commit:rescue balancing RHO", per_bench(S["rho_commit_rescue"], lambda v: f"{v:.1f}")),
    ]),
    ("SFT (each round)", [
        ("Epochs", f"{sft['num_train_epochs']}"),
        ("Learning rate", ("LR_SFT",)),
        ("Batch", f"{sft['per_device_batch_size']} per device x {sft['gradient_accumulation_steps']} gradient-accumulation steps "
                  f"(effective {sft['per_device_batch_size'] * sft['gradient_accumulation_steps']})"),
        ("Max sequence length", f"{sft['max_seq_len']:,} tokens (GPQA {GPQA_MAX_SEQ_LEN:,})"),
        ("Seed / context / draft supervision", f"{sft['seed']} / {sft['context']} (tool schemas rendered as in the S_1 prompt) / "
                                               f"{sft['draft_supervision']} (route-only loss)"),
    ]),
    ("FA-GRPO (each round)", [
        ("Steps", f"{g['steps']} steps x {g['questions_per_step']} questions per step "
                  f"({S['grpo_questions_per_round']} questions per round; GPQA pool {S['grpo_gpqa_pool']})"),
        ("Learning rate / max grad norm / seed", ("LR_GRPO", f" / {g['max_grad_norm']:.1f} / {g['seed']}")),
        ("BETA (root / dec / rev)", f"{num(g['beta_root'])} / {num(g['beta_dec'])} / {num(g['beta_rev'])}"),
        ("Anchor LAMBDA", f"{num(g['anchor_lambda'])} ({g['anchor_rows_per_step']} anchor rows per step); estimator: {g['estimator']}"),
        ("Guard", f"every {g['guard_every']} steps on a {g['guard_probe_size']}-question probe: call-rate delta <= {g['guard_max_call_rate_delta']:.2f} "
                  f"and KL_dec <= {g['guard_max_kl_dec']:.2f}, otherwise roll back to the last good step"),
        ("Informative fraction", f">= {num(g['informative_min_fraction'])} of revision states with J in (0.02, 0.98)"),
    ]),
    ("Acceptance and gates", [
        ("GRPO acceptance", S["grpo_acceptance"]),
        ("Gates", f"accuracy tolerance {num(gates['acc_tolerance'])}; extra calls {num(gates['extra_calls'])}; gap fraction {num(gates['gap_fraction'])}; "
                  f"SFT max calls {num(gates['sft_max_calls'])}; SFT max accuracy drop {num(gates['sft_max_acc_drop'])}"),
    ]),
    ("Pools", [
        ("Development", per_bench(pools["dev"])),
        ("Locked test", ", ".join(f"{style.BENCH_LABEL[b]} {pools['locked_test'][b]}" + (" (pool test)" if b == "mmlu_pro" else "")
                                  for b in style.BENCH_ORDER)),
    ]),
    ("Arms", [
        ("dynamic", arms["dynamic"]),
        ("static", arms["static"]),
        ("success", arms["success"]),
    ]),
    ("Versions", [
        ("v1", ver["v1"]),
        ("v3", ver["v3"]),
        ("v3+GRPO (main)", ver["v3+GRPO (main)"]),
    ]),
]

# symbols that differ between LaTeX and Markdown
SYM = {"RHO": (r"$\rho$", "ρ"), "BETA": (r"$\beta$", "β"), "LAMBDA": (r"$\lambda$", "λ"), "LR_SFT": (lr_sft_tex, lr_sft_md), "LR_GRPO": (lr_g_tex, lr_g_md)}


def render(component, setting, fn, which):
    comp = fn(component)
    for k, (t, m) in SYM.items():
        comp = comp.replace(k, t if which == "tex" else m)
    if isinstance(setting, tuple):  # (symbol key, optional plain tail)
        head = SYM[setting[0]][0 if which == "tex" else 1]
        tail = fn(setting[1]) if len(setting) > 1 else ""
        return comp, head + tail
    return comp, fn(setting)


# ------------------------------------------------------------------------------------------------------------ LaTeX
cap = (r"Experimental setup. S$_1$ is the paper's one-round SFT manager; S$_k$ and G$_k$ are the round-$k$ manager before and "
       r"after FA-GRPO. The GPQA sequence-length override accommodates its longer depth-2 label rows. "
       r"The MMLU-Pro \emph{test\_paper} pool overlaps training pools and is used only in the appendix.")
L = [r"% generated by scripts/tab_setup.py from data.json -- do not edit by hand",
     r"% requires \usepackage{booktabs,array}",
     r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
     rf"\caption{{{cap}}}", r"\label{tab:setup}",
     r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.72\linewidth}@{}}",
     r"\toprule", r"Component & Setting \\", r"\midrule"]
for gi, (gname, rows) in enumerate(groups):
    if gi:
        L.append(r"\addlinespace")
    L.append(rf"\multicolumn{{2}}{{@{{}}l}}{{\textit{{{tex(gname)}}}}} \\")
    for comp, setting in rows:
        a, b = render(comp, setting, tex, "tex")
        L.append(f"{a} & {b.replace('4-5', '4--5')} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
TAB_DIR.mkdir(parents=True, exist_ok=True)
(TAB_DIR / "tab_setup.tex").write_text("\n".join(L) + "\n")

# --------------------------------------------------------------------------------------------------------- Markdown
cap_md = ("Experimental setup. S_1 is the paper's one-round SFT manager; S_k and G_k are the round-k manager before and after "
          "FA-GRPO. The GPQA sequence-length override accommodates its longer depth-2 label rows. The MMLU-Pro *test_paper* pool "
          "overlaps training pools and is used only in the appendix.")
M = [f"**Table: {cap_md}**", "", "| Component | Setting |", "|:--|:--|"]
for gname, rows in groups:
    M.append(f"| **{md(gname)}** | |")
    for comp, setting in rows:
        a, b = render(comp, setting, md, "md")
        M.append(f"| {a} | {b.replace('4-5', '4–5')} |")
(TAB_DIR / "tab_setup.md").write_text("\n".join(M) + "\n")

print(TAB_DIR / "tab_setup.tex")
print(TAB_DIR / "tab_setup.md")
