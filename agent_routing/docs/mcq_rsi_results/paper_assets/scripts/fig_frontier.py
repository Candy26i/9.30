"""fig_frontier: cost-accuracy frontier on the locked tests, one panel per benchmark (2x2, 7.2 in wide).
x = total tokens per question (manager + advisor prefill and output), y = locked-test accuracy (%).
All numbers come from data.json; nothing is typed by hand."""
import sys, json
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + ""
sys.path.insert(0, ROOT)
import style
style.setup()
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.lines import Line2D

d = json.load(open(style.ROOT / "data.json"))

# marker shape per policy (fill / edge / hatch come from style.POLICY so every figure agrees)
MARKER = {"base_only": "o", "base_system": "o", "S_1": "s", "dynamic": "D", "static": "^", "success": "h", "single_cot": "v"}
SHORT = {"base_only": "Base only", "base_system": "Base system", "S_1": "S$_1$", "dynamic": "RSI dynamic",
         "static": "RSI static", "success": "RSI success", "single_cot": "Single model,\nreasoning"}
MS = 70


def collect(bench):
    """Return (points, floor, notes). points = list of (policy, acc, total_tokens, token_source)."""
    key = f"{bench}:test"
    sa, tk, lk = d["single_agent"][key], d["tokens"][key], d["locked"][key]
    pts = []
    b = sa["base_nocot_f"]
    pts.append(("base_only", b["acc"], b["total_tok"], "single_agent/base_nocot_f"))
    pts.append(("base_system", d["base_system"][key]["acc"], tk["base/manager"]["total"], "tokens/base/manager"))
    s1_tok = "v3g/S_1" if "v3g/S_1" in tk else "v3/S_1"
    pts.append(("S_1", lk["v3g/S_1"]["acc"], tk[s1_tok]["total"], f"tokens/{s1_tok}"))
    pts.append(("dynamic", lk["v3g/dynamic"]["acc"], tk["v3g/dynamic"]["total"], "tokens/v3g/dynamic"))
    if "v3g/static" in tk:
        pts.append(("static", lk["v3g/static"]["acc"], tk["v3g/static"]["total"], "tokens/v3g/static"))
    suc_tok = next(k for k in ("v3g/success", "abl/success", "main/success") if k in tk)
    pts.append(("success", lk["v3g/success"]["acc"], tk[suc_tok]["total"], f"tokens/{suc_tok}"))
    return pts, tk["draft_only"]["total"]


# per-panel axis ranges and label offsets (points) -- tuned by looking at the rendered PNG
YLIM = {"medqa": (70, 90), "mmlu_pro": (48, 76), "gpqa": (34, 66), "aqua": (36, 88)}
DEFAULT_OFF = (8, 0, "left", "center")
XMAX = {"medqa": 13000, "mmlu_pro": 10500, "gpqa": 11000, "aqua": 6500}
# (dx, dy, ha, va) per policy per panel
OFF = {
    "medqa": {"base_only": (8, -2, "left", "top"), "base_system": (-8, 4, "right", "bottom"), "S_1": (8, 0, "left", "center"),
              "dynamic": (-18, -9, "left", "top"), "static": (-12, 9, "left", "bottom"), "success": (8, 0, "left", "center"),
              "single_cot": (8, 0, "left", "center")},
    "mmlu_pro": {"base_only": (8, 2, "left", "bottom"), "base_system": (-8, 4, "right", "bottom"), "S_1": (8, -4, "left", "top"),
                 "dynamic": (8, -1, "left", "center"), "success": (8, 3, "left", "bottom")},
    "gpqa": {"base_only": (8, -3, "left", "top"), "base_system": (-8, 4, "right", "bottom"), "S_1": (-8, -3, "right", "top"),
             "dynamic": (8, -3, "left", "top"), "static": (8, -4, "left", "top"), "success": (8, 2, "left", "bottom"),
             "single_cot": (8, 3, "left", "bottom")},
    "aqua": {"base_only": (8, 2, "left", "bottom"), "base_system": (0, -9, "center", "top"), "S_1": (-8, -5, "right", "top"),
             "dynamic": (-8, 5, "right", "bottom"), "success": (8, 2, "left", "bottom")},
}
LABEL_BOX = dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.9)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
handles = {}
summary = {}
for i, (bench, ax) in enumerate(zip(style.BENCH_ORDER, axes.flat)):
    pts, floor = collect(bench)
    summary[bench] = {"floor": floor, "points": pts}
    # prompt floor
    ax.axvline(floor, color=style.C["axis"], linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ylo, yhi = YLIM[bench]
    ax.text(floor + XMAX[bench] * 0.015, ylo + (yhi - ylo) * 0.025, "prompt floor", ha="left", va="bottom",
            fontsize=8.5, color=style.C["muted"], zorder=4)
    for pol, acc, tok, _src in pts:
        p = style.POLICY[pol]
        # S_1 sits almost on top of RSI dynamic in some panels: draw it smaller and above, so both stay visible
        size, z = (MS * 0.55, 3.6) if pol == "S_1" else ((MS * 1.25, 3.0) if pol == "dynamic" else (MS, 3.3))
        sc = ax.scatter([tok], [acc], s=size, marker=MARKER[pol], facecolor=p["facecolor"], edgecolor=p["edgecolor"],
                        hatch=p["hatch"], linewidth=0.8 if pol != "S_1" else 1.0, zorder=z)
        handles.setdefault(pol, sc)
        dx, dy, ha, va = OFF[bench].get(pol, DEFAULT_OFF)
        ax.annotate(f"{SHORT[pol]}  {acc:.1f}", (tok, acc), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=9, color=style.C["num"], zorder=4, bbox=LABEL_BOX)
    ax.set_xlim(0, XMAX[bench]); ax.set_ylim(ylo, yhi)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_major_locator(MultipleLocator(4000 if XMAX[bench] > 8000 else 2000))
    ax.yaxis.set_major_locator(MultipleLocator(5 if (yhi - ylo) <= 35 else 10))
    ax.set_title(style.BENCH_LABEL[bench], loc="left", fontsize=12, pad=12)
    style.panel_tag(ax, f"({'abcd'[i]})", x=-0.04, y=1.045)
    if i % 2 == 0:
        ax.set_ylabel("Accuracy (%)")
    if i >= 2:
        ax.set_xlabel("Total tokens per question")

order = ["base_only", "base_system", "S_1", "dynamic", "static", "success"]
hs = [handles[k] for k in order] + [Line2D([0], [0], color=style.C["axis"], linestyle=(0, (4, 3)), linewidth=0.9)]
ls = [style.POLICY[k]["label"] for k in order] + ["Prompt floor (draft only)"]
fig.subplots_adjust(left=0.09, right=0.98, top=0.865, bottom=0.2, hspace=0.42, wspace=0.16)
style.top_legend(fig, hs, ls, ncol=4, y=0.89, handletextpad=0.4, markerscale=1.0, fontsize=10.5)
N = d["meta"]["n_test"]; sa_g = d["single_agent"]
letter_budget = sa_g["medqa:test"]["base_nocot_f"]["max_new_tokens"]
style.caption(fig, [
    f"Locked tests (MedQA {N['medqa']}, MMLU-Pro {N['mmlu_pro']} test, GPQA {N['gpqa']}, AQuA {N['aqua']}); tokens = manager + advisor prefill",
    "and output per question. RSI: v3 + GRPO, round 3. Base only: base model answers with the letter only (" + f"{letter_budget}" + "-token budget).",
    "S$_1$ tokens on MMLU-Pro and AQuA from the v3 S$_1$ run. RSI (success) tokens from the v1 success runs (same labelling",
    "rule) except GPQA, where the RSI run itself was measured. RSI (static) omitted where its tokens were not measured."],
    y=-0.035, x=0.03, fontsize=9.5)
out = style.save(fig, "fig_frontier")
print("\n".join(out))
for bench, s in summary.items():
    print(f"{bench}: floor={s['floor']:.0f}")
    for pol, acc, tok, src in s["points"]:
        print(f"   {pol:12s} acc={acc:5.1f} tok={tok:8.1f}  <- {src}")
