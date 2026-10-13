"""fig_grpo_ablation: locked-test accuracy (a) and mean calls per question (b) for the dynamic and static RSI arms,
with and without FA-GRPO in each round. Hatched bars = SFT only (v3 chain, keys v3/*_sft); solid = SFT + FA-GRPO
(v3+GRPO chain, keys v3g/*). All numbers come from data.json (locked, pool "test")."""
import sys, json
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + ""
sys.path.insert(0, ROOT)
import style; style.setup()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

d = json.load(open(style.ROOT / "data.json"))
B = style.BENCH_ORDER
POOL = "test"

# (policy style key, data key, with GRPO?, legend label)
SERIES = [
    ("dynamic", "v3/dynamic_sft", False, "MARGENT RSI (dynamic), SFT only"),
    ("dynamic", "v3g/dynamic",    True,  "MARGENT RSI (dynamic), SFT + FA-GRPO"),
    ("static",  "v3/static_sft",  False, "MARGENT RSI (static), SFT only"),
    ("static",  "v3g/static",     True,  "MARGENT RSI (static), SFT + FA-GRPO"),
]
W = 0.2                     # bar width (bars within a group touch, as in the reference figures)
HATCH = style.NO_GRPO["hatch"]
ALPHA = style.NO_GRPO["alpha"]
# hatch colour for the "without GRPO" overlay: the shared edge colour is invisible on the navy bar, so use white there
HATCH_COLOR = {"dynamic": "white", "static": style.C["edge"]}


def face(pol, grpo):
    fc = style.POLICY[pol]["facecolor"]
    return fc if grpo else to_rgba(fc, ALPHA)


def draw_series(ax, xs, vals, pol, grpo):
    edge = style.POLICY[pol]["edgecolor"]
    bars = ax.bar(xs, vals, W, facecolor=face(pol, grpo), edgecolor=edge, linewidth=0.8, zorder=2)
    if not grpo:   # NO_GRPO overlay: hatch in a visible colour, then redraw the outline on top
        ax.bar(xs, vals, W, facecolor="none", edgecolor=HATCH_COLOR[pol], hatch=HATCH, linewidth=0, zorder=3)
        ax.bar(xs, vals, W, facecolor="none", edgecolor=edge, linewidth=0.8, zorder=4)
    return bars


def legend_handle(pol, grpo):
    edge = style.POLICY[pol]["edgecolor"]
    solid = Patch(facecolor=face(pol, grpo), edgecolor=edge, linewidth=0.8)
    if grpo:
        return solid
    hatch = Patch(facecolor="none", edgecolor=HATCH_COLOR[pol], hatch=HATCH, linewidth=0)
    outline = Patch(facecolor="none", edgecolor=edge, linewidth=0.8)
    return (solid, hatch, outline)


acc = {k: [d["locked"][f"{b}:{POOL}"][k]["acc"] for b in B] for _, k, _, _ in SERIES}
calls = {k: [d["locked"][f"{b}:{POOL}"][k]["calls"] for b in B] for _, k, _, _ in SERIES}
for b in B:
    for _, k, _, _ in SERIES:
        assert d["locked"][f"{b}:{POOL}"][k]["n"] == d["meta"]["n_test"][b], (b, k)

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[2.4, 1.25]))
fig.subplots_adjust(left=0.09, right=0.99, top=0.83, bottom=0.185, hspace=0.32)
x = np.arange(len(B))

for i, (pol, key, grpo, _) in enumerate(SERIES):
    xs = x + (i - 1.5) * W
    bars = draw_series(ax_a, xs, acc[key], pol, grpo)
    style.label_bars(ax_a, bars, acc[key], fmt="{:.1f}", fontsize=9)
    bars = draw_series(ax_b, xs, calls[key], pol, grpo)
    style.label_bars(ax_b, bars, calls[key], fmt="{:.2f}", fontsize=9)

ax_a.set_ylim(0, 100); ax_a.set_yticks([0, 25, 50, 75, 100]); ax_a.set_ylabel("Accuracy (%)")
ax_a.tick_params(axis="x", length=0)
ax_b.set_ylim(0, 1.0); ax_b.set_yticks([0, 0.5, 1.0]); ax_b.set_ylabel("Mean calls")
ax_b.set_xticks(x, [style.BENCH_LABEL[b] for b in B]); ax_b.tick_params(axis="x", labelsize=12)
ax_a.set_xlim(-0.55, len(B) - 0.45)
style.panel_tag(ax_a, "(a)"); style.panel_tag(ax_b, "(b)")

handles = [legend_handle(pol, grpo) for pol, _, grpo, _ in SERIES]
labels = [lab for _, _, _, lab in SERIES]
style.top_legend(fig, handles, labels, ncol=2, y=0.875, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})

n = d["meta"]["n_test"]
style.caption(fig, [f"Locked tests (MedQA {n['medqa']}, MMLU-Pro {n['mmlu_pro']} test, GPQA {n['gpqa']}, AQuA {n['aqua']}).",
                    "Hatched = SFT only; solid = SFT + FA-GRPO each round.",
                    "Round-3 managers of the no-GRPO (v3) and v3+GRPO chains."], y=0.005)
print(style.save(fig, "fig_grpo_ablation"))

# compact dump for the verifier
for b in B:
    row = []
    for _, k, _, _ in SERIES:
        e = d["locked"][f"{b}:{POOL}"][k]
        row.append(f"{k}: acc {e['acc']:.1f} calls {e['calls']:.2f}")
    print(style.BENCH_LABEL[b], "|", " | ".join(row))
