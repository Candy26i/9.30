"""fig_rounds: no-GRPO chain (v3), rounds 1-5. Top row = dev accuracy per round, bottom row = calls per question per
round; dynamic (navy, circles) vs static (slate, squares); hollow markers = locked-test accuracy at rounds 1 (S_1), 3, 5."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "")
import style
style.setup()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from matplotlib.text import Text

d = json.load(open(style.ROOT / "data.json"))
TRAJ = d["dev_trajectory"]; LOCKED = d["locked"]
ARMS = {"dynamic": dict(color=style.C["dark"], marker="o", ms=4.6, label="dynamic (dev)", num=style.C["num"]),
        "static":  dict(color=style.C["slate"], marker="s", ms=4.4, label="static (dev)", num=style.C["numgray"])}
LOCKED_KEY = {1: {"dynamic": "v3/S_1", "static": "v3/S_1"},
              3: {"dynamic": "v3/dynamic_sft", "static": "v3/static_sft"},
              5: {"dynamic": "v3r5/dynamic_sft", "static": "v3r5/static_sft"}}
LABEL_FS = 7.5

# smaller type for a 2x4 grid
plt.rcParams.update({"xtick.labelsize": 9, "ytick.labelsize": 9, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 10})

fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.9), sharex=True)
fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.14, hspace=0.32, wspace=0.42)

summary = {}
pending_labels = []   # (ax, x, y, text, color) -> placed after the layout is known
obstacles = {}        # ax -> list of data-space points that labels must not cover

for j, bench in enumerate(style.BENCH_ORDER):
    ax_acc, ax_calls = axes[0, j], axes[1, j]
    obstacles[ax_acc] = []
    traj = TRAJ[bench]["v3"]
    rounds = sorted(int(r) for r in traj["dynamic"])
    locked_vals = {}  # round -> {arm: acc}
    for r in (1, 3, 5):
        if r not in rounds:
            continue
        locked_vals[r] = {}
        for arm in ARMS:
            rec = LOCKED[f"{bench}:test"].get(LOCKED_KEY[r][arm])
            if rec is not None:
                locked_vals[r][arm] = rec["acc"]
    # dev lines
    for arm, st in ARMS.items():
        acc = [traj[arm][str(r)]["acc"] for r in rounds]
        calls = [traj[arm][str(r)]["calls"] for r in rounds]
        ax_acc.plot(rounds, acc, color=st["color"], marker=st["marker"], ms=st["ms"], lw=1.5, zorder=3, solid_capstyle="round")
        ax_calls.plot(rounds, calls, color=st["color"], marker=st["marker"], ms=st["ms"], lw=1.5, zorder=3, solid_capstyle="round")
        for (x0, y0), (x1, y1) in zip(zip(rounds[:-1], acc[:-1]), zip(rounds[1:], acc[1:])):   # sample the polyline
            obstacles[ax_acc] += [(x0 + t * (x1 - x0), y0 + t * (y1 - y0)) for t in np.linspace(0, 1, 25)]
        summary.setdefault(bench, {})[arm] = {"dev_acc": acc, "dev_calls": calls}
    # locked-test overlay (top row only)
    for r, vals in locked_vals.items():
        if r == 1:  # S_1 is shared by both arms: one neutral hollow marker
            v = vals["dynamic"]
            ax_acc.plot([r], [v], marker="o", ms=6.8, mfc="white", mec=style.C["edge"], mew=1.3, ls="none", zorder=5)
            obstacles[ax_acc].append((r, v)); pending_labels.append((ax_acc, r, v, f"{v:.1f}", style.C["numgray"]))
            summary[bench]["locked_S1"] = v
            continue
        order = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
        same = len(order) == 2 and abs(order[0][1] - order[1][1]) < 0.05
        for i, (arm, v) in enumerate(order):
            st = ARMS[arm]
            dx = -0.14 if arm == "dynamic" else 0.14
            ax_acc.plot([r + dx], [v], marker=st["marker"], ms=st["ms"] + (2.4 if arm == "static" else 2.2), mfc="white",
                        mec=st["color"], mew=1.3, ls="none", zorder=5 if arm == "static" else 6)
            obstacles[ax_acc].append((r + dx, v))
            summary[bench].setdefault(f"locked_r{r}", {})[arm] = v
            if same and i == 1:
                continue
            pending_labels.append((ax_acc, r, v, f"{v:.1f}", style.C["num"] if same else st["num"]))
    # axes ranges: data span plus room for the labels
    all_acc = [traj[a][str(r)]["acc"] for a in ARMS for r in rounds] + [v for vals in locked_vals.values() for v in vals.values()]
    lo, hi = min(all_acc), max(all_acc); m = max(1.5, 0.3 * (hi - lo))
    ax_acc.set_ylim(lo - m, hi + m)
    all_calls = [traj[a][str(r)]["calls"] for a in ARMS for r in rounds]
    lo, hi = min(all_calls), max(all_calls); m = max(0.04, 0.22 * (hi - lo))
    ax_calls.set_ylim(max(0, lo - m), hi + m)
    for ax in (ax_acc, ax_calls):
        ax.set_xlim(0.55, 5.45); ax.set_xticks([1, 2, 3, 4, 5])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(axis="x", length=3, pad=2); ax.tick_params(axis="y", length=0, pad=2)
    ax_acc.set_title(style.BENCH_LABEL[bench], pad=6)
    ax_calls.set_xlabel("Round", labelpad=2)
    if bench == "gpqa":
        for ax in (ax_acc, ax_calls):
            pass  # GPQA ran three rounds; said in the caption

axes[0, 0].set_ylabel("Accuracy (%)")
axes[1, 0].set_ylabel("Calls / question")
style.panel_tag(axes[0, 0], "(a)", x=-0.42, y=1.04)
style.panel_tag(axes[1, 0], "(b)", x=-0.42, y=1.04)

handles = [Line2D([], [], color=ARMS["dynamic"]["color"], marker="o", ms=4.6, lw=1.5),
           Line2D([], [], color=ARMS["static"]["color"], marker="s", ms=4.4, lw=1.5),
           Line2D([], [], ls="none", marker="o", ms=6.8, mfc="white", mec=ARMS["dynamic"]["color"], mew=1.3),
           Line2D([], [], ls="none", marker="s", ms=6.8, mfc="white", mec=ARMS["static"]["color"], mew=1.3),
           Line2D([], [], ls="none", marker="o", ms=6.8, mfc="white", mec=style.C["edge"], mew=1.3)]
style.top_legend(fig, handles, ["dynamic (dev)", "static (dev)", "dynamic (test)", "static (test)", "MARGENT S$_1$ (test)"], ncol=5, y=0.93, handletextpad=0.4)

style.caption(fig, ["No-GRPO chain (v3): dev accuracy and calls per question per round; round 1 = S$_1$. GPQA: 3 rounds.",
                    "Hollow markers = locked test (MedQA 500, MMLU-Pro 500 test, GPQA 100, AQuA 254).",
                    "Dev pools: MedQA 200, MMLU-Pro 200, GPQA 100, AQuA 254."], y=-0.085, x=0.085, fontsize=9.5)

# ---- value labels for the locked markers: first offset (in points) that overlaps neither lines, markers nor other labels
fig.canvas.draw(); renderer = fig.canvas.get_renderer()
CANDS = [(0, 7, "center", "bottom"), (0, -7, "center", "top"), (6, 5, "left", "bottom"), (-6, 5, "right", "bottom"),
         (6, -5, "left", "top"), (-6, -5, "right", "top"), (9, 0, "left", "center"), (-9, 0, "right", "center"),
         (0, 13, "center", "bottom"), (0, -13, "center", "top")]
LEADER = [(12, 14, "left", "bottom"), (-12, 14, "right", "bottom"), (12, -14, "left", "top"), (-12, -14, "right", "top"),
          (0, 22, "center", "bottom"), (0, -22, "center", "top")]   # farther away, joined to the marker by a thin line
LEADER_KW = dict(arrowstyle="-", lw=0.6, color=style.C["axis"], shrinkA=1, shrinkB=5)
RAD = 6.0  # px clearance around every obstacle point (marker radius + line width at figure dpi)
placed = {ax: [] for ax in obstacles}
obst_px = {ax: ax.transData.transform(np.array(pts)) for ax, pts in obstacles.items()}
for ax, x, y, text, color in pending_labels:
    axbb = ax.get_window_extent(renderer); axbb = Bbox.from_extents(axbb.x0 - 5, axbb.y0 - 2, axbb.x1 + 5, axbb.y1 + 2)
    chosen = None
    for dx, dy, ha, va in CANDS + LEADER:
        leader = (dx, dy, ha, va) in LEADER
        t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=LABEL_FS, color=color, zorder=8,
                        arrowprops=LEADER_KW if leader else None)
        t.update_positions(renderer); bb = Text.get_window_extent(t, renderer)   # text box only (not the leader line)
        test = Bbox.from_extents(bb.x0 - RAD, bb.y0 - RAD, bb.x1 + RAD, bb.y1 + RAD)
        pts = obst_px[ax]
        hit = np.any((pts[:, 0] > test.x0) & (pts[:, 0] < test.x1) & (pts[:, 1] > test.y0) & (pts[:, 1] < test.y1))
        inside = axbb.contains(bb.x0, bb.y0) and axbb.contains(bb.x1, bb.y1)
        clash = any(bb.overlaps(Bbox.from_extents(p.x0 - 2, p.y0 - 2, p.x1 + 2, p.y1 + 2)) for p in placed[ax])
        if not hit and inside and not clash:
            chosen = (dx, dy); placed[ax].append(bb); break
        t.remove()
    if chosen is None:  # fall back to "above" and report it
        t = ax.annotate(text, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=LABEL_FS, color=color, zorder=8)
        placed[ax].append(t.get_window_extent(renderer)); print("WARN: no clean spot for label", ax.get_title(), text)

print(style.save(fig, "fig_rounds"))
print(json.dumps(summary, indent=1))
