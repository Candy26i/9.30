"""Shared matplotlib style for the MARGENT-RSI paper figures: serif (Times New Roman), muted blue palette, hatched
reference bars, value labels above bars, light horizontal grid, a gray caption line under the axes, PNG + PDF output.
Mirrors the look of the six reference figures (images/1.jpg .. 6.jpg)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "fig"

# palette (sampled from the reference figures)
C = {  # 2026-10-12: palette sampled from the user's hand-drawn figures (Figure 1 and Figure 4 of the paper)
     "light": "#CFE5F7",    # sky-blue tint (#88C3F1 family): direct answer, always-Verifier, base system
     "mid": "#3B8FC9",      # sky blue: S_1, Verifier, selected MARGENT
     "dark": "#43A44A",     # green (the selected path in Figure 1): RSI dynamic, first call, Reasoner, correction
     "slate": "#967CA9",    # purple (the JSON badge): RSI static
     "dots": "#FCB81A",     # amber (the emoji): RSI success
     "tan": "#FCB81A",      # amber, hatched: force all, second call
     "unsolved": "#ECEEF0", # light gray: unsolved, base only
     "red": "#F5536F",      # coral red (the books): corruption
     "purple": "#967CA9",
     "green": "#43A44A", "blue": "#3B8FC9", "amber": "#FCB81A", "sky": "#88C3F1",
     "edge": "#3F5765",     # dark slate (Figure 1 icons): outlines and hatches
     "grid": "#E6E8EA", "axis": "#8A969E", "text": "#222222", "muted": "#66757F", "num": "#3F5765", "numgray": "#5F6E78"}

# one style per policy, used by every figure so the same thing always looks the same
POLICY = {
    "base_only":   dict(label="Base only",            facecolor=C["unsolved"], edgecolor=C["edge"], hatch=""),
    "base_system": dict(label="Base system",          facecolor=C["light"],    edgecolor=C["edge"], hatch="//"),
    "S_1":         dict(label="S$_1$ MARGENT",        facecolor=C["mid"],      edgecolor=C["edge"], hatch=""),
    "dynamic":     dict(label="MARGENT RSI (dynamic)", facecolor=C["dark"],     edgecolor=C["edge"], hatch=""),
    "static":      dict(label="RSI (static)",         facecolor=C["slate"],    edgecolor=C["edge"], hatch=""),
    "success":     dict(label="RSI (success)",        facecolor=C["dots"],     edgecolor=C["edge"], hatch=".."),
    "single_cot":  dict(label="Single model, reasoning", facecolor=C["tan"],   edgecolor=C["edge"], hatch="//"),
    "verifier":    dict(label="Force Verifier",       facecolor=C["light"],    edgecolor=C["edge"], hatch=""),
    "force_all":   dict(label="Force all",            facecolor=C["tan"],      edgecolor=C["edge"], hatch="//"),
    "oracle":      dict(label="Oracle",               facecolor="white",       edgecolor=C["dark"], hatch="", linestyle="--"),
}
NO_GRPO = dict(hatch="\\\\", alpha=0.85)   # overlay for "without GRPO" variants of dynamic / static

def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": C["axis"], "axes.linewidth": 0.8, "xtick.color": C["text"], "ytick.color": C["text"], "xtick.direction": "out", "ytick.direction": "out",
        "ytick.major.size": 0, "xtick.major.size": 3, "axes.axisbelow": True, "axes.grid": True, "axes.grid.axis": "y", "grid.color": C["grid"], "grid.linewidth": 0.8,
        "hatch.linewidth": 0.6, "hatch.color": C["edge"], "savefig.dpi": 300, "figure.dpi": 110, "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.linewidth": 1.6, "patch.linewidth": 0.8, "text.color": C["text"], "axes.labelcolor": C["text"]})

def horizontal_axes(ax):
    """Variant for horizontal bar charts: vertical grid, left spine only."""
    ax.grid(False); ax.grid(True, axis="x", color=C["grid"], linewidth=0.8); ax.spines["left"].set_visible(True); ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=3)

def label_bars(ax, bars, values, fmt="{:.1f}", color=None, dy=0.012, fontsize=10, inside=False, white=False):
    """Value labels above (or inside) bars; dy is a fraction of the y-range."""
    lo, hi = ax.get_ylim(); off = (hi - lo) * dy
    for b, v in zip(bars, values):
        if v is None:
            continue
        x = b.get_x() + b.get_width() / 2; h = b.get_height()
        if inside:
            ax.text(x, h - off * 2, fmt.format(v), ha="center", va="top", fontsize=fontsize, color="white" if white else (color or C["text"]))
        else:
            ax.text(x, h + off, fmt.format(v), ha="center", va="bottom", fontsize=fontsize, color=color or C["num"])

def top_legend(fig_or_ax, handles=None, labels=None, ncol=4, y=1.0, **kw):
    target = fig_or_ax
    leg = target.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=1.6, handleheight=1.0, columnspacing=1.6, **kw)
    return leg

def policy_patch(key, **over):
    p = dict(POLICY[key]); p.update(over); ls = p.pop("linestyle", "-")
    return Patch(facecolor=p["facecolor"], edgecolor=p["edgecolor"], hatch=p["hatch"], label=p["label"], linestyle=ls)

SHOW_CAPTIONS = False  # 2026-10-11: the user wants no caption text inside the images; captions live in LaTeX


def caption(fig, lines, y=0.01, x=0.12, fontsize=10):
    """Gray caption text under the axes (disabled unless SHOW_CAPTIONS)."""
    if not SHOW_CAPTIONS:
        return
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(reversed(lines)):
        fig.text(x, y + i * 0.035, line, ha="left", va="bottom", fontsize=fontsize, color=C["muted"])

def panel_tag(ax, tag, x=-0.06, y=1.02):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=13, ha="right", va="bottom", color=C["text"])

def save(fig, name, png=True, pdf=True):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    if png:
        fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", facecolor="white"); out.append(str(FIG_DIR / f"{name}.png"))
    if pdf:
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white"); out.append(str(FIG_DIR / f"{name}.pdf"))
    plt.close(fig)
    return out

BENCH_ORDER = ["medqa", "mmlu_pro", "gpqa", "aqua"]
BENCH_LABEL = {"medqa": "MedQA", "mmlu_pro": "MMLU-Pro", "gpqa": "GPQA", "aqua": "AQuA-RAT"}
