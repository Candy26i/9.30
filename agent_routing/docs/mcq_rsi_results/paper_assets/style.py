"""Shared matplotlib style for the MARGENT-RSI paper figures: serif (Times New Roman), muted blue palette, hatched
reference bars, value labels above bars, light horizontal grid, a gray caption line under the axes, PNG + PDF output.
Mirrors the look of the six reference figures (images/1.jpg .. 6.jpg)."""
from pathlib import Path
import os as _os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
_PAL = _os.environ.get("MARGENT_PALETTE", "tol")  # 2026-10-13: Paul Tol muted is the default (judged best); others go to fig_<name>/
FIG_DIR = ROOT / ("fig" if _PAL == "tol" else f"fig_{_PAL}")

# palette (sampled from the reference figures)
C = {  # 2026-10-12: one blue-green family with five clearly separated lightness levels (user asked for stronger contrast)
     "light": "#D6E9F8",    # very light blue: direct answer, always-Verifier, base system
     "mid": "#1F5A96",      # deep blue: S_1 MARGENT, Verifier, selected MARGENT
     "dark": "#3A9E4B",     # green: MARGENT RSI (dynamic), first call, Reasoner, correction
     "slate": "#7DB9E8",    # sky blue: MARGENT RSI (static)
     "dots": "#1E7A3C",     # deep green: RSI success
     "tan": "#BFE3B4",      # light green (hatched): force all, second call
     "unsolved": "#ECEEF0", # light gray: unsolved, base only
     "red": "#1F5A96",
     "purple": "#7DB9E8",
     "green": "#3A9E4B", "blue": "#1F5A96", "lgreen": "#BFE3B4", "sky": "#7DB9E8",
     "edge": "#2E4250",     # dark slate: outlines and hatches
     "grid": "#E6E8EA", "axis": "#8A969E", "text": "#222222", "muted": "#66757F", "num": "#2E4250", "numgray": "#55636D"}

# Alternative palette: MARGENT_PALETTE=google renders with the Google Sheets / Charts default colours into fig_google/
PALETTE = _os.environ.get("MARGENT_PALETTE", "tol")
if PALETTE == "google":  # bright Google tones, two hues x two lightness levels (as in the GRPO figure the user liked, 2026-10-13)
    C.update({"light": "#AECBFA", "mid": "#1A73E8", "dark": "#1E8E3E", "slate": "#1A73E8", "dots": "#188038", "tan": "#A8DAB5",
              "unsolved": "#E8EAED", "red": "#1A73E8", "purple": "#AECBFA", "green": "#1E8E3E", "blue": "#1A73E8", "lgreen": "#A8DAB5",
              "sky": "#AECBFA", "edge": "#5F6368", "grid": "#E8EAED", "axis": "#9AA0A6", "text": "#202124", "muted": "#5F6368",
              "num": "#3C4043", "numgray": "#5F6368"})
HATCH_DARK = ({"#A8DAB5": "#1E8E3E", "#AECBFA": "#1A73E8"} if PALETTE == "google" else
              {"#A6D8B5": "#117733", "#88CCEE": "#2F7FA8"} if PALETTE == "tol" else {})

# Candidate academic palettes (2026-10-13, the user found the Google tones too loud). Keys used by the figures:
# unsolved = base only / unsolved, light = base system / direct / always-Verifier, mid = S_1 / Verifier / selected,
# dark = RSI dynamic / first call / Reasoner / correction, slate = RSI static, lgreen = RSI success / Extractor,
# tan = force all / second call, red = corruption, line = tool-call line.
CANDIDATES = {
    "npg":     dict(unsolved="#E3E3E3", light="#91D1C2", mid="#4DBBD5", dark="#00A087", slate="#3C5488", lgreen="#8491B4", tan="#F39B7F", red="#E64B35", line="#E64B35"),
    "tol":     dict(unsolved="#DDDDDD", light="#88CCEE", mid="#44AA99", dark="#117733", slate="#332288", lgreen="#A6D8B5", tan="#A6D8B5", red="#2F7FA8", line="#333333"),
    "gnbu":    dict(unsolved="#E6F2E1", light="#A8DDB5", mid="#7BCCC4", dark="#43A2CA", slate="#7BCCC4", lgreen="#0868AC", tan="#CCEBC5", red="#0868AC", line="#252525"),
    "morandi": dict(unsolved="#E6E1DC", light="#C9D6DF", mid="#7895B2", dark="#5E7D7E", slate="#A7B4C2", lgreen="#D4B499", tan="#E2D4C3", red="#B5838D", line="#4A4A4A"),
    "deep":    dict(unsolved="#D9D9D9", light="#A1C9F4", mid="#4C72B0", dark="#55A868", slate="#64B5CD", lgreen="#8172B3", tan="#CCB974", red="#C44E52", line="#C44E52"),
    "lancet":  dict(unsolved="#E5E5E5", light="#ADB6B6", mid="#00468B", dark="#42B540", slate="#0099B4", lgreen="#925E9F", tan="#FDAF91", red="#ED0000", line="#AD002A"),
}
FLAT = PALETTE in CANDIDATES
if FLAT:
    _c = CANDIDATES[PALETTE]
    C.update(_c); C.update({"green": _c["dark"], "blue": _c["mid"], "sky": _c["light"], "dots": _c["lgreen"], "purple": _c["slate"],
              "edge": "#6B6B6B" if PALETTE == "tol" else "#4D4D4D", "grid": "#E8E8E8", "axis": "#9A9A9A", "text": "#222222", "muted": "#666666", "num": "#333333", "numgray": "#555555"})
C.setdefault("line", C["num"])


# one style per policy, used by every figure so the same thing always looks the same
POLICY = {
    "base_only":   dict(label="Base only",            facecolor=C["unsolved"], edgecolor=C["edge"], hatch=""),
    "base_system": dict(label="Base system",          facecolor=C["light"],    edgecolor=C["edge"], hatch="//"),
    "S_1":         dict(label="MARGENT S$_1$",        facecolor=C["mid"],      edgecolor=C["edge"], hatch=""),
    "dynamic":     dict(label="MARGENT RSI (dynamic)", facecolor=C["dark"],     edgecolor=C["edge"], hatch=""),
    "static":      dict(label="MARGENT RSI (static)", facecolor=C["slate"],    edgecolor=C["edge"], hatch=""),
    "success":     dict(label="MARGENT RSI (success)", facecolor=C["lgreen"],   edgecolor=C["edge"], hatch=".."),
    "single_cot":  dict(label="Single model, reasoning", facecolor=C["tan"],   edgecolor=C["edge"], hatch="//"),
    "verifier":    dict(label="Force Verifier",       facecolor=C["light"],    edgecolor=C["edge"], hatch=""),
    "force_all":   dict(label="Force all",            facecolor=C["tan"],      edgecolor=C["edge"], hatch="//"),
    "oracle":      dict(label="Oracle",               facecolor="white",       edgecolor=C["dark"], hatch="", linestyle="--"),
}
if FLAT:
    POLICY["base_system"]["hatch"] = ""; POLICY["success"]["hatch"] = "//" if PALETTE == "tol" else ""
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

def ink_on(face):
    """Text colour for a label drawn on top of a filled bar: white on dark fills, the number colour on light ones."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(face)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.42 else C["num"]


def _tint_hatches(fig):
    """Hatch lines on a light bar take the dark shade of the same hue (matplotlib 3.10 ties hatch colour to the edge)."""
    if not HATCH_DARK:
        return
    from matplotlib.patches import Patch as _P
    from matplotlib.colors import to_hex, to_rgba
    for p in fig.findobj(_P):
        if p.get_hatch() and p.get_fill():
            dark = HATCH_DARK.get(to_hex(p.get_facecolor()[:3]).upper())
            if dark:
                p._hatch_color = to_rgba(dark)


def save(fig, name, png=True, pdf=True):
    _tint_hatches(fig)
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
