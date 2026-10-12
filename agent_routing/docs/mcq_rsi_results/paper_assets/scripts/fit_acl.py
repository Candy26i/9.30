#!/usr/bin/env python3
"""Make every LaTeX table fit the ACL text block (16 cm = 455.24 pt, two-column, table* = full width).
Compiles each tab/*.tex alone at that width, reads the 'Overfull \\hbox ... at lines A--B' warnings, and wraps only the
tabular environments that overflow in \\resizebox{\\linewidth}{!}{...} (column width in table, text width in table*) (needs \\usepackage{graphicx}, which the ACL
template loads). Idempotent: an already wrapped tabular is left alone. Re-run after regenerating the tables."""
import re, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; TAB = ROOT / "tab"
PRE = r"""\def\DoNotLoadEpstopdf{}% this local TeX lacks grfext; EPS conversion is not needed here
\documentclass[11pt,twocolumn]{article}
\usepackage[a4paper,textwidth=455.24pt,columnsep=0.6cm,textheight=24.7cm]{geometry}
\usepackage{booktabs,amsmath,amssymb,array,graphics}% graphics provides \resizebox (the ACL template's graphicx includes it)
\usepackage{times}\renewcommand{\ttdefault}{cmtt}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{xcolor}\definecolor{rsiblue}{HTML}{2C5A7E}
\begin{document}
"""
def compile_one(tex_path, work):
    doc = work / "probe.tex"
    doc.write_text(PRE + "\\input{" + str(tex_path) + "}\n\\clearpage\n\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "probe.tex"], cwd=work, capture_output=True)
    log = (work / "probe.log").read_text(errors="replace")
    errors = [l for l in log.splitlines() if l.startswith("!")]
    over = [(float(m.group(1)), int(m.group(2)), int(m.group(3))) for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)", log)]
    return errors, over

def tabular_spans(lines):
    spans, start = [], None
    for i, l in enumerate(lines, 1):
        if "\\begin{tabular}" in l and start is None:
            start = i
        if "\\end{tabular}" in l and start is not None:
            spans.append((start, i)); start = None
    return spans

ok = True
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    for f in sorted(TAB.glob("*.tex")):
        errors, over = compile_one(f, work)
        if errors:
            print(f"{f.name}: LaTeX ERROR {errors[:2]}"); ok = False; continue
        lines = f.read_text().splitlines()
        wrap = set()
        for pt, a, b in over:
            for s, e in tabular_spans(lines):
                if s <= b and e >= a and "\\resizebox{" not in lines[s - 2]:
                    wrap.add((s, e, pt))
        for s, e, pt in sorted(wrap, reverse=True):
            lines[e - 1] = lines[e - 1].replace("\\end{tabular}", "\\end{tabular}}", 1)
            lines.insert(s - 1, "\\resizebox{\\linewidth}{!}{%  natural width exceeded the ACL line width by " + f"{pt:.1f}pt")
        if wrap:
            f.write_text("\n".join(lines) + "\n")
            errors, over = compile_one(f, work)
        state = "ERROR" if errors else ("OVERFULL %s" % [o[0] for o in over] if over else "fits")
        ok &= state == "fits"
        print(f"{f.name:28s} {'scaled ' + str(len(wrap)) + ' tabular(s); ' if wrap else ''}{state}")
sys.exit(0 if ok else 1)
