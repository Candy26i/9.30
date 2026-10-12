#!/usr/bin/env python3
"""Build index.html: every figure (PNG) with its caption and every table (Markdown rendered as HTML) in one page."""
import base64, html, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ORDER = ["tab_final_main", "tab_final_labels", "fig_final_value", "fig_final_tokens", "tab_final_reliab", "fig_final_dev", "fig_final_outcomes", "fig_frontier", "fig_single_agent", "fig_grpo_ablation", "fig_rounds", "fig_label_quality", "tab_final_reliab", "tab_main", "fig_grpo_ablation", "tab_grpo_ablation", "fig_label_quality", "tab_label_quality", "fig_rounds", "tab_rounds", "fig_single_agent", "tab_single_agent",
         "fig_frontier", "fig_oracle", "tab_oracle", "fig_selfcheck", "tab_selfcheck", "tab_replay", "tab_setup", "tab_grpo_log"]
def md_to_html(md):
    out, in_tab = [], False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            if not in_tab:
                out.append("<table>"); in_tab = True; tag = "th"
            else:
                tag = "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_tab:
                out.append("</table>"); in_tab = False
            if line.startswith("#"):
                lvl = len(line) - len(line.lstrip("#")); out.append(f"<h{min(lvl+1,4)}>{html.escape(line.lstrip('#').strip())}</h{min(lvl+1,4)}>")
            elif line.strip():
                out.append(f"<p>{html.escape(line)}</p>")
    if in_tab:
        out.append("</table>")
    return "\n".join(out)
parts = []
seen = set()
names = ORDER + sorted({p.stem for p in (ROOT / "fig").glob("*.png") if not p.stem.startswith("_")} | {p.stem for p in (ROOT / "tab").glob("*.md")} - set(ORDER))
for name in names:
    if name in seen: continue
    seen.add(name)
    png = ROOT / "fig" / f"{name}.png"; md = ROOT / "tab" / f"{name}.md"
    if png.is_file():
        b64 = base64.b64encode(png.read_bytes()).decode()
        parts.append(f'<section id="{name}"><h2>{name}</h2><img src="data:image/png;base64,{b64}" alt="{name}"><p class="dl">PDF: fig/{name}.pdf · script: scripts/{name}.py</p></section>')
    tpng = ROOT / "tab" / f"{name}.png"
    if tpng.is_file():
        b64 = base64.b64encode(tpng.read_bytes()).decode()
        parts.append(f'<section id="{name}"><h2>{name}</h2><img src="data:image/png;base64,{b64}" alt="{name}"><p class="dl">LaTeX: tab/{name}.tex (preview rendered at ACL width)</p></section>')
    elif md.is_file():
        parts.append(f'<section id="{name}"><h2>{name}</h2>{md_to_html(md.read_text())}<p class="dl">LaTeX: tab/{name}.tex</p></section>')
toc = "".join(f'<a href="#{n}">{n}</a>' for n in seen if (ROOT / "fig" / f"{n}.png").is_file() or (ROOT / "tab" / f"{n}.md").is_file())
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RSI Paper Assets</title>
<style>:root{{--bg:#fff;--fg:#222;--muted:#666;--line:#e3e8ee;--acc:#2c5a7e}}@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#14181d;--fg:#e6e9ee;--muted:#9aa4b2;--line:#2a323c;--acc:#9db4cc}}}}:root[data-theme=dark]{{--bg:#14181d;--fg:#e6e9ee;--muted:#9aa4b2;--line:#2a323c;--acc:#9db4cc}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font-family:"Times New Roman",Times,serif;line-height:1.45;max-width:1100px;margin:0 auto}}h1{{font-size:1.6rem;margin:0 0 4px}}h2{{font-size:1.15rem;color:var(--acc);border-top:1px solid var(--line);padding-top:16px;margin-top:28px}}h3,h4{{font-size:1rem;margin:14px 0 6px}}
img{{max-width:100%;height:auto;display:block;background:#fff;border:1px solid var(--line)}}table{{border-collapse:collapse;font-size:.86rem;margin:8px 0;max-width:100%;display:block;overflow-x:auto}}th,td{{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{border-bottom:2px solid var(--acc)}}
.dl{{color:var(--muted);font-size:.85rem}}nav{{display:flex;flex-wrap:wrap;gap:6px 14px;margin:10px 0 6px;font-size:.9rem}}nav a{{color:var(--acc);text-decoration:none}}p.sub{{color:var(--muted);margin-top:0}}</style></head>
<body><h1>MARGENT multi-round RSI: paper figures and tables</h1><p class="sub">Final set first. Main text: tab_final_main, tab_final_labels, fig_final_value, fig_final_tokens. Appendix: tab_final_reliab, fig_final_dev, fig_final_outcomes. Everything below that is an earlier draft. Every number comes from data.json and is reproducible with make_all.sh; PDFs in fig/, LaTeX in tab/ (sized for the ACL text block).</p><nav>{toc}</nav>{''.join(parts)}</body></html>"""
(ROOT / "index.html").write_text(page); print("index.html", len(page) // 1024, "KB", len(seen), "assets")
