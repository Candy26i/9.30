#!/usr/bin/env bash
# Rebuild every paper figure and table. `--data` first re-reads the raw result files into data.json (needs the local
# result folders named in build_data.py; recomputes all paired bootstraps, a few minutes). Then: figures + tables,
# ACL-width fitting of the LaTeX tables, and the gallery page index.html.
set -euo pipefail
cd "$(dirname "$0")"
[ "${1:-}" = "--data" ] && python3 build_data.py
for s in tab_main fig_grpo_ablation tab_grpo_ablation fig_label_quality fig_rounds tab_rounds fig_single_agent fig_frontier fig_oracle fig_selfcheck tab_replay tab_setup tab_grpo_log tab_final; do
  printf '%-20s' "$s"; python3 "scripts/$s.py" > /dev/null && echo ok
done
python3 scripts/fit_acl.py
bash scripts/render_previews.sh
python3 build_gallery.py
