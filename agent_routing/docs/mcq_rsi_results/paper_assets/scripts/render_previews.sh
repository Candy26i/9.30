#!/usr/bin/env bash
# PNG previews of the final tables at ACL text width (one-column page so table* lands on page 1; Times; cropped).
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p _preview && cd _preview
for t in tab_final_main tab_final_labels tab_final_reliab tab_app_advisors tab_app_draft tab_app_single tab_app_tokens tab_app_mix tab_app_testpaper tab_setup tab_grpo_log tab_replay; do
  W=455.24pt; grep -q 'begin{table}\[' ../tab/$t.tex && W=219.08pt
  printf '%s\n' '\def\DoNotLoadEpstopdf{}' '\documentclass[11pt]{article}' "\\usepackage[a4paper,textwidth=$W,textheight=24.7cm]{geometry}" \
    '\usepackage{times}\renewcommand{\ttdefault}{cmtt}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}' '\usepackage{booktabs,amsmath,amssymb,array,graphics,xcolor}\definecolor{rsiblue}{HTML}{2C5A7E}' \
    '\pagestyle{empty}\renewcommand{\topfraction}{1}\renewcommand{\textfraction}{0}\renewcommand{\floatpagefraction}{0.95}\setcounter{totalnumber}{5}' '\begin{document}' "\\input{../tab/$t.tex}" '\null\clearpage' '\end{document}' > $t.tex
  pdflatex -interaction=nonstopmode $t.tex > /dev/null 2>&1 || { grep '^!' $t.log; exit 1; }
  rm -f $t.pdf.png; qlmanage -t -s 2600 -o . $t.pdf > /dev/null 2>&1
  python3 -c "
from PIL import Image, ImageChops
im = Image.open('$t.pdf.png').convert('RGB'); b = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox(); p = 40
im.crop((max(b[0]-p, 0), max(b[1]-p, 0), min(b[2]+p, im.size[0]), min(b[3]+p, im.size[1]))).save('../tab/$t.png')"
  echo "preview $t.png"
done
