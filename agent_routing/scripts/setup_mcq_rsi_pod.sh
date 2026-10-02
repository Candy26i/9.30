#!/usr/bin/env bash
# One-time RunPod setup for MCQ RSI (docs/MCQ_RSI_RUNBOOK.md §2): experiment venv + a separate vLLM venv,
# HF cache and work tree on the persistent network volume (/workspace). Tokens are never stored by this
# script: it only reminds you to log in (HF/W&B keep their own credentials under HF_HOME / ~/.netrc).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WORK="${MCQ_WORK:-/workspace/mcq_rsi}"
MCQ_VENV="${MCQ_VENV:-/workspace/mcq-venv}"
VLLM_VENV="${VLLM_VENV:-/workspace/vllm-venv}"
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"           # the paper-era advisor servers ran vLLM 0.26.0
VLLM_EXTRA_INDEX_URL="${VLLM_EXTRA_INDEX_URL:-}"  # e.g. a CUDA 12.x vLLM wheel index when the driver is < 580
VLLM_TORCH_INDEX_URL="${VLLM_TORCH_INDEX_URL:-}"  # the matching torch index, e.g. https://download.pytorch.org/whl/cu128
BASE_MODEL="Qwen/Qwen3.5-9B"
BASE_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"

log() { printf '[mcq-setup] %s\n' "$*"; }
die() { printf '[mcq-setup] ERROR: %s\n' "$*" >&2; exit 1; }

mkdir -p "$WORK"/{import,advisor_cache,runs,logs,backups,recorded} "$HF_HOME" "$TMPDIR"

# --- volume and GPUs --------------------------------------------------------------------------------
avail_gb="$(df -BG --output=avail /workspace | tail -n 1 | tr -dc '0-9')"
log "/workspace free: ${avail_gb} GB"
(( avail_gb >= 150 )) || log "WARNING: < 150 GB free on /workspace; the full grid needs ~200 GB (runbook §1)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: choose a RunPod CUDA PyTorch image"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)"
ngpu="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( ngpu >= 2 )) || die "need 2 GPUs (advisors on GPU 0, manager training on GPU 1); found ${ngpu}"
nvidia-smi > "$WORK/logs/nvidia-smi.txt"

# --- experiment venv (manager eval / collection / SFT / FA-GRPO) -------------------------------------
python3 -c 'import torch; assert torch.cuda.is_available(), "Choose a RunPod CUDA PyTorch image first"; print("system torch", torch.__version__)'
if [[ ! -x "$MCQ_VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$MCQ_VENV"
fi
"$MCQ_VENV/bin/python" -m pip install -r requirements-math.txt 'pytest>=8,<9'
"$MCQ_VENV/bin/python" -m pip check
"$MCQ_VENV/bin/python" -c 'import torch, transformers, peft; assert torch.cuda.is_available(); print("experiment torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)'
"$MCQ_VENV/bin/python" -m pip freeze > "$WORK/logs/environment_experiment.lock.txt"

# --- vLLM venv (advisor server only; vLLM pins its own torch, so it never shares the experiment venv) ---
# The venv is rebuilt whenever the requested build (version + wheel indexes) differs from the one installed:
# pip would otherwise keep the installed vllm/torch wheels ("Requirement already satisfied").
build="vllm==${VLLM_VERSION} extra=${VLLM_EXTRA_INDEX_URL} torch=${VLLM_TORCH_INDEX_URL}"
if [[ -x "$VLLM_VENV/bin/python" && "$(cat "$VLLM_VENV/.mcq_build" 2>/dev/null)" != "$build" ]]; then
  log "rebuilding ${VLLM_VENV}: installed build '$(cat "$VLLM_VENV/.mcq_build" 2>/dev/null || echo unknown)' != '${build}'"
  rm -rf "$VLLM_VENV"
fi
if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  python3 -m venv "$VLLM_VENV"
fi
"$VLLM_VENV/bin/python" -m pip install --upgrade pip
extra=()
[[ -z "$VLLM_EXTRA_INDEX_URL" ]] || extra+=(--extra-index-url "$VLLM_EXTRA_INDEX_URL")
[[ -z "$VLLM_TORCH_INDEX_URL" ]] || extra+=(--extra-index-url "$VLLM_TORCH_INDEX_URL")
"$VLLM_VENV/bin/python" -m pip install "vllm==${VLLM_VERSION}" "${extra[@]}"
echo "$build" > "$VLLM_VENV/.mcq_build"
# CUDA build vs driver first (no GPU needed), so a too-old driver gets the specific message.
cuda_major="$("$VLLM_VENV/bin/python" -c 'import torch; print((torch.version.cuda or "0").split(".")[0])')"
if (( cuda_major >= 13 && driver < 580 )); then
  die "the vLLM venv's torch is built for CUDA ${cuda_major} but the driver is ${driver} (< 580): rerun with
    VLLM_EXTRA_INDEX_URL=<index of a cu12x vLLM build> VLLM_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu12x
  (the venv is rebuilt automatically), or pick a pod with driver >= 580 (runbook §2)"
fi
"$VLLM_VENV/bin/python" - <<'PY' || die "the vLLM venv cannot use the GPU (driver/CUDA mismatch?); rm -rf ${VLLM_VENV} and see runbook §2"
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch in the vLLM venv sees no GPU"
PY
"$VLLM_VENV/bin/python" -m pip freeze > "$WORK/logs/environment_vllm.lock.txt"

# --- base model at the pinned revision (shared HF cache) ---------------------------------------------
"$MCQ_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
print("base snapshot", snapshot_download("${BASE_MODEL}", revision="${BASE_REVISION}"))
PY

# --- credentials: reminders only ---------------------------------------------------------------------
if ! "$MCQ_VENV/bin/python" -c 'from huggingface_hub import whoami; whoami()' >/dev/null 2>&1; then
  log "HF: not logged in. All MaliDDD artifacts are public, so downloads work anonymously (rate-limited);"
  log "    to upload backups run: HF_HOME=${HF_HOME} ${MCQ_VENV}/bin/hf auth login   (do not paste tokens into scripts)"
fi
if "$MCQ_VENV/bin/python" -c 'import wandb' >/dev/null 2>&1; then
  log "W&B (optional): ${MCQ_VENV}/bin/wandb login, then set \"wandb\": {\"enabled\": true} in the config"
fi

# --- CPU test suite (one process per file, with a watchdog) ------------------------------------------
# MCQ_RSI_TOKENIZER_DIR enables the SFT tokenisation-parity test on the real round-1 label files (design §7.1 item 7);
# it needs the imported S_1 tokenizer (`runpod_mcq_rsi.sh import`). -rs prints every skip.
tok_dir="$WORK/import/medqa/round1/sft"
for f in tests/test_mcq_rsi_*.py; do
  log "pytest ${f}"
  out="$(MARGENT_WANDB_MODE=disabled MCQ_RSI_IMPORT_DIR="$WORK/import" MCQ_RSI_TOKENIZER_DIR="$tok_dir" \
    timeout 400 "$MCQ_VENV/bin/python" -m pytest -q -rs -p no:cacheprovider -o faulthandler_timeout=60 "$f" 2>&1)" \
    || { printf '%s\n' "$out"; die "tests failed: ${f}"; }
  printf '%s\n' "$out" | tail -n 15
  if [[ "$f" == *test_mcq_rsi_sft.py && -f "$tok_dir/tokenizer.json" ]] && grep -qi "skipped.*tokenizer_dir" <<<"$out"; then
    die "the tokenisation-parity test was skipped although ${tok_dir} exists"
  fi
done
[[ -f "$tok_dir/tokenizer.json" ]] || log "NOTE: rerun this script after the import: the tokenisation-parity test needs ${tok_dir}"
log "ready: experiment venv ${MCQ_VENV}, vLLM venv ${VLLM_VENV}, work dir ${WORK}"
