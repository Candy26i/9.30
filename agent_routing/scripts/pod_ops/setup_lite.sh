#!/usr/bin/env bash
# One-GPU setup for inference-only baselines: the experiment venv (+ the Qwen3.5 linear-attention kernels, no causal-conv1d
# build, no vLLM venv, no CPU test pass) and the pinned base model snapshot. scripts/setup_mcq_rsi_pod.sh needs 2 GPUs.
set -euo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MCQ_VENV=/workspace/mcq-venv
python3 -c 'import torch; assert torch.cuda.is_available(); print("system torch", torch.__version__)'
[ -x $MCQ_VENV/bin/python ] || python3 -m venv --system-site-packages $MCQ_VENV
$MCQ_VENV/bin/python -m pip install -q -r requirements-math.txt
$MCQ_VENV/bin/python -m pip install -q --no-deps "fla-core==0.4.1" "flash-linear-attention==0.4.1" "einops==0.8.2" "ninja==1.13.2"
$MCQ_VENV/bin/python -c 'import torch, transformers, peft; assert torch.cuda.is_available(); print("experiment torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)'
$MCQ_VENV/bin/python -c 'from huggingface_hub import snapshot_download; print("base snapshot", snapshot_download("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a"))'
