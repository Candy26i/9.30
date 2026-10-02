#!/usr/bin/env bash
# Serve all 12 MCQ advisor LoRAs (<bench>_<kind>) from one vLLM server (design §6: GPU 0, ~0.45 of its memory).
#
#   bash scripts/start_mcq_advisors.sh [start|stop|status|evidence]
#
# vLLM runs from its own venv (VLLM_VENV, default /workspace/vllm-venv; scripts/setup_mcq_rsi_pod.sh),
# never the experiment venv: vLLM pins its own torch.
#
# Shared with the paper-era servers (logs vllm_medqa_8003.log, vllm_mmlu_8002.log, advisor_server_gpqa.log):
# vLLM 0.26.0, "Resolved architecture: Qwen3_5ForConditionalGeneration" (no architecture override), bf16,
# max_model_len 16384, greedy requests with thinking disabled per request (chat_template_kwargs, which the
# advisor client always sends). Every difference from the paper servers (runbook §6 lists them too):
#   * hardware: the paper ran one ~96 GB SM 12.x (Blackwell) GPU per server ("Free memory on device 94.97 GiB",
#     "SM 12.x"); this pod has A100/H100-80GB (SM80/SM90), so bf16 kernels and numerics differ;
#   * one server for all 12 LoRAs (paper: one server, 3 LoRAs, per benchmark): --max-loras 12 and
#     --max-cpu-loras 12 (paper: vLLM defaults, max_loras 1), --gpu-memory-utilization 0.45 (paper 0.85, GPQA 0.92);
#   * --max-lora-rank 16 (MedQA/MMLU-Pro paper default 16; the GPQA server used 64; all adapters are r16);
#     the GPQA server also had trust_remote_code and override_generation_config enable_thinking=False;
#   * --served-model-name Qwen/Qwen3.5-9B and the pinned --revision (paper: unpinned hub id; the GPQA server
#     reported a snapshot path as its name; the client requires LoRA cards whose parent is Qwen/Qwen3.5-9B);
#   * concurrency: the RSI cache is filled advisor_workers (32) requests at a time; the paper servers logged
#     "Running: 1 reqs" throughout (vLLM is not batch-invariant; the controller rechecks a sample sequentially);
#   * the LoRAs are served from renamed copies (LORA_MODE=multimodal, default; `serve-loras`). The PEFT
#     keys base_model.model.model.layers.N.* match no module of the multimodal model (whose language
#     model is language_model.model.layers.N.*), so vLLM would load the adapters and silently serve the
#     plain base. The copies carry the identical tensors under base_model.model.model.language_model.layers.N.*,
#     which vLLM's Qwen3-VL weight mapper maps onto the language model. LORA_MODE=as_is serves the
#     original files (the paper-era behaviour) for diagnosis only.
# `preflight` (python -m src.manager.mcq_rsi preflight) then proves behaviourally that the LoRAs change
# the outputs and reproduce the recorded paper outputs (exactly, or closer than the base under numeric
# drift); the controller refuses to fetch before it passes. The flags fingerprint written next to the
# served LoRAs (server_flags_<port>.json) binds the passing preflight to these flags and this vLLM version.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VLLM_VENV="${VLLM_VENV:-/workspace/vllm-venv}"
MCQ_PYTHON="${MCQ_PYTHON:-/workspace/mcq-venv/bin/python}"
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
IMPORT_DIR="${MCQ_IMPORT_DIR:-/workspace/mcq_rsi/import}"
SERVED_DIR="${MCQ_SERVED_LORAS:-/workspace/mcq_rsi/served_loras}"
LORA_MODE="${LORA_MODE:-multimodal}"
BENCHES="${MCQ_BENCHES:-all}"
HOST="${MCQ_ADVISOR_HOST:-127.0.0.1}"
PORT="${MCQ_ADVISOR_PORT:-18002}"
GPU="${MCQ_ADVISOR_GPU:-0}"
GPU_UTIL="${MCQ_ADVISOR_GPU_UTIL:-0.45}"
MAX_MODEL_LEN="${MCQ_ADVISOR_MAX_MODEL_LEN:-16384}"
BASE_MODEL="Qwen/Qwen3.5-9B"
BASE_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LOG_DIR="${MCQ_LOG_DIR:-/workspace/mcq_rsi/logs}"
LOG="${LOG_DIR}/vllm_advisors_${PORT}.log"
PIDFILE="${LOG_DIR}/vllm_advisors_${PORT}.pid"  # on the volume: survives a pod restart, so every use checks the command
HEALTH_TIMEOUT="${MCQ_ADVISOR_HEALTH_TIMEOUT:-1200}"
RUNPOD_PORTS="3001 7270 7861 8001 8081 9091"  # held by RunPod's nginx on the previous pod
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

log() { printf '[mcq-advisors] %s\n' "$*"; }
die() { printf '[mcq-advisors] ERROR: %s\n' "$*" >&2; exit 1; }

port_free() {
  # Bind test on the requested host and on 0.0.0.0 (nginx-style listeners hold the wildcard address).
  python3 - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
for h in dict.fromkeys([host, "0.0.0.0"]):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((h, port))
    except OSError as e:
        print(f"port {port} on {h} is in use: {e}")
        sys.exit(1)
    finally:
        s.close()
PY
}

served_pid() {
  # The pid must still be our vLLM server: after a pod restart the recorded pid can belong to anything.
  [[ -f "$PIDFILE" ]] || return 0
  local pid cmd
  pid="$(cat "$PIDFILE")"
  kill -0 "$pid" 2>/dev/null || return 0
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  else
    cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
  fi
  if [[ "$cmd" == *vllm.entrypoints* && "$cmd" == *"--port ${PORT}"* ]]; then
    echo "$pid"
  else
    log "stale pid file ${PIDFILE} (pid ${pid} is not this vLLM server); removing it" >&2
    rm -f "$PIDFILE"
  fi
}

evidence() {
  log "evidence from ${LOG}:"
  grep -E "version [0-9]+\.[0-9]+|Resolved architecture|non-default args|Loaded new LoRA adapter|max_loras|PunicaWrapper is found; model\.|supports adding LoRA" "$LOG" \
    | grep -v "visual\." | cut -c1-400 | head -40 || true
  log "/v1/models:"
  curl -fsS "http://${HOST}:${PORT}/v1/models" | python3 -c '
import json, sys
for m in json.load(sys.stdin)["data"]:
    print("  %-22s parent=%s root=%s" % (m["id"], m.get("parent"), m.get("root")))'
  log "LoRA key layout of the served files (expect base_model.model.model.language_model.layers. for multimodal):"
  cat "${SERVED_DIR}/serve_loras.log" 2>/dev/null | sed 's/^/  /' || true
}

start() {
  local pid
  pid="$(served_pid)"
  [[ -z "$pid" ]] || die "already running (pid ${pid}); use: $0 stop"
  case " ${RUNPOD_PORTS} " in *" ${PORT} "*) die "port ${PORT} is a RunPod nginx port; choose another (default 18002)";; esac
  port_free || die "port ${PORT} is in use (ss -ltnp | grep :${PORT}); set MCQ_ADVISOR_PORT to a free port"
  [[ -x "${VLLM_VENV}/bin/python" ]] || die "no vLLM venv at ${VLLM_VENV} (run scripts/setup_mcq_rsi_pod.sh)"
  local have
  have="$("${VLLM_VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')"
  if [[ "$have" != "$VLLM_VERSION" && "${ALLOW_VLLM_VERSION:-0}" != 1 ]]; then
    die "vLLM ${have} != pinned ${VLLM_VERSION} (paper servers ran 0.26.0); set ALLOW_VLLM_VERSION=1 to override and re-run preflight"
  fi
  [[ -x "$MCQ_PYTHON" ]] || die "no experiment python at ${MCQ_PYTHON}"
  if command -v nvidia-smi >/dev/null; then
    local free
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    log "GPU ${GPU} free memory: ${free} MiB"
    (( free >= 40000 )) || die "GPU ${GPU} has ${free} MiB free; need >= 40000 for the advisor server"
  fi
  mkdir -p "$LOG_DIR" "$SERVED_DIR"
  log "preparing served LoRAs (mode ${LORA_MODE}) in ${SERVED_DIR}"
  "$MCQ_PYTHON" -m src.manager.mcq_rsi serve-loras --bench "$BENCHES" --import-dir "$IMPORT_DIR" \
    --out "$SERVED_DIR" --mode "$LORA_MODE" | tee "${SERVED_DIR}/serve_loras.log"
  mapfile -t MODULES < "${SERVED_DIR}/lora_modules.txt"
  # Extra already-renamed adapters, e.g. retrained advisors: MCQ_EXTRA_LORAS="medqa_extractor_v2=/path ..."
  if [[ -n "${MCQ_EXTRA_LORAS:-}" ]]; then
    read -r -a extra <<< "$MCQ_EXTRA_LORAS"
    MODULES+=("${extra[@]}")
    log "extra LoRAs: ${extra[*]}"
  fi
  (( ${#MODULES[@]} > 0 )) || die "no LoRA modules"
  local max_loras=${#MODULES[@]}
  local args=(--model "$BASE_MODEL" --revision "$BASE_REVISION" --served-model-name "$BASE_MODEL"
    --host "$HOST" --port "$PORT" --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_UTIL" --enable-lora --max-loras "$max_loras" --max-cpu-loras "$max_loras"
    --max-lora-rank 16 --lora-modules "${MODULES[@]}")
  # Flags fingerprint read by preflight / require_passed (a passing preflight is bound to these flags).
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" 2>/dev/null | head -n 1 || true)"
  # Advisors decode greedily, which never uses the sampler; flashinfer's top-k/top-p sampler would JIT-compile a
  # kernel at startup (needs ninja + a matching nvcc; it failed on the first A100 pod), so it is off by default.
  local fi_sampler="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  python3 - "${SERVED_DIR}/server_flags_${PORT}.json" "$have" "$LORA_MODE" "$gpu_name" "$fi_sampler" "${args[@]}" <<'PY'
import json, sys
path, version, mode, gpu, fi_sampler, *args = sys.argv[1:]
json.dump({"vllm_version": version, "lora_mode": mode, "gpu": gpu, "args": args,
           "env": {"VLLM_USE_FLASHINFER_SAMPLER": fi_sampler}}, open(path, "w"), indent=2, sort_keys=True)
PY
  # A fresh log per start (the previous one is kept), so `evidence` shows this server's lines only.
  [[ ! -f "$LOG" ]] || mv "$LOG" "${LOG%.log}.$(date +%Y%m%d_%H%M%S).log"
  log "serving ${max_loras} LoRAs on ${HOST}:${PORT} (GPU ${GPU}, utilisation ${GPU_UTIL}); log ${LOG}"
  # The venv's bin on PATH: JIT builders (ninja) are found without activating the venv.
  CUDA_VISIBLE_DEVICES="$GPU" PATH="${VLLM_VENV}/bin:${PATH}" VLLM_USE_FLASHINFER_SAMPLER="$fi_sampler" \
    nohup "${VLLM_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    "${args[@]}" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  log "pid $(cat "$PIDFILE"); waiting for /health (up to ${HEALTH_TIMEOUT}s)"
  local waited=0
  until curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -n 40 "$LOG"; die "vLLM exited during startup"; }
    (( waited < HEALTH_TIMEOUT )) || { tail -n 40 "$LOG"; die "vLLM not healthy after ${HEALTH_TIMEOUT}s"; }
    sleep 5
    waited=$((waited + 5))
  done
  log "healthy after ${waited}s"
  evidence
  log "next: python -m src.manager.mcq_rsi preflight --config configs/mcq_rsi_<bench>.json"
}

stop() {
  local pid
  pid="$(served_pid)"
  [[ -n "$pid" ]] || { log "not running"; return 0; }
  kill "$pid"
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  rm -f "$PIDFILE"
  log "stopped ${pid}"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) pid="$(served_pid)"; if [[ -n "$pid" ]]; then log "running pid ${pid} on ${HOST}:${PORT}"; else log "not running"; fi ;;
  evidence) evidence ;;
  *) die "usage: $0 [start|stop|status|evidence]" ;;
esac
