#!/usr/bin/env bash
# Extra training seeds of the paper's one-round MARGENT manager S_1 (rebuttal, 2026-11).
# Recipe = the paper's manager SFT, read from the training logs kept with the paper assets (logs/sft_*9b*.log in
# assets_0814.tgz: "[MANAGER_SFT] ... lr=1e-05 | epochs=10", LoRA r=16 alpha=32 on the attention and MLP projections,
# batch 1 x gradient accumulation 8, max_seq_len 4096), on the imported round-1 labels of each benchmark
# ($WORK/import/<bench>/round1/labels.jsonl: the rows S_1 itself was trained on). Only the seed changes.
# A finished adapter is moved to $CKPT/<bench>_seed<K>, where scripts/pod_ops/verifier_grid_queue.py picks it up; while
# this chain runs, $CKPT/TRAINING tells the queue to wait for those checkpoints. Idempotent: finished seeds are skipped.
#   SEEDS="1 2" BENCHES="gpqa medqa mmlu_pro aqua" SFT_GPU=1 bash scripts/pod_ops/seed_sft_chain.sh
# Log: $WORK/logs/verifier_grid/seed_sft.log
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1 MARGENT_WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${SFT_GPU:-1}"
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python
CKPT=$WORK/ckpt/verifier_grid; OUT=$WORK/runs/verifier_grid/seed_sft; LOG=$WORK/logs/verifier_grid/seed_sft.log
SEEDS="${SEEDS:-1 2}"; BENCHES="${BENCHES:-gpqa medqa mmlu_pro aqua}"
mkdir -p $CKPT $OUT $WORK/logs/verifier_grid
log() { printf '[seed-sft %s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $LOG; }
BASE=$(cat $CKPT/base_model_path.txt) || { log "no $CKPT/base_model_path.txt (run verifier_grid_chain.sh first)"; exit 1; }
touch $CKPT/TRAINING; trap 'rm -f $CKPT/TRAINING' EXIT
cd $TREE || exit 1
failed=0
for seed in $SEEDS; do for b in $BENCHES; do
  name=${b}_seed${seed}; out=$OUT/$name; labels=$WORK/import/$b/round1/labels.jsonl
  if [ -f $CKPT/$name/adapter_model.safetensors ] && [ -f $CKPT/$name/adapter_config.json ]; then log "$name: present"; continue; fi
  [ -f $labels ] || { log "$name: FAILED, no $labels"; failed=1; continue; }
  rm -rf $out $CKPT/.$name.part
  log "$name: start ($(wc -l < $labels) rows, seed $seed, gpu $CUDA_VISIBLE_DEVICES)"
  started=$(date +%s)
  $PY -m src.pipeline.cli train_manager_sft --base_model "$BASE" --teacher_id "$name" --output_root $OUT/_root --seed $seed \
      --manager_sft_train_jsonl $labels --manager_sft_output_dir $out --manager_sft_epochs 10 --manager_sft_lr 1e-5 \
      --sft_max_seq_len 4096 --sft_bs 1 --sft_grad_accum 8 > $WORK/logs/verifier_grid/seed_sft__$name.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f $out/adapter_model.safetensors ] && [ -f $out/adapter_config.json ]; then
    mkdir -p $CKPT/.$name.part
    for f in adapter_config.json adapter_model.safetensors chat_template.jinja tokenizer.json tokenizer_config.json; do
      [ -f $out/$f ] && cp $out/$f $CKPT/.$name.part/; done
    rm -rf $out/checkpoint-*
    mv $CKPT/.$name.part $CKPT/$name
    log "$name: ok in $(( $(date +%s) - started ))s; $(tr '\r' '\n' < $WORK/logs/verifier_grid/seed_sft__$name.log | grep -E "train examples|train_runtime" | cut -c1-150 | tr '\n' ' ')"
  else
    failed=1; log "$name: FAILED rc=$rc; $(tail -c 400 $WORK/logs/verifier_grid/seed_sft__$name.log | tr '\r\n' '  ')"
  fi
done; done
log "chain finished (failed=$failed)"
exit $failed
