#!/usr/bin/env bash
# Extra training seeds of the paper's one-round MARGENT manager S_1 (rebuttal, 2026-11).
# Recipe = the paper's manager SFT, read from the training logs kept with the paper assets (logs/sft_*9b*.log in
# assets_0814.tgz: "[MANAGER_SFT] ... lr=1e-05 | epochs=10", LoRA r=16 alpha=32 on the attention and MLP projections,
# batch 1 x gradient accumulation 8, max_seq_len 4096), on the imported round-1 labels of each benchmark
# ($WORK/import/<bench>/round1/labels.jsonl: the rows S_1 itself was trained on). Only the seed changes.
# A finished adapter is moved to $CKPT/<bench>_seed<K>, where scripts/pod_ops/verifier_grid_queue.py picks it up; while
# this chain runs, $CKPT/TRAINING tells the queue to wait for those checkpoints. Idempotent: finished seeds are skipped.
#   SEEDS="1 2" BENCHES="gpqa medqa mmlu_pro aqua" SFT_GPU=1 bash scripts/pod_ops/seed_sft_chain.sh
# PLAN lists what to train, in order: "seed1 sucd seed2" (default "seed1 seed2").
#   seedK  the S_1 labels with training seed K
#   sucd   matched-data success distillation: labels rebuilt from the SAME round-1 records with the same commit cap by
#          `python -m src.manager.mcq_rsi select --arm success` (one successful option drawn at random per kept record
#          instead of the shortest one), trained with the S_1 recipe and the training command's default seed (42).
#          Before that, `select --arm dynamic` on the same records must reproduce the rows of the S_1 label file
#          in the same order (checked: 315 / 331 / 148 / 402 rows on 2026-11-26).
#   cmto   commit-only control: the selection of the same records with NO commit cap (`select --arm dynamic --rho -1`),
#          keeping only the rows whose decision is the root commit (the questions the manager answers correctly on its
#          own: DRAFT_ANSWER_X then ANSWER_X). The manager is trained with the S_1 recipe and seed 42 and never sees a
#          call or an advisor output: the "equally trained manager-only" control.
# Log: $WORK/logs/verifier_grid/seed_sft.log
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1 MARGENT_WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${SFT_GPU:-1}"
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python
CKPT=$WORK/ckpt/verifier_grid; OUT=$WORK/runs/verifier_grid/seed_sft; LOG=$WORK/logs/verifier_grid/seed_sft.log
SEEDS="${SEEDS:-1 2}"; BENCHES="${BENCHES:-gpqa medqa mmlu_pro aqua}"
PLAN="${PLAN:-$(for k in $SEEDS; do printf 'seed%s ' "$k"; done)}"
mkdir -p $CKPT $OUT $WORK/logs/verifier_grid
log() { printf '[seed-sft %s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $LOG; }
BASE=$(cat $CKPT/base_model_path.txt) || { log "no $CKPT/base_model_path.txt (run verifier_grid_chain.sh first)"; exit 1; }
touch $CKPT/TRAINING; trap 'rm -f $CKPT/TRAINING' EXIT
cd $TREE || exit 1
failed=0
success_labels() {  # $1 = benchmark; prints the label file, or fails
  local b=$1 dir=$OUT/labels records=$WORK/import/$1/round1/records.jsonl
  [ -f $WORK/import/$b/round1/label_records.jsonl ] && records=$WORK/import/$b/round1/label_records.jsonl  # GPQA: depth-1 records
  mkdir -p $dir
  CUDA_VISIBLE_DEVICES= $PY -m src.manager.mcq_rsi select --bench $b --arm dynamic --records $records --out $dir/${b}_dynamic_check.jsonl >> $LOG 2>&1 || return 1
  # same rows in the same order as the file S_1 was trained on (the selector only adds a "split" field)
  if ! $PY - $dir/${b}_dynamic_check.jsonl $WORK/import/$b/round1/labels.jsonl <<'PYEOF'
import json, sys
core = lambda r: (r["example_id"], r["decision_type"], json.dumps(r["prompt"], sort_keys=True), json.dumps(r["response"], sort_keys=True))
a, b = ([core(json.loads(l)) for l in open(p, encoding="utf-8") if l.strip()] for p in sys.argv[1:3])
sys.exit(0 if a == b else 1)
PYEOF
  then log "$b: the dynamic selection of $records does NOT reproduce the S_1 labels; no success labels are built" >&2; return 1; fi
  CUDA_VISIBLE_DEVICES= $PY -m src.manager.mcq_rsi select --bench $b --arm success --records $records --out $dir/${b}_success.jsonl >> $LOG 2>&1 || return 1
  echo $dir/${b}_success.jsonl
}
commit_only_labels() {  # $1 = benchmark; prints the label file, or fails
  local b=$1 dir=$OUT/labels records=$WORK/import/$1/round1/records.jsonl
  [ -f $WORK/import/$b/round1/label_records.jsonl ] && records=$WORK/import/$b/round1/label_records.jsonl
  mkdir -p $dir
  CUDA_VISIBLE_DEVICES= $PY -m src.manager.mcq_rsi select --bench $b --arm dynamic --rho -1 --records $records --out $dir/${b}_nocap.jsonl >> $LOG 2>&1 || return 1
  $PY - $dir/${b}_nocap.jsonl $dir/${b}_commit_only.jsonl >> $LOG 2>&1 <<'PYEOF' || return 1
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
keep = [r for r in rows if r["decision_type"] == "commit"]
assert keep and len(keep) < len(rows), (len(keep), len(rows))
assert len({r["question_hash"] for r in keep}) == len(keep), "one root commit per question"
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
print("commit-only labels:", len(keep), "of", len(rows), "rows;", dict(collections.Counter(r["decision_type"] for r in rows)))
PYEOF
  echo $dir/${b}_commit_only.jsonl
}
for item in $PLAN; do for b in $BENCHES; do
  name=${b}_${item}; out=$OUT/$name
  if [ -f $CKPT/$name/adapter_model.safetensors ] && [ -f $CKPT/$name/adapter_config.json ]; then log "$name: present"; continue; fi
  case $item in
    seed[0-9]*) seed=${item#seed}; labels=$WORK/import/$b/round1/labels.jsonl ;;
    sucd) seed=42; labels=$(success_labels $b) || { log "$name: FAILED to build the success labels"; failed=1; continue; } ;;
    cmto) seed=42; labels=$(commit_only_labels $b) || { log "$name: FAILED to build the commit-only labels"; failed=1; continue; } ;;
    *) log "$name: unknown plan item $item"; failed=1; continue ;;
  esac
  [ -f $labels ] || { log "$name: FAILED, no $labels"; failed=1; continue; }
  rm -rf $out $CKPT/.$name.part
  log "$name: start ($(grep -c . $labels) rows from $(basename $labels), seed $seed, gpu $CUDA_VISIBLE_DEVICES)"
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
