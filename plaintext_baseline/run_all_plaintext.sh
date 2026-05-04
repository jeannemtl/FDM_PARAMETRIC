#!/usr/bin/env bash
# run_all_plaintext.sh — full plaintext-40 baseline sweep
# Usage:
#   tmux new -s plaintext40
#   bash run_all_plaintext.sh
#   (Ctrl+b d to detach, tmux attach -t plaintext40 to reattach)

set -e  # exit on first error
mkdir -p logs models results

echo "==========================================================="
echo "  Plaintext-40 baseline sweep"
echo "  Started: $(date)"
echo "==========================================================="

# -----------------------------------------------------------
# Step 1: Generate data (once, shared across all models)
# -----------------------------------------------------------
if [ ! -f data/train.jsonl ]; then
  echo ""
  echo ">>> Generating Plaintext-40 dataset..."
  python make_plaintext_data.py --out_dir data/ --n_train 115000 --n_eval 1500
else
  echo ">>> Data already exists, skipping generation"
fi

# -----------------------------------------------------------
# Step 2: Train + eval each architecture sequentially
# -----------------------------------------------------------

run_one_model() {
  local NAME=$1
  local MODEL=$2
  local BATCH=$3
  local GRAD_ACCUM=$4
  local LR=$5
  local EXTRA_FLAGS=$6

  echo ""
  echo "==========================================================="
  echo "  TRAINING: $NAME"
  echo "  Started:  $(date)"
  echo "==========================================================="

  python train_plaintext.py \
      --model_name "$MODEL" \
      --train_path data/train.jsonl \
      --output_dir "models/${NAME}" \
      --batch_size $BATCH \
      --grad_accum $GRAD_ACCUM \
      --lr $LR \
      --max_length 512 \
      $EXTRA_FLAGS \
      2>&1 | tee "logs/${NAME}_train.log"

  echo ""
  echo "  >>> Eval on FIXED ordering..."
  python eval_plaintext.py \
      --model_dir "models/${NAME}/final" \
      --eval_path data/eval_fixed.jsonl \
      --out_json "results/${NAME}_eval_fixed.json" \
      --batch_size 16 \
      2>&1 | tee "logs/${NAME}_eval_fixed.log"

  echo ""
  echo "  >>> Eval on RANDOM ordering..."
  python eval_plaintext.py \
      --model_dir "models/${NAME}/final" \
      --eval_path data/eval_random.jsonl \
      --out_json "results/${NAME}_eval_random.json" \
      --batch_size 16 \
      2>&1 | tee "logs/${NAME}_eval_random.log"

  echo "==========================================================="
  echo "  $NAME DONE: $(date)"
  echo "==========================================================="
}

# Order from cheapest to most expensive — fail-fast on cheap runs first
# Args:                 NAME              MODEL                                     BATCH GRAD_ACCUM LR     EXTRA
run_one_model  "qwen3_0p6b_plaintext"   "Qwen/Qwen3-0.6B"                         16    1          1e-4   ""
run_one_model  "gpt2_medium_plaintext"  "openai-community/gpt2-medium"             16    1          1e-4   ""
run_one_model  "lfm2_1p2b_plaintext"    "LiquidAI/LFM2-1.2B"                       8     2          1e-4   ""
run_one_model  "hermes3_3b_plaintext"   "NousResearch/Hermes-3-Llama-3.2-3B"       4     4          1e-4   ""

# -----------------------------------------------------------
# Step 3: Summary
# -----------------------------------------------------------
echo ""
echo "==========================================================="
echo "  SWEEP COMPLETE: $(date)"
echo "==========================================================="
echo ""
echo "Results files:"
ls -la results/
echo ""
echo "To view summary:"
echo "  python -c \"import json,glob; [print(f, json.load(open(f))['action']['p']) for f in sorted(glob.glob('results/*.json'))]\""
