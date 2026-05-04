#!/usr/bin/env bash
# run_K_sweep.sh — extend the plaintext baseline to K=160 and K=320
# Run this AFTER run_all_plaintext.sh has finished the K=40 sweep.
# Usage:
#   tmux new -s plaintext_K_sweep
#   bash run_K_sweep.sh

set -e
mkdir -p logs models results

echo "==========================================================="
echo "  Plaintext K-sweep (K=160, K=320 on Qwen3; conditional Hermes3)"
echo "  Started: $(date)"
echo "==========================================================="

# -----------------------------------------------------------
# Step 1: Generate K=160 and K=320 datasets
# -----------------------------------------------------------
for K in 160 320; do
  if [ ! -f "data_K${K}/train.jsonl" ]; then
    echo ""
    echo ">>> Generating Plaintext-${K} dataset..."
    python make_plaintext_data.py --K $K --out_dir "data_K${K}/" \
        --n_train 115000 --n_eval 1500
  else
    echo ">>> Plaintext-${K} dataset exists, skipping"
  fi
done

# -----------------------------------------------------------
# Step 2: Run Qwen3 at K=160 and K=320
# -----------------------------------------------------------

run_one_K() {
  local K=$1
  local NAME=$2
  local MODEL=$3
  local BATCH=$4
  local GRAD_ACCUM=$5
  local LR=$6
  local EXTRA=$7

  echo ""
  echo "==========================================================="
  echo "  TRAINING: $NAME (K=$K)"
  echo "  Started:  $(date)"
  echo "==========================================================="

  python train_plaintext.py \
      --model_name "$MODEL" \
      --train_path "data_K${K}/train.jsonl" \
      --output_dir "models/${NAME}" \
      --batch_size $BATCH \
      --grad_accum $GRAD_ACCUM \
      --lr $LR \
      $EXTRA \
      2>&1 | tee "logs/${NAME}_train.log"

  for SPLIT in fixed random; do
    echo ""
    echo "  >>> Eval ${NAME} on ${SPLIT} ordering..."
    python eval_plaintext.py \
        --model_dir "models/${NAME}/final" \
        --eval_path "data_K${K}/eval_${SPLIT}.jsonl" \
        --out_json "results/${NAME}_eval_${SPLIT}.json" \
        --batch_size 8 \
        2>&1 | tee "logs/${NAME}_eval_${SPLIT}.log"
  done
}

# K=160 on Qwen3 — ~1.5-2h
# (auto-scaled max_length=2048; reduced batch to fit memory + give attention room)
run_one_K 160 "qwen3_0p6b_plaintext160" "Qwen/Qwen3-0.6B"  4 4 1e-4 ""

# K=320 on Qwen3 — ~3-4h
# (auto-scaled max_length=4096; tighter batch + grad_accum for the longer context)
run_one_K 320 "qwen3_0p6b_plaintext320" "Qwen/Qwen3-0.6B"  2 8 1e-4 ""

# -----------------------------------------------------------
# Step 3: Decide whether to run Hermes3
# -----------------------------------------------------------
echo ""
echo "==========================================================="
echo "  Qwen3 K-sweep complete. Inspect results before Hermes3."
echo "==========================================================="
python -c "
import json, glob
print('Qwen3 plaintext K-sweep summary:')
for f in sorted(glob.glob('results/qwen3_*plaintext*_eval_random.json')):
    r = json.load(open(f))
    K = r.get('K', 40)
    name = f.split('/')[-1]
    print(f'  K={K:>3}  Action: {r[\"action\"][\"p\"]*100:5.1f}%  '
          f'Extra: {r[\"extra_slot\"][\"p\"]*100:5.1f}%  ({name})')
"

echo ""
echo "If Qwen3 K=320 shows clear degradation in Extra accuracy,"
echo "uncomment the Hermes3 lines below to extend the sweep."
echo "==========================================================="

# -----------------------------------------------------------
# Hermes3 K=160 and K=320 — uncomment after reviewing Qwen3 numbers
# -----------------------------------------------------------
# K=160 on Hermes3 — ~3-4h
# run_one_K 160 "hermes3_3b_plaintext160" "NousResearch/Hermes-3-Llama-3.2-3B"  4 4 1e-4 ""
#
# K=320 on Hermes3 — ~6-8h, may need grad_ckpt
# run_one_K 320 "hermes3_3b_plaintext320" "NousResearch/Hermes-3-Llama-3.2-3B"  2 8 1e-4 "--grad_ckpt"

echo ""
echo "==========================================================="
echo "  K-SWEEP DONE: $(date)"
echo "==========================================================="
