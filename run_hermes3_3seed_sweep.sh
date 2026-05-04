#!/bin/bash
set -e

SCRIPT=/workspace/FDM_IN_WEIGHTS/fdm_split_ratio_sweep.py
HOST_MODEL=/workspace/FDM_IN_WEIGHTS/two_block_hermes3
BASE_DIR=/workspace/FDM_IN_WEIGHTS/split_ratio_sweep_hermes3_3seed
SEEDS=(42 1337 7777)   # 3 seeds for Hermes3 (compute budget)
LOG=/workspace/FDM_IN_WEIGHTS/sweep_hermes3_run.log

mkdir -p "$BASE_DIR"

echo "===== Hermes3 3-seed sweep started at $(date) =====" | tee -a "$LOG"
echo "Seeds: ${SEEDS[@]}" | tee -a "$LOG"
echo "Host model: $HOST_MODEL" | tee -a "$LOG"
echo "Base dir: $BASE_DIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
    OUT_DIR="$BASE_DIR/seed_$seed"
    
    if [ -f "$OUT_DIR/sweep_results.json" ]; then
        N_RATIOS=$(python3 -c "import json; d=json.load(open('$OUT_DIR/sweep_results.json')); print(len(d))" 2>/dev/null || echo 0)
        if [ "$N_RATIOS" = "9" ]; then
            echo "[skip] seed=$seed already complete (9 ratios)" | tee -a "$LOG"
            continue
        fi
    fi
    
    mkdir -p "$OUT_DIR"
    
    echo "===== seed=$seed started at $(date) =====" | tee -a "$LOG"
    
    python3 "$SCRIPT" \
        --model "$HOST_MODEL" \
        --seed "$seed" \
        --n_steps 4000 \
        --n_eval 200 \
        --lr 1e-4 \
        --splits "0,4,8,12,16,20,24,28,32" \
        --output_dir "$OUT_DIR" \
        2>&1 | tee -a "$LOG"
    
    echo "===== seed=$seed completed at $(date) =====" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "===== ALL DONE at $(date) =====" | tee -a "$LOG"
