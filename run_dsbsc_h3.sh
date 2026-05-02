#!/bin/bash
cd /workspace/FDM_IN_WEIGHTS
python fdm_carrier_head_dsb_sc_hermes3.py \
    --n_carrier_channels 16 \
    --n_steps 5000 \
    --lr 3e-4 \
    --n_eval 100 \
    --seed 42 \
    --output_dir carrier_head_dsb_sc_hermes3_16_16 \
    2>&1 | tee carrier_head_dsb_sc_hermes3_16_16.log
