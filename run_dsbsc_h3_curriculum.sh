#!/bin/bash
cd /workspace/FDM_IN_WEIGHTS
python fdm_carrier_head_dsb_sc_hermes3_curriculum.py \
    --n_eval 100 \
    --seed 42 \
    --output_dir carrier_head_dsb_sc_hermes3_curriculum_16_16 \
    2>&1 | tee carrier_head_dsb_sc_hermes3_curriculum_16_16.log
