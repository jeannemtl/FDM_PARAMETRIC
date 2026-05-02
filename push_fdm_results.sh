#!/usr/bin/env bash
# push_fdm_results.sh
#
# Staged commit script for /workspace/FDM_IN_WEIGHTS.
# Strategy: trained model WEIGHTS go to HuggingFace (excluded by file pattern),
# but small results.json / config.json files inside model dirs DO get pushed
# to GitHub for paper reproducibility.
#
# Run from /workspace/FDM_IN_WEIGHTS:
#   bash push_fdm_results.sh

set -euo pipefail
cd /workspace/FDM_IN_WEIGHTS

# ----- Helpers -----
confirm() {
    read -p "$1 [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Skipping."; return 1; }
}

stage_if_exists() {
    for p in "$@"; do
        if [[ -e "$p" ]]; then
            git add "$p" || true
        fi
    done
}

# ============================================================
# Step 1: Update .gitignore (file-pattern based, not directory based)
# ============================================================
echo "============================================================"
echo "Step 1: Updating .gitignore"
echo "============================================================"

# Build new gitignore from scratch (clearer than merging)
cat > .gitignore << 'EOF'
# ============================================================
# Trained model weight files -- these go to HuggingFace, not git
# ============================================================
*.pt
*.safetensors

# Tokenizer artifacts (large, regenerable, ship via HF)
**/tokenizer.json

# HuggingFace cache directories
*/.cache/
.cache/

# ============================================================
# Backup and temporary files
# ============================================================
*.bak
*.pyc
__pycache__/

# ============================================================
# Smoke / debug artifacts (not reproducible, not paper-relevant)
# ============================================================
smoke_test.log
smoke_check.log
sweep_12_retry.log
two_block_model_lfm2_smoketest/

# ============================================================
# Specific large generated files
# ============================================================
# (file-pattern *.pt above already covers these, but listed for clarity)
# fdm_*.pt, turbo_*.pt, plain_write_head.pt, hybrid_write_head.pt
EOF

echo "Updated .gitignore:"
echo "----"
cat .gitignore
echo "----"
echo
echo "Effect: weights (.pt, .safetensors) excluded -- ship to HuggingFace"
echo "But: results.json, config.json, generation_config.json INSIDE model dirs"
echo "       WILL be tracked, since only weight files are excluded by pattern."
echo

confirm "Look right?" || exit 1
git add .gitignore
git commit -m "Update .gitignore: exclude trained weights (.pt, .safetensors) and tokenizer.json (-> HuggingFace), keep small JSONs"

# ============================================================
# Step 2: Sanity check -- nothing >90MB would be staged
# ============================================================
echo
echo "============================================================"
echo "Step 2: Verify no large files would be committed"
echo "============================================================"
echo
echo "Files >90MB anywhere in working dir (these MUST be in .gitignore or HF):"
find . -type f -size +90M -not -path "./.git/*" 2>/dev/null | head -30
echo
echo "GitHub hard limit: 100MB per file. Soft limit: 1GB per repo."
echo
echo "Running 'git add -n' dry-run on entire repo to see what would be staged..."
git add -n . 2>&1 | head -50
echo
echo "(scroll up to verify nothing huge or sensitive would sneak in)"
confirm "Looks safe?" || exit 1

# ============================================================
# Commit 1: Cache-substrate experiments (paper's core)
# ============================================================
echo
echo "============================================================"
echo "Commit 1: Cache-substrate experiments (paper main contribution)"
echo "============================================================"

# Python scripts
stage_if_exists \
    eval_split_source_n200.py \
    fdm_split_source_hybrid_hermes3.py \
    fdm_two_block_training_lfm2.py \
    download_hermes3_twoblock.py \
    fdm_split_ratio_sweep.py

# Result JSONs from cache-substrate experiments
stage_if_exists \
    split_source_hybrid/eval_n200x2.json \
    split_source_hybrid/results.json

# Two-block model dirs -- only the small files get staged thanks to gitignore
# (model.safetensors and tokenizer.json are excluded; results.json, config.json,
#  generation_config.json, chat_template.jinja, tokenizer_config.json get staged)
stage_if_exists \
    two_block_model/ \
    two_block_model_hermes3/

# Sweep result dirs (no .pt files inside, just JSONs and logs)
stage_if_exists \
    split_ratio_sweep/ \
    split_ratio_sweep_n200/ \
    split_ratio_sweep_n200_12/

# Logs
stage_if_exists \
    split_ratio_sweep_log.txt \
    sweep_n200.log \
    sweep_n200_qwen3.log \
    sweep_extended_log.txt \
    twoblock_log.txt \
    split_source_log.txt

echo
echo "Files staged for commit 1:"
git diff --cached --name-only
echo
echo "Total size of staged files:"
git diff --cached --name-only | xargs -I {} du -b {} 2>/dev/null | awk '{s+=$1} END {print s/1024 " KB"}'
confirm "Commit?" || { git reset; exit 1; }
git commit -m "Cache-substrate experiments: split-source eval (n=400 sample-level joint), partition sweep at n=200, two-block training results.json for Qwen3 and Hermes3, cross-architecture training scripts (LFM2/Hermes3 vocab patches)"

# ============================================================
# Commit 2: Carrier head / parametric weight space (Paper 2 prep)
# ============================================================
echo
echo "============================================================"
echo "Commit 2: Carrier head + DSB-SC experiments (Paper 2 work)"
echo "============================================================"

# Python scripts
stage_if_exists \
    fdm_carrier_head.py \
    fdm_carrier_head_hermes3.py \
    fdm_carrier_head_dsb_sc.py \
    fdm_carrier_head_dsb_sc_hermes3.py \
    fdm_carrier_head_dsb_sc_hermes3_curriculum.py \
    fdm_carrier_head_dsb_sc_qwen3.py \
    carrier_head_reeval.py \
    carrier_head_reeval_seed7.py \
    carrier_head_hermes3_reeval.py \
    carrier_head_position_independence.py \
    carrier_head_position_shift_eval.py

# Shell scripts
stage_if_exists \
    run_dsbsc_h3.sh \
    run_dsbsc_h3_curriculum.sh \
    run_dsbsc_qwen3.sh

# Logs
stage_if_exists \
    carrier_head_reeval.log \
    carrier_head_reeval_seed7.log \
    carrier_head_16_16_seed1337.log \
    carrier_head_16_16_seed7.log \
    carrier_head_dsb_sc_hermes3_16_16.log \
    carrier_head_dsb_sc_hermes3_curriculum_16_16.log \
    carrier_head_dsb_sc_qwen3_16_16.log \
    carrier_head_hermes3_16_16.log \
    carrier_head_hermes3_16_16_lr3e4.log \
    hermes3_download.log

# Carrier head experiment dirs -- staging the dirs is fine since
# the .pt files inside are excluded by gitignore *.pt pattern
stage_if_exists \
    carrier_head_16_16/ \
    carrier_head_16_16_seed1337/ \
    carrier_head_16_16_seed7/ \
    carrier_head_dsb_sc_hermes3_16_16/ \
    carrier_head_dsb_sc_hermes3_curriculum_16_16/ \
    carrier_head_dsb_sc_qwen3_16_16/ \
    carrier_head_hermes3_16_16/ \
    carrier_head_hermes3_16_16_lr3e4/

echo
echo "Files staged for commit 2:"
git diff --cached --name-only
echo
confirm "Commit?" || { git reset; exit 1; }
git commit -m "Carrier head experiments (Paper 2 prep): ASK + DSB-SC training scripts for Qwen3 and Hermes3, position-shift eval, multi-seed reeval, training logs and results JSONs"

# ============================================================
# Commit 3: Standalone weight retrieval (option_c series)
# ============================================================
echo
echo "============================================================"
echo "Commit 3: Standalone weight retrieval + FDM-V3 adaptive curriculum"
echo "============================================================"

stage_if_exists \
    option_c_weight_retrieval.py \
    option_c_v2_weight_retrieval.py \
    option_c_v3_lora.py \
    option_c_v4_curriculum.py \
    option_c_v5_curriculum_varying.py \
    option_c_v7_curriculum_amplitude.py \
    option_c_v8_extended_stage3.py \
    fdm_v3_adaptive.py \
    fdm_adaptive_results.json \
    v3_extended_log.txt

# These all-Qwen3 dirs from option_c experiments contain .safetensors + .pt
# excluded by gitignore patterns, but their tokenizer_config.json etc get staged
stage_if_exists \
    two_block_qwen3_curriculum/ \
    two_block_qwen3_lora/ \
    two_block_qwen3_low_lr/ \
    two_block_qwen3_v5/ \
    two_block_qwen3_v7/ \
    two_block_qwen3_v8/ \
    two_block_qwen3_weight_retrieval/

echo
echo "Files staged for commit 3:"
git diff --cached --name-only
echo
confirm "Commit?" || { git reset; exit 1; }
git commit -m "Standalone parametric weight retrieval (Paper 2 prep): option_c v1-v8 progression and FDM-V3 adaptive curriculum reaching 99.4% at 32 channels (115K cumulative steps, 1024 samples per encoder, 400Hz)"

# ============================================================
# Commit 4: Utilities
# ============================================================
echo
echo "============================================================"
echo "Commit 4: Utility scripts"
echo "============================================================"

stage_if_exists \
    compute_table5.py \
    fix_regex.py \
    patch_sweep.py \
    default_memory_probe.json

echo
echo "Files staged for commit 4:"
git diff --cached --name-only
echo
confirm "Commit?" || { git reset; exit 1; }
git commit -m "Utility scripts: table generation, regex fix, sweep patch, memory probe"

# ============================================================
# Commit 5: Anything else still unstaged
# ============================================================
echo
echo "============================================================"
echo "Commit 5: Catch-all for anything still unstaged"
echo "============================================================"

echo "Remaining unstaged changes:"
git status --short

if [[ -z "$(git status --porcelain)" ]]; then
    echo "Nothing left to commit."
else
    confirm "Stage all remaining tracked changes (git add -u)?" && {
        git add -u
        echo
        echo "Staged:"
        git diff --cached --name-only
        confirm "Commit?" && {
            git commit -m "Update tracked results files with latest run data"
        } || { git reset; }
    } || true
fi

# ============================================================
# Final review and push
# ============================================================
echo
echo "============================================================"
echo "Final review before push"
echo "============================================================"
git log --oneline -10
echo
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Remote: $(git remote get-url origin 2>/dev/null || echo 'none')"
echo
confirm "Push to origin?" || { echo "Done. Push manually when ready: git push"; exit 0; }
git push
echo
echo "Done."
