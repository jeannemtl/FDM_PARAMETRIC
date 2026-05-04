#!/bin/bash
# Push plaintext_k_sweep code + results to existing FDM_PARAMETRIC repo on GitHub
set -e

REPO_ROOT=/workspace/FDM_IN_WEIGHTS
SUBDIR=plaintext_k_sweep

cd "$REPO_ROOT"

# === Make sure .gitignore protects us from committing big stuff ===
GITIGNORE_LINE_DATA="$SUBDIR/data_*/"
GITIGNORE_LINE_MODELS="$SUBDIR/models/"
GITIGNORE_LINE_LOGS="$SUBDIR/logs/"
GITIGNORE_LINE_PYCACHE="$SUBDIR/__pycache__/"
GITIGNORE_LINE_JSONL="$SUBDIR/**/*.jsonl"

# Append to .gitignore if entries don't already exist
touch .gitignore
for line in "$GITIGNORE_LINE_DATA" "$GITIGNORE_LINE_MODELS" "$GITIGNORE_LINE_LOGS" "$GITIGNORE_LINE_PYCACHE" "$GITIGNORE_LINE_JSONL"; do
    grep -qxF "$line" .gitignore || echo "$line" >> .gitignore
done

echo "Updated .gitignore:"
tail -10 .gitignore
echo ""

# === Stage what we want ===
git add .gitignore
git add "$SUBDIR/"*.py
git add "$SUBDIR/"*.sh 2>/dev/null || true
git add "$SUBDIR/results/" 2>/dev/null || true

# Stage README if it exists
[ -f "$SUBDIR/README.md" ] && git add "$SUBDIR/README.md"

echo "=== Files to commit ==="
git status --short

echo ""
read -p "Commit and push? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 0
fi

git commit -m "Add plaintext K-sweep experiment: code + results

K-sweep tests plaintext encoding baseline at K=160, 320, 480, 960.
Includes K=960 isolated training variant.
Only K=960_isolated produced a saved model checkpoint.

Code: train_plaintext.py, train_plaintext_isolated.py, eval_plaintext.py,
      eval_plaintext_K960_isolated.py, channels.py, make_plaintext_data.py,
      convert_to_isolated.py, run_K_sweep.sh

Results: plaintext_K960_isolated_fixed.json, plaintext_K960_isolated_random.json

Data and model weights are NOT in this repo — see HuggingFace:
  prompterminal/fdm-plaintext-k-sweep
"

git push origin main
echo ""
echo "Pushed to GitHub."
