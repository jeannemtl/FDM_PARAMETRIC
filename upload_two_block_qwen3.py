#!/usr/bin/env python3
"""
Upload the two-block Qwen3 host model to HuggingFace.

This is the model that knows how to handle FDM content at TWO position ranges
(both as input tokens, with the architectural fine-tuning that enables
split-source composition). It's the host model for the partition-ratio sweep.
"""

import os
from huggingface_hub import HfApi, create_repo, login

# === CONFIG ===
LOCAL_DIR = "/workspace/FDM_IN_WEIGHTS/two_block_model"
REPO_ID = "prompterminal/fdm-40ch-two-block-qwen3"
COMMIT_MSG = "Upload two-block Qwen3 host for partition-ratio sweep"

# === LOGIN ===
# Either set HF_TOKEN env var, or paste your token when prompted
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    hf_token = input("Enter your HF token: ").strip()
login(token=hf_token)

# === VERIFY LOCAL DIR EXISTS ===
if not os.path.isdir(LOCAL_DIR):
    raise SystemExit(f"ERROR: {LOCAL_DIR} does not exist")

print(f"\nFiles in {LOCAL_DIR}:")
total_size = 0
for f in sorted(os.listdir(LOCAL_DIR)):
    full = os.path.join(LOCAL_DIR, f)
    if os.path.isfile(full):
        sz_mb = os.path.getsize(full) / 1024 / 1024
        total_size += sz_mb
        print(f"  {f}  ({sz_mb:.1f} MB)")
print(f"Total: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

# === CREATE REPO IF NEEDED ===
api = HfApi()
print(f"\nCreating repo {REPO_ID} (if it doesn't exist)...")
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"  Repo ready")
except Exception as e:
    print(f"  Repo creation: {e}")

# === UPLOAD ===
print(f"\nUploading {LOCAL_DIR} -> {REPO_ID} ...")
print("(this may take a while for multi-GB models)")

api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message=COMMIT_MSG,
    ignore_patterns=["*.pyc", "__pycache__/*", ".git/*"],
)

print(f"\nDone!")
print(f"View at: https://huggingface.co/{REPO_ID}")
