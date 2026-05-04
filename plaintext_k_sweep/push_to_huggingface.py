#!/usr/bin/env python3
"""
Upload plaintext_k_sweep data + model to a single HuggingFace repo.

Repo: prompterminal/fdm-plaintext-k-sweep
Layout:
    data/K160/{train,eval_fixed,eval_random}.jsonl
    data/K320/...
    data/K480/...
    data/K960/...
    data/K960_isolated/...
    model/K960_isolated/final/  (model weights + config + tokenizer)

Note: Only K=960_isolated has a trained model; others have data only.
"""
import os
from huggingface_hub import HfApi, create_repo, login

REPO_ID = "prompterminal/fdm-plaintext-k-sweep"
ROOT = "/workspace/FDM_IN_WEIGHTS/plaintext_k_sweep"

# === LOGIN ===
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    hf_token = input("Enter your HF token: ").strip()
login(token=hf_token)

# === CREATE REPO ===
print(f"Creating repo {REPO_ID} (if it doesn't exist)...")
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"  Repo ready")
except Exception as e:
    print(f"  Repo creation: {e}")

api = HfApi()

# === PRE-FLIGHT: list what will be uploaded ===
data_dirs = ["K160", "K320", "K480", "K960", "K960_isolated"]
total_size_gb = 0

print("\n=== Upload plan ===")
for k in data_dirs:
    src = os.path.join(ROOT, f"data_{k}")
    if not os.path.isdir(src):
        print(f"  [skip] data_{k}: not found")
        continue
    sz = sum(os.path.getsize(os.path.join(src, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)))
    sz_gb = sz / (1024**3)
    total_size_gb += sz_gb
    print(f"  data/{k}/  ({sz_gb:.2f} GB)")

# Model
model_src = os.path.join(ROOT, "models/qwen3_0p6b_plaintext960_isolated/final")
if os.path.isdir(model_src):
    sz = sum(
        os.path.getsize(os.path.join(model_src, f))
        for f in os.listdir(model_src)
        if os.path.isfile(os.path.join(model_src, f))
    )
    sz_gb = sz / (1024**3)
    total_size_gb += sz_gb
    print(f"  model/K960_isolated/final/  ({sz_gb:.2f} GB)")

print(f"\nTotal upload: {total_size_gb:.2f} GB")
print(f"Estimated time at 100 MB/s: ~{total_size_gb * 1024 / 100 / 60:.0f} min")
print(f"\nView at: https://huggingface.co/{REPO_ID}")
print()
confirm = input("Proceed with upload? (y/n): ").strip().lower()
if confirm != "y":
    print("Aborted.")
    exit(0)

# === UPLOAD DATA ===
for k in data_dirs:
    src = os.path.join(ROOT, f"data_{k}")
    if not os.path.isdir(src):
        continue
    target_path = f"data/{k}"
    print(f"\n  Uploading data_{k} -> {target_path}/ ...")
    api.upload_folder(
        folder_path=src,
        path_in_repo=target_path,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Upload data/{k}",
        ignore_patterns=["*.pyc", "__pycache__/*"],
    )
    print(f"  -> Done")

# === UPLOAD MODEL ===
if os.path.isdir(model_src):
    print(f"\n  Uploading model/K960_isolated/final/ ...")
    api.upload_folder(
        folder_path=model_src,
        path_in_repo="model/K960_isolated/final",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload K960_isolated final model",
        ignore_patterns=["*.pyc", "__pycache__/*"],
    )
    print(f"  -> Done")

print(f"\n  All uploads complete!")
print(f"  View at: https://huggingface.co/{REPO_ID}")
