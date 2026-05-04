#!/usr/bin/env python3
"""Upload Qwen3 5-seed partition sweep results to HuggingFace."""
import os
from huggingface_hub import HfApi, create_repo, login

REPO_ID = "prompterminal/fdm-40ch-partition-sweep-qwen3"
ROOT = "/workspace/FDM_IN_WEIGHTS/split_ratio_sweep_5seed"

hf_token = os.environ.get("HF_TOKEN") or input("HF token: ").strip()
login(token=hf_token)

print(f"Creating repo {REPO_ID}...")
create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

api = HfApi()
print(f"Uploading {ROOT} -> {REPO_ID}")
api.upload_folder(
    folder_path=ROOT,
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="5-seed × 9-partition sweep at 4K steps, n=200 eval per cell",
)
print(f"\nDone: https://huggingface.co/datasets/{REPO_ID}")
