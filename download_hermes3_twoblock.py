#!/usr/bin/env python3
"""Download two-block Hermes3 from private HF repo."""
import os, sys
from huggingface_hub import snapshot_download

REPO_ID = "prompterminal/fdm-40ch-two-block-hermes3"
LOCAL_DIR = "/workspace/FDM_IN_WEIGHTS/two_block_model_hermes3"

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not token:
    print("ERROR: set HF_TOKEN env var first")
    sys.exit(1)

os.makedirs(LOCAL_DIR, exist_ok=True)
print(f"Downloading {REPO_ID} -> {LOCAL_DIR}")

path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    token=token,
    local_dir_use_symlinks=False,
)
print(f"\nDone: {path}")
print("\nFiles:")
for f in sorted(os.listdir(LOCAL_DIR)):
    full = os.path.join(LOCAL_DIR, f)
    if os.path.isfile(full):
        sz = os.path.getsize(full)
        if sz > 1e9:
            print(f"  {f}  ({sz/1e9:.2f} GB)")
        elif sz > 1e6:
            print(f"  {f}  ({sz/1e6:.1f} MB)")
        else:
            print(f"  {f}  ({sz} B)")
    else:
        print(f"  {f}/")
