#!/usr/bin/env python3
"""
Re-evaluate the seed=7 carrier head at n=200 x 2 (same protocol as
carrier_head_reeval.py for seed=42). Two eval rounds with seeds 999
and 2024 to estimate eval-side variance.

Total runtime: ~30-40 minutes on H200.
"""

import os, sys, json, random, math
import torch

WORK_DIR = "/workspace/FDM_IN_WEIGHTS"
CARRIER_DIR = os.path.join(WORK_DIR, "carrier_head_16_16_seed7")  # NOTE: seed7 dir
HOST_MODEL = os.path.join(WORK_DIR, "two_block_model")
N_EVAL = 200
N_CARRIER_CHANNELS = 16

sys.path.insert(0, WORK_DIR)
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

import importlib.util
spec = importlib.util.spec_from_file_location(
    "fdm_ch", os.path.join(WORK_DIR, "fdm_carrier_head.py")
)
fdm_ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdm_ch)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("CARRIER HEAD seed=7 RE-EVAL at n=200 (two eval seeds)")
print("=" * 70)

print(f"\nLoading host model from {HOST_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(HOST_MODEL, trust_remote_code=True)
host_model = AutoModelForCausalLM.from_pretrained(
    HOST_MODEL, dtype=torch.float32, trust_remote_code=True
).to(device)
host_model.eval()
for p in host_model.parameters():
    p.requires_grad = False

n_layers, n_kv_heads, head_dim = fdm_ch.detect_kv_shape(
    host_model, host_model.config, device
)
print(f"  Detected: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

print(f"\nConstructing CarrierAttentionHead (n_carriers={N_CARRIER_CHANNELS})...")
carrier_head = fdm_ch.CarrierAttentionHead(
    n_layers=n_layers,
    kv_dim=head_dim,
    n_kv_heads=n_kv_heads,
    n_carriers=N_CARRIER_CHANNELS,
    n_kv_positions=513,
    sample_rate=100.0,
    n_values=64,
).to(device)

state_path = os.path.join(CARRIER_DIR, "carrier_head.pt")
print(f"  Loading state from {state_path}")
state_dict = torch.load(state_path, map_location=device)
carrier_head.load_state_dict(state_dict, strict=True)
carrier_head.eval()

encoder = fdm_ch.make_encoder(tokenizer)
print(f"  Encoder built (vocab_size=151936 hardcoded)")

carrier_channels = list(range(8, 8 + N_CARRIER_CHANNELS))
context_channels = list(range(8 + N_CARRIER_CHANNELS, 40))
print(f"\nPartition: carrier={carrier_channels[0]}..{carrier_channels[-1]}, "
      f"context={context_channels[0]}..{context_channels[-1]}")

# Original n=100 from training-time eval
results_path = os.path.join(CARRIER_DIR, "carrier_head_results.json")
original = None
if os.path.exists(results_path):
    with open(results_path) as f:
        original = json.load(f)
    print(f"\nOriginal seed=7 training-time n=100 results:")
    print(f"  Carrier slot:  {original.get('carrier_slot', 0)*100:.1f}%")
    print(f"  Context slot:  {original.get('context_slot', 0)*100:.1f}%")
    print(f"  All joint:     {original.get('all_joint', 0)*100:.1f}%")

# ---- Round 1 ----
print(f"\n{'-' * 70}")
print(f"Round 1: n={N_EVAL}, eval_seed=999")
print(f"{'-' * 70}")
random.seed(999)
torch.manual_seed(999)
result_999 = fdm_ch.evaluate_carrier_head(
    host_model, tokenizer, carrier_head, encoder, device,
    carrier_channels, context_channels, n_eval=N_EVAL,
)

# ---- Round 2 ----
print(f"\n{'-' * 70}")
print(f"Round 2: n={N_EVAL}, eval_seed=2024")
print(f"{'-' * 70}")
random.seed(2024)
torch.manual_seed(2024)
result_2024 = fdm_ch.evaluate_carrier_head(
    host_model, tokenizer, carrier_head, encoder, device,
    carrier_channels, context_channels, n_eval=N_EVAL,
)

# ---- Summary ----
print(f"\n{'=' * 70}")
print(f"SUMMARY: carrier_head_16_16_seed7 re-eval at n={N_EVAL}")
print(f"{'=' * 70}")
if original is not None:
    print(f"  Training-time (n=100):     "
          f"car_slot={original.get('carrier_slot', 0)*100:.1f}%, "
          f"all_joint={original.get('all_joint', 0)*100:.1f}%")
print(f"  Round 1 (eval_seed=999):   "
      f"car_slot={result_999['carrier_slot']*100:.1f}%, "
      f"ctx_slot={result_999['context_slot']*100:.1f}%, "
      f"all_joint={result_999['all_joint']*100:.1f}%")
print(f"  Round 2 (eval_seed=2024):  "
      f"car_slot={result_2024['carrier_slot']*100:.1f}%, "
      f"ctx_slot={result_2024['context_slot']*100:.1f}%, "
      f"all_joint={result_2024['all_joint']*100:.1f}%")

combined_carrier = (result_999['carrier_correct'] + result_2024['carrier_correct']) / \
                   (result_999['carrier_total'] + result_2024['carrier_total'])
combined_context = (result_999['context_correct'] + result_2024['context_correct']) / \
                   (result_999['context_total'] + result_2024['context_total'])
n_total = result_999['n_samples'] + result_2024['n_samples']
combined_joint_all = (result_999['all_joint'] * result_999['n_samples'] +
                      result_2024['all_joint'] * result_2024['n_samples']) / n_total

def wilson_hw(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    return (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)

joint_correct_total = round(combined_joint_all * n_total)
joint_hw = wilson_hw(joint_correct_total, n_total)

print(f"\n  Combined (n={n_total}):     "
      f"car_slot={combined_carrier*100:.1f}%, "
      f"ctx_slot={combined_context*100:.1f}%, "
      f"all_joint={combined_joint_all*100:.1f}% +/- {joint_hw*100:.1f}pp")

# Cross-seed comparison
print(f"\n{'=' * 70}")
print(f"CROSS-SEED COMPARISON")
print(f"{'=' * 70}")
print(f"  seed=42 (n=400):    car_slot=98.2%, ctx_slot=97.8%, joint=57.8% +/- 4.8pp")
print(f"  seed=7  (n={n_total}):    "
      f"car_slot={combined_carrier*100:.1f}%, "
      f"ctx_slot={combined_context*100:.1f}%, "
      f"joint={combined_joint_all*100:.1f}% +/- {joint_hw*100:.1f}pp")

slot_diff = abs(combined_carrier - 0.982) * 100
joint_diff = abs(combined_joint_all - 0.578) * 100
print(f"\n  Slot difference between seeds:  {slot_diff:.1f}pp")
print(f"  Joint difference between seeds: {joint_diff:.1f}pp")
if slot_diff < 1.0 and joint_diff < 5.0:
    print(f"  --> Strong reproducibility. Training is stable across seeds.")
elif slot_diff < 2.0 and joint_diff < 10.0:
    print(f"  --> Acceptable reproducibility. Some seed variance.")
else:
    print(f"  --> High training variance. Need more seeds to characterize.")

# Save
output = {
    "trained_seed": 7,
    "training_time_n100": original,
    "round_999": result_999,
    "round_2024": result_2024,
    "combined": {
        "n_samples": n_total,
        "carrier_slot": combined_carrier,
        "context_slot": combined_context,
        "all_joint": combined_joint_all,
        "joint_ci_pp_wilson": joint_hw * 100,
    },
    "comparison_to_seed42": {
        "seed42_slot": 0.982,
        "seed42_joint": 0.578,
        "slot_diff_pp": slot_diff,
        "joint_diff_pp": joint_diff,
    },
}
out_path = os.path.join(CARRIER_DIR, f"reeval_n{N_EVAL}x2.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved to {out_path}")
