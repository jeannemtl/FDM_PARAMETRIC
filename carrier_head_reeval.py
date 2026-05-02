#!/usr/bin/env python3
"""
Re-evaluate the existing carrier_head_16_16 model at n=200 by importing
the training script's own evaluation function AND its make_encoder helper.

The previous version of this script built its own TurboFDMSignalEncoder with
vocab_size from model.config (151671). The training script hardcodes
vocab_size=151936 inside make_encoder(). The mismatch re-permutes the
token_map and produces signals the trained model can't decode (36% slot vs
trained 98.5%).

This version calls fdm_ch.make_encoder(tokenizer) so the encoder is
byte-identical to the one used at training time.

Total runtime: ~30-40 minutes on H200.
"""

import os, sys, json, random, math
import torch

WORK_DIR = "/workspace/FDM_IN_WEIGHTS"
CARRIER_DIR = os.path.join(WORK_DIR, "carrier_head_16_16")
HOST_MODEL = os.path.join(WORK_DIR, "two_block_model")
N_EVAL = 200
N_CARRIER_CHANNELS = 16

sys.path.insert(0, WORK_DIR)
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

# Import the training script as a module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fdm_ch", os.path.join(WORK_DIR, "fdm_carrier_head.py")
)
fdm_ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdm_ch)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("CARRIER HEAD RE-EVALUATION at n=200 (two seeds)")
print("=" * 70)

# Load host model
print(f"\nLoading host model from {HOST_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(HOST_MODEL, trust_remote_code=True)
host_model = AutoModelForCausalLM.from_pretrained(
    HOST_MODEL, dtype=torch.float32, trust_remote_code=True
).to(device)
host_model.eval()
for p in host_model.parameters():
    p.requires_grad = False

# Detect KV shape using training script's own function
n_layers, n_kv_heads, head_dim = fdm_ch.detect_kv_shape(
    host_model, host_model.config, device
)
print(f"  Detected: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

# Build carrier head with EXACT signature from training script main()
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

# Load saved weights
state_path = os.path.join(CARRIER_DIR, "carrier_head.pt")
print(f"  Loading state from {state_path}")
state_dict = torch.load(state_path, map_location=device)
carrier_head.load_state_dict(state_dict, strict=True)
carrier_head.eval()
print(f"  state_dict loaded strictly (all keys matched)")

# Build encoder using training script's own helper -- this guarantees
# identical token_map (vocab_size=151936 hardcoded inside make_encoder)
print(f"\nBuilding encoder via fdm_ch.make_encoder(tokenizer)...")
encoder = fdm_ch.make_encoder(tokenizer)
print(f"  Encoder vocab_size: 151936 (hardcoded in training script)")
print(f"  First 5 token_map entries: {list(encoder.token_map[:5])}")

# Channel partition: matches training script main() exactly
carrier_channels = list(range(8, 8 + N_CARRIER_CHANNELS))
context_channels = list(range(8 + N_CARRIER_CHANNELS, 40))
print(f"\nPartition: carrier={carrier_channels[0]}..{carrier_channels[-1]} "
      f"({len(carrier_channels)} ch), "
      f"context={context_channels[0]}..{context_channels[-1]} "
      f"({len(context_channels)} ch)")

# Print original n=100 result for comparison
results_path = os.path.join(CARRIER_DIR, "carrier_head_results.json")
original = None
if os.path.exists(results_path):
    with open(results_path) as f:
        original = json.load(f)
    print(f"\nOriginal n=100 results from {results_path}:")
    print(f"  Carrier slot:  {original.get('carrier_slot', 0)*100:.1f}%")
    print(f"  Context slot:  {original.get('context_slot', 0)*100:.1f}%")
    print(f"  Carrier joint: {original.get('carrier_joint', 0)*100:.1f}%")
    print(f"  Context joint: {original.get('context_joint', 0)*100:.1f}%")
    print(f"  All joint:     {original.get('all_joint', 0)*100:.1f}%")

# ---- Round 1: eval_seed = 999 ----
print(f"\n{'-' * 70}")
print(f"Round 1: n={N_EVAL}, eval_seed=999")
print(f"{'-' * 70}")
random.seed(999)
torch.manual_seed(999)
result_999 = fdm_ch.evaluate_carrier_head(
    host_model, tokenizer, carrier_head, encoder, device,
    carrier_channels, context_channels, n_eval=N_EVAL,
)

# ---- Round 2: eval_seed = 2024 ----
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
print(f"SUMMARY: carrier_head_16_16 re-eval at n={N_EVAL}")
print(f"{'=' * 70}")
if original is not None:
    print(f"  Original (seed=42, n=100): "
          f"car_slot={original.get('carrier_slot', 0)*100:.1f}%, "
          f"ctx_slot={original.get('context_slot', 0)*100:.1f}%, "
          f"all_joint={original.get('all_joint', 0)*100:.1f}%")
print(f"  Round 1 (seed=999, n={N_EVAL}):  "
      f"car_slot={result_999['carrier_slot']*100:.1f}%, "
      f"ctx_slot={result_999['context_slot']*100:.1f}%, "
      f"all_joint={result_999['all_joint']*100:.1f}%")
print(f"  Round 2 (seed=2024, n={N_EVAL}): "
      f"car_slot={result_2024['carrier_slot']*100:.1f}%, "
      f"ctx_slot={result_2024['context_slot']*100:.1f}%, "
      f"all_joint={result_2024['all_joint']*100:.1f}%")

# Combined
combined_carrier = (result_999['carrier_correct'] + result_2024['carrier_correct']) / \
                   (result_999['carrier_total'] + result_2024['carrier_total'])
combined_context = (result_999['context_correct'] + result_2024['context_correct']) / \
                   (result_999['context_total'] + result_2024['context_total'])
n_total_samples = result_999['n_samples'] + result_2024['n_samples']
combined_joint_all = (result_999['all_joint'] * result_999['n_samples'] +
                      result_2024['all_joint'] * result_2024['n_samples']) / n_total_samples
print(f"\n  Combined (n={n_total_samples}):    "
      f"car_slot={combined_carrier*100:.1f}%, "
      f"ctx_slot={combined_context*100:.1f}%, "
      f"all_joint={combined_joint_all*100:.1f}%")

# Wilson 95% CI
def wilson_hw(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    return (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)

joint_correct_total = round(combined_joint_all * n_total_samples)
joint_hw = wilson_hw(joint_correct_total, n_total_samples)
print(f"  Wilson 95% CI on combined joint: "
      f"{combined_joint_all*100:.1f}% +/- {joint_hw*100:.1f}pp")

# Eval-side variance
joint_diff = abs(result_999['all_joint'] - result_2024['all_joint'])
print(f"\n  Eval-side variance |R1-R2|: {joint_diff*100:.1f}pp")
if joint_diff < 0.05:
    print(f"  --> Low eval variance. Result is reproducible at fixed weights.")
elif joint_diff < 0.10:
    print(f"  --> Moderate eval variance. Combined n=400 estimate is more reliable.")
else:
    print(f"  --> High eval variance. Investigate before scaling experiments.")

# Save
output = {
    "original_n100": original,
    "round_999": result_999,
    "round_2024": result_2024,
    "combined": {
        "n_samples": n_total_samples,
        "carrier_slot": combined_carrier,
        "context_slot": combined_context,
        "all_joint": combined_joint_all,
        "joint_ci_pp_wilson": joint_hw * 100,
    },
}
out_path = os.path.join(CARRIER_DIR, f"reeval_n{N_EVAL}x2.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved to {out_path}")
