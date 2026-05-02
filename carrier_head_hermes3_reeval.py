#!/usr/bin/env python3
"""Hermes3 carrier head re-eval at n=400 (two rounds of 200)."""

import os, sys, json, random, math
import torch

WORK_DIR = "/workspace/FDM_IN_WEIGHTS"
CARRIER_DIR = os.path.join(WORK_DIR, "carrier_head_hermes3_16_16")
HOST_MODEL = os.path.join(WORK_DIR, "two_block_model_hermes3")
N_EVAL = 200
N_CARRIER_CHANNELS = 16

sys.path.insert(0, WORK_DIR)
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

import importlib.util
spec = importlib.util.spec_from_file_location(
    "fdm_ch_hermes", os.path.join(WORK_DIR, "fdm_carrier_head_hermes3.py")
)
fdm_ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdm_ch)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("HERMES3 CARRIER HEAD RE-EVAL at n=200 (two eval seeds)")
print("=" * 70)

print(f"\nLoading Hermes3 host model from {HOST_MODEL}...")
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
print(f"  Host vocab_size: {host_model.config.vocab_size}")

print(f"\nConstructing CarrierAttentionHead (n_carriers={N_CARRIER_CHANNELS})...")
carrier_head = fdm_ch.CarrierAttentionHead(
    n_layers=n_layers, kv_dim=head_dim, n_kv_heads=n_kv_heads,
    n_carriers=N_CARRIER_CHANNELS, n_kv_positions=513,
    sample_rate=100.0, n_values=64,
).to(device)

state_path = os.path.join(CARRIER_DIR, "carrier_head.pt")
print(f"  Loading state from {state_path}")
state_dict = torch.load(state_path, map_location=device)
carrier_head.load_state_dict(state_dict, strict=True)
carrier_head.eval()

encoder = fdm_ch.make_encoder(tokenizer)
print(f"  Encoder built (vocab_size=128258 for Hermes3)")
print(f"  First 5 token_map: {list(encoder.token_map[:5])}")

carrier_channels = list(range(8, 8 + N_CARRIER_CHANNELS))
context_channels = list(range(8 + N_CARRIER_CHANNELS, 40))
print(f"\nPartition: carrier={carrier_channels[0]}..{carrier_channels[-1]}, "
      f"context={context_channels[0]}..{context_channels[-1]}")

results_path = os.path.join(CARRIER_DIR, "carrier_head_results.json")
original = None
if os.path.exists(results_path):
    with open(results_path) as f:
        original = json.load(f)
    print(f"\nHermes3 training-time n=100:")
    print(f"  car_slot={original.get('carrier_slot', 0)*100:.1f}% "
          f"ctx_slot={original.get('context_slot', 0)*100:.1f}% "
          f"all_joint={original.get('all_joint', 0)*100:.1f}%")

# Round 1
print(f"\n{'-' * 70}\nRound 1: n={N_EVAL}, eval_seed=999\n{'-' * 70}")
random.seed(999)
torch.manual_seed(999)
result_999 = fdm_ch.evaluate_carrier_head(
    host_model, tokenizer, carrier_head, encoder, device,
    carrier_channels, context_channels, n_eval=N_EVAL,
)

# Round 2
print(f"\n{'-' * 70}\nRound 2: n={N_EVAL}, eval_seed=2024\n{'-' * 70}")
random.seed(2024)
torch.manual_seed(2024)
result_2024 = fdm_ch.evaluate_carrier_head(
    host_model, tokenizer, carrier_head, encoder, device,
    carrier_channels, context_channels, n_eval=N_EVAL,
)

# Summary
print(f"\n{'=' * 70}\nSUMMARY: Hermes3 carrier head re-eval at n={N_EVAL}\n{'=' * 70}")
print(f"  Round 1 (seed=999):  car={result_999['carrier_slot']*100:.1f}% "
      f"ctx={result_999['context_slot']*100:.1f}% "
      f"joint={result_999['all_joint']*100:.1f}%")
print(f"  Round 2 (seed=2024): car={result_2024['carrier_slot']*100:.1f}% "
      f"ctx={result_2024['context_slot']*100:.1f}% "
      f"joint={result_2024['all_joint']*100:.1f}%")

combined_car = (result_999['carrier_correct'] + result_2024['carrier_correct']) / \
               (result_999['carrier_total'] + result_2024['carrier_total'])
combined_ctx = (result_999['context_correct'] + result_2024['context_correct']) / \
               (result_999['context_total'] + result_2024['context_total'])
n_tot = result_999['n_samples'] + result_2024['n_samples']
combined_joint = (result_999['all_joint'] * result_999['n_samples'] +
                  result_2024['all_joint'] * result_2024['n_samples']) / n_tot

def wilson_hw(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)

joint_hw = wilson_hw(round(combined_joint * n_tot), n_tot)

print(f"\n  Combined (n={n_tot}): car={combined_car*100:.1f}% "
      f"ctx={combined_ctx*100:.1f}% "
      f"joint={combined_joint*100:.1f}% +/- {joint_hw*100:.1f}pp")

print(f"\n{'=' * 70}\nCROSS-ARCHITECTURE COMPARISON\n{'=' * 70}")
print(f"  Qwen3 (n=400):   car=98.2%  ctx=97.8%  joint=57.8% +/- 4.8pp")
print(f"  Hermes3 (n={n_tot}): car={combined_car*100:.1f}%  "
      f"ctx={combined_ctx*100:.1f}%  joint={combined_joint*100:.1f}% +/- {joint_hw*100:.1f}pp")

indep = combined_car ** 32 * 100
gap = combined_joint * 100 - indep
print(f"\n  Hermes3 independence: {combined_car*100:.1f}^32 = {indep:.1f}%")
print(f"  Actual: {combined_joint*100:.1f}% -> gap = {gap:+.1f}pp")
print(f"  (Qwen3 gap = +1.8pp)")

output = {
    "architecture": "Hermes3-3B",
    "trained_seed": 42,
    "training_time_n100": original,
    "round_999": result_999,
    "round_2024": result_2024,
    "combined": {
        "n_samples": n_tot,
        "carrier_slot": combined_car,
        "context_slot": combined_ctx,
        "all_joint": combined_joint,
        "joint_ci_pp_wilson": joint_hw * 100,
    },
    "independence_prediction": indep / 100,
    "independence_gap_pp": gap,
}
out_path = os.path.join(CARRIER_DIR, f"reeval_n{N_EVAL}x2.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}")
