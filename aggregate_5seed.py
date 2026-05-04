#!/usr/bin/env python3
"""Aggregate 5-seed sweep results: per-ratio mean + Wilson 95% CI from pooled n=1000."""
import json, math, os

BASE = "/workspace/FDM_IN_WEIGHTS/split_ratio_sweep_5seed"
SEEDS = [42, 123, 1337, 2024, 7777]

def wilson_hw(p, n, z=1.96):
    if n == 0: return 0.0
    return (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)

all_results = {}
for seed in SEEDS:
    path = f"{BASE}/seed_{seed}/sweep_results.json"
    if not os.path.exists(path):
        print(f"  [missing] seed={seed}")
        continue
    seed_data = json.load(open(path))
    for entry in seed_data:
        key = (entry['n_param'], entry['n_ctx'])
        all_results.setdefault(key, []).append(entry)

print(f"\n{'n_p':>4} {'n_c':>4} {'seeds':>6}  {'mean slot':>12} {'mean joint':>12}  {'pooled n':>10} {'pooled joint':>20}")
print("-" * 100)
table_rows = []
for (n_param, n_ctx) in sorted(all_results.keys()):
    seed_runs = all_results[(n_param, n_ctx)]
    n_seeds = len(seed_runs)
    
    slot_vals = [r['all_acc_slot'] for r in seed_runs]
    joint_vals = [r['all_acc_joint'] for r in seed_runs]
    
    n_per_seed = seed_runs[0]['n_samples']
    pooled_n = n_seeds * n_per_seed
    pooled_joint_correct = sum(int(round(r['all_acc_joint'] * n_per_seed)) for r in seed_runs)
    pooled_joint_p = pooled_joint_correct / pooled_n if pooled_n else 0
    pooled_joint_hw = wilson_hw(pooled_joint_p, pooled_n)
    
    pooled_slot_n = n_seeds * n_per_seed * 32
    pooled_slot_correct = sum(int(round(r['all_acc_slot'] * n_per_seed * 32)) for r in seed_runs)
    pooled_slot_p = pooled_slot_correct / pooled_slot_n if pooled_slot_n else 0
    pooled_slot_hw = wilson_hw(pooled_slot_p, pooled_slot_n)
    
    mean_slot = sum(slot_vals) / n_seeds
    mean_joint = sum(joint_vals) / n_seeds
    
    print(f"{n_param:>4} {n_ctx:>4} {n_seeds:>6}  {mean_slot*100:>10.2f}%  {mean_joint*100:>10.2f}%  {pooled_n:>10}  {pooled_joint_p*100:.2f} +/- {pooled_joint_hw*100:.2f}")
    
    table_rows.append({
        'n_param': n_param, 'n_ctx': n_ctx, 'n_seeds': n_seeds,
        'mean_slot': mean_slot, 'mean_joint': mean_joint,
        'per_seed_slot': slot_vals, 'per_seed_joint': joint_vals,
        'pooled_n': pooled_n,
        'pooled_slot': pooled_slot_p, 'pooled_slot_hw': pooled_slot_hw,
        'pooled_joint': pooled_joint_p, 'pooled_joint_hw': pooled_joint_hw,
    })

out = f"{BASE}/aggregated_results.json"
json.dump(table_rows, open(out, 'w'), indent=2)
print(f"\n  Saved to {out}")

print("\n=== Per-seed values for 12/20 (the original failure case) ===")
for entry in all_results.get((12, 20), []):
    print(f"  slot={entry['all_acc_slot']*100:.2f}%  joint={entry['all_acc_joint']*100:.2f}%")
