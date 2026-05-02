#!/usr/bin/env python3
"""
eval_split_source_n200_hermes3.py - n=200x2 sample-level joint eval for Hermes3.
Mirrors eval_split_source_n200.py methodology used for Qwen3.
"""
import os, sys, json, math, re
import torch
from tqdm import tqdm

WORK_DIR = "/workspace/FDM_IN_WEIGHTS"
WRITE_HEAD_DIR = os.path.join(WORK_DIR, "split_source_hermes3_correct")
WRITE_HEAD_PT = os.path.join(WRITE_HEAD_DIR, "split_source_write_head.pt")
HOST_MODEL = os.path.join(WORK_DIR, "two_block_hermes3_correct")
N_EVAL = 200

sys.path.insert(0, WORK_DIR)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "fdm_ss_h", os.path.join(WORK_DIR, "fdm_split_source_hybrid_hermes3.py")
)
fdm_ss_h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdm_ss_h)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

CrossAttentionWriteHead = fdm_ss_h.CrossAttentionWriteHead
make_encoder            = fdm_ss_h.make_encoder
random_memory           = fdm_ss_h.random_memory
generate_with_hybrid    = fdm_ss_h.generate_with_hybrid
BLOCK_A_CHANNELS        = fdm_ss_h.BLOCK_A_CHANNELS
BLOCK_B_CHANNELS        = fdm_ss_h.BLOCK_B_CHANNELS
ALL_CHANNELS            = fdm_ss_h.ALL_CHANNELS
CHANNEL_NAMES           = fdm_ss_h.CHANNEL_NAMES


def wilson_hw(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def evaluate(model, tokenizer, write_head, encoder, device, n_eval, seed):
    write_head.eval()
    a_correct = a_total = 0
    b_correct = b_total = 0
    n_a_all = n_b_all = n_all = 0
    
    import random
    random.seed(seed); torch.manual_seed(seed)
    
    for _ in tqdm(range(n_eval), desc=f"seed={seed}", leave=False):
        mem = random_memory()
        answer = generate_with_hybrid(model, tokenizer, write_head, mem, encoder, device)
        a_all = b_all = True
        for k in BLOCK_A_CHANNELS:
            pat = rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b"
            if re.search(pat, answer): a_correct += 1
            else: a_all = False
            a_total += 1
        for k in BLOCK_B_CHANNELS:
            pat = rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b"
            if re.search(pat, answer): b_correct += 1
            else: b_all = False
            b_total += 1
        if a_all: n_a_all += 1
        if b_all: n_b_all += 1
        if a_all and b_all: n_all += 1
    
    return {
        "n_samples": n_eval,
        "a_correct": a_correct, "a_total": a_total,
        "b_correct": b_correct, "b_total": b_total,
        "n_a_all": n_a_all, "n_b_all": n_b_all, "n_all": n_all,
    }


def main():
    print(f"Loading host from {HOST_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(HOST_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HOST_MODEL, dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    encoder = make_encoder(tokenizer)

    print(f"Loading write head from {WRITE_HEAD_PT}...")
    write_head = CrossAttentionWriteHead(
        n_channels=16, n_values=64, embed_dim=384,
        n_kv_positions=513, n_layers=28, kv_dim=128, n_kv_heads=8,
    ).to(device)
    write_head.load_state_dict(torch.load(WRITE_HEAD_PT, map_location=device), strict=True)
    write_head.eval()

    print("\n--- Round 1: seed=999 ---")
    r1 = evaluate(model, tokenizer, write_head, encoder, device, N_EVAL, 999)
    print(f"  C_A slot: {100*r1['a_correct']/r1['a_total']:.2f}%  ({r1['a_correct']}/{r1['a_total']})")
    print(f"  C_B slot: {100*r1['b_correct']/r1['b_total']:.2f}%  ({r1['b_correct']}/{r1['b_total']})")
    print(f"  All-32 sample-joint: {100*r1['n_all']/r1['n_samples']:.2f}%  ({r1['n_all']}/{r1['n_samples']})")
    
    print("\n--- Round 2: seed=2024 ---")
    r2 = evaluate(model, tokenizer, write_head, encoder, device, N_EVAL, 2024)
    print(f"  C_A slot: {100*r2['a_correct']/r2['a_total']:.2f}%  ({r2['a_correct']}/{r2['a_total']})")
    print(f"  C_B slot: {100*r2['b_correct']/r2['b_total']:.2f}%  ({r2['b_correct']}/{r2['b_total']})")
    print(f"  All-32 sample-joint: {100*r2['n_all']/r2['n_samples']:.2f}%  ({r2['n_all']}/{r2['n_samples']})")
    
    n = r1['n_samples'] + r2['n_samples']
    a_c = r1['a_correct'] + r2['a_correct']
    a_t = r1['a_total'] + r2['a_total']
    b_c = r1['b_correct'] + r2['b_correct']
    b_t = r1['b_total'] + r2['b_total']
    n_all = r1['n_all'] + r2['n_all']
    
    print(f"\n=== COMBINED (n={n}) ===")
    print(f"  C_A slot:        {100*a_c/a_t:.2f}% +/- {100*wilson_hw(a_c, a_t):.2f}pp")
    print(f"  C_B slot:        {100*b_c/b_t:.2f}% +/- {100*wilson_hw(b_c, b_t):.2f}pp")
    print(f"  All-32 joint:    {100*n_all/n:.2f}% +/- {100*wilson_hw(n_all, n):.2f}pp")
    
    out = {
        "round_999": r1, "round_2024": r2,
        "combined": {
            "n_samples": n,
            "C_A_slot": a_c/a_t, "C_A_slot_ci_pp": 100*wilson_hw(a_c, a_t),
            "C_B_slot": b_c/b_t, "C_B_slot_ci_pp": 100*wilson_hw(b_c, b_t),
            "all32_joint": n_all/n, "all32_joint_ci_pp": 100*wilson_hw(n_all, n),
        }
    }
    out_path = os.path.join(WRITE_HEAD_DIR, f"eval_n{N_EVAL}x2_hermes3.json")
    with open(out_path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
