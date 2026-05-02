#!/usr/bin/env python3
"""
eval_split_source_n200.py

Sample-level joint accuracy evaluation for the Qwen3 split-source write head.
Reuses fdm_split_source_hybrid's pipeline (CrossAttentionWriteHead, encoder,
generate_with_hybrid) and extends evaluate_split_source to track per-sample
all-correct outcomes for both substrates and the full 32-channel join.

Outputs slot-level per-substrate accuracy (matches existing 99.06% / 100.0%)
plus sample-level joint metrics:
  - C_A joint:  fraction of samples where all 16 ch8-23 hit
  - C_B joint:  fraction of samples where all 16 ch24-39 hit
  - All joint:  fraction of samples where all 32 ch8-39 hit (the strict metric)

Wilson 95% CI half-margins reported on each.
"""

import os, sys, json, math, re
import torch
from tqdm import tqdm

WORK_DIR = "/workspace/FDM_IN_WEIGHTS"
WRITE_HEAD_DIR = os.path.join(WORK_DIR, "split_source_hybrid")
WRITE_HEAD_PT = os.path.join(WRITE_HEAD_DIR, "split_source_write_head.pt")
HOST_MODEL = os.path.join(WORK_DIR, "two_block_model")
N_EVAL = 200

sys.path.insert(0, WORK_DIR)

# Import training script as a module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fdm_ss", os.path.join(WORK_DIR, "fdm_split_source_hybrid.py")
)
fdm_ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdm_ss)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pull symbols
CrossAttentionWriteHead = fdm_ss.CrossAttentionWriteHead
make_encoder            = fdm_ss.make_encoder
random_memory           = fdm_ss.random_memory
generate_with_hybrid    = fdm_ss.generate_with_hybrid
BLOCK_A_CHANNELS        = fdm_ss.BLOCK_A_CHANNELS
BLOCK_B_CHANNELS        = fdm_ss.BLOCK_B_CHANNELS
ALL_CHANNELS            = fdm_ss.ALL_CHANNELS
CHANNEL_NAMES           = fdm_ss.CHANNEL_NAMES


def wilson_hw(k, n, z=1.96):
    """Wilson 95% CI half-margin (conservative side)."""
    if n == 0:
        return 0.0
    p = k / n
    return (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def evaluate_split_source_full(model, tokenizer, write_head, encoder, device,
                                n_eval=200, seed=12345):
    """Sample-level joint version of evaluate_split_source.

    For each sample:
      - check each of 32 channels independently (slot-level counts)
      - record per-sample flags: all_A_correct, all_B_correct, all_correct
    """
    write_head.eval()

    # Slot-level counts (same as original eval)
    a_correct = 0
    a_total = 0
    b_correct = 0
    b_total = 0

    # Sample-level joint counts (the new metric)
    n_sample_a_all = 0
    n_sample_b_all = 0
    n_sample_all_all = 0
    n_samples = 0

    import random
    random.seed(seed)
    torch.manual_seed(seed)

    for _ in tqdm(range(n_eval), desc=f"Eval n={n_eval}", leave=False):
        mem = random_memory()
        answer = generate_with_hybrid(
            model, tokenizer, write_head, mem, encoder, device
        )

        a_all = True
        b_all = True

        # Block A channels (parametric)
        for k in BLOCK_A_CHANNELS:
            pat = rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b"
            hit = re.search(pat, answer) is not None
            if hit:
                a_correct += 1
            else:
                a_all = False
            a_total += 1

        # Block B channels (context)
        for k in BLOCK_B_CHANNELS:
            pat = rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b"
            hit = re.search(pat, answer) is not None
            if hit:
                b_correct += 1
            else:
                b_all = False
            b_total += 1

        if a_all:
            n_sample_a_all += 1
        if b_all:
            n_sample_b_all += 1
        if a_all and b_all:
            n_sample_all_all += 1
        n_samples += 1

    # Compute slot-level
    a_slot = a_correct / a_total
    b_slot = b_correct / b_total

    # Compute sample-level joint
    a_joint = n_sample_a_all / n_samples
    b_joint = n_sample_b_all / n_samples
    all_joint = n_sample_all_all / n_samples

    # Wilson CIs
    a_slot_ci = wilson_hw(a_correct, a_total)
    b_slot_ci = wilson_hw(b_correct, b_total)
    a_joint_ci = wilson_hw(n_sample_a_all, n_samples)
    b_joint_ci = wilson_hw(n_sample_b_all, n_samples)
    all_joint_ci = wilson_hw(n_sample_all_all, n_samples)

    print()
    print("=" * 60)
    print(f"  EVAL RESULTS (n={n_samples} samples)")
    print("=" * 60)
    print(f"  C_A (ch8-23) slot:     {a_slot*100:6.2f}% +/- {a_slot_ci*100:.2f}pp"
          f"  ({a_correct}/{a_total})")
    print(f"  C_B (ch24-39) slot:    {b_slot*100:6.2f}% +/- {b_slot_ci*100:.2f}pp"
          f"  ({b_correct}/{b_total})")
    print()
    print(f"  C_A all-16 joint:      {a_joint*100:6.2f}% +/- {a_joint_ci*100:.2f}pp"
          f"  ({n_sample_a_all}/{n_samples})")
    print(f"  C_B all-16 joint:      {b_joint*100:6.2f}% +/- {b_joint_ci*100:.2f}pp"
          f"  ({n_sample_b_all}/{n_samples})")
    print(f"  All-32 sample-joint:   {all_joint*100:6.2f}% +/- {all_joint_ci*100:.2f}pp"
          f"  ({n_sample_all_all}/{n_samples})")
    print()
    print(f"  Independence prediction (slot^32):  "
          f"{(a_slot ** 16) * (b_slot ** 16) * 100:.2f}%")
    print()

    return {
        "n_samples": n_samples,
        "C_A_slot": a_slot,
        "C_A_slot_correct": a_correct,
        "C_A_slot_total": a_total,
        "C_A_slot_ci_pp": a_slot_ci * 100,
        "C_B_slot": b_slot,
        "C_B_slot_correct": b_correct,
        "C_B_slot_total": b_total,
        "C_B_slot_ci_pp": b_slot_ci * 100,
        "C_A_all16_joint": a_joint,
        "C_A_all16_joint_count": n_sample_a_all,
        "C_A_all16_joint_ci_pp": a_joint_ci * 100,
        "C_B_all16_joint": b_joint,
        "C_B_all16_joint_count": n_sample_b_all,
        "C_B_all16_joint_ci_pp": b_joint_ci * 100,
        "all32_sample_joint": all_joint,
        "all32_sample_joint_count": n_sample_all_all,
        "all32_sample_joint_ci_pp": all_joint_ci * 100,
        "independence_prediction": (a_slot ** 16) * (b_slot ** 16),
    }


def main():
    print("=" * 60)
    print("  SPLIT-SOURCE n=200 SAMPLE-LEVEL EVAL")
    print("  Qwen3-0.6B + trained write head, 16/16 partition")
    print("=" * 60)

    print(f"\nLoading host model from {HOST_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(HOST_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HOST_MODEL, dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    encoder = make_encoder(tokenizer)

    print(f"\nLoading write head from {WRITE_HEAD_PT}...")
    write_head = CrossAttentionWriteHead(
        n_channels=16, n_values=64, embed_dim=384,
        n_kv_positions=513, n_layers=28, kv_dim=128, n_kv_heads=8,
    ).to(device)
    state_dict = torch.load(WRITE_HEAD_PT, map_location=device)
    write_head.load_state_dict(state_dict, strict=True)
    write_head.eval()
    print(f"  Loaded ({sum(p.numel() for p in write_head.parameters())/1e6:.1f}M params)")

    # Run two seeds for robustness, like carrier_head_reeval.py does
    print("\n--- Round 1: seed=999 ---")
    result_999 = evaluate_split_source_full(
        model, tokenizer, write_head, encoder, device,
        n_eval=N_EVAL, seed=999
    )

    print("\n--- Round 2: seed=2024 ---")
    result_2024 = evaluate_split_source_full(
        model, tokenizer, write_head, encoder, device,
        n_eval=N_EVAL, seed=2024
    )

    # Combined summary
    n_total = result_999['n_samples'] + result_2024['n_samples']
    a_slot_combined = (result_999['C_A_slot_correct'] + result_2024['C_A_slot_correct']) / \
                      (result_999['C_A_slot_total'] + result_2024['C_A_slot_total'])
    b_slot_combined = (result_999['C_B_slot_correct'] + result_2024['C_B_slot_correct']) / \
                      (result_999['C_B_slot_total'] + result_2024['C_B_slot_total'])
    all_joint_combined = (result_999['all32_sample_joint_count'] +
                          result_2024['all32_sample_joint_count']) / n_total

    a_slot_ci_c = wilson_hw(
        result_999['C_A_slot_correct'] + result_2024['C_A_slot_correct'],
        result_999['C_A_slot_total'] + result_2024['C_A_slot_total']
    )
    b_slot_ci_c = wilson_hw(
        result_999['C_B_slot_correct'] + result_2024['C_B_slot_correct'],
        result_999['C_B_slot_total'] + result_2024['C_B_slot_total']
    )
    all_joint_ci_c = wilson_hw(
        result_999['all32_sample_joint_count'] + result_2024['all32_sample_joint_count'],
        n_total
    )

    print()
    print("=" * 60)
    print(f"  COMBINED (n={n_total})")
    print("=" * 60)
    print(f"  C_A slot:        {a_slot_combined*100:6.2f}% +/- {a_slot_ci_c*100:.2f}pp")
    print(f"  C_B slot:        {b_slot_combined*100:6.2f}% +/- {b_slot_ci_c*100:.2f}pp")
    print(f"  All-32 joint:    {all_joint_combined*100:6.2f}% +/- {all_joint_ci_c*100:.2f}pp")
    print()

    # Save
    out_path = os.path.join(WRITE_HEAD_DIR, f"eval_n{N_EVAL}x2.json")
    output = {
        "round_999": result_999,
        "round_2024": result_2024,
        "combined": {
            "n_samples": n_total,
            "C_A_slot": a_slot_combined,
            "C_A_slot_ci_pp": a_slot_ci_c * 100,
            "C_B_slot": b_slot_combined,
            "C_B_slot_ci_pp": b_slot_ci_c * 100,
            "all32_sample_joint": all_joint_combined,
            "all32_sample_joint_ci_pp": all_joint_ci_c * 100,
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
