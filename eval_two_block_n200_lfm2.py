#!/usr/bin/env python3
"""
eval_two_block_n200_lfm2.py — n=200x2 sample-level eval of LFM2.5 on the
two-block in-context format. Mirrors Hermes3 split-source methodology but
without parametric KV (LFM2.5 conv layers don't support that).

Both FDM blocks delivered as input tokens. Score channels 8-39 (32 channels)
on regex `\\bNAME=VALUE\\b` matching the model output.
"""
import os, sys, json, math, re
import torch
from tqdm import tqdm

WORK_DIR  = "/workspace/FDM_IN_WEIGHTS"
HOST      = os.path.join(WORK_DIR, "two_block_model_lfm2")
N_EVAL    = 200

sys.path.insert(0, WORK_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tb", os.path.join(WORK_DIR, "fdm_two_block_training_lfm2.py")
)
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
ALL_CH = list(range(8, 40))


def wilson_hw(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def evaluate(model, tokenizer, encoder, device, n_eval, seed):
    """Per-sample two-block eval. Returns dict with both slot and joint counts."""
    import random
    random.seed(seed); torch.manual_seed(seed)

    slot_correct = 0
    slot_total   = 0
    n_sample_all = 0   # samples where ALL 32 channels correct (sample-level joint)

    for _ in tqdm(range(n_eval), desc=f"seed={seed}", leave=False):
        mem = tb.random_memory()
        fdm_text_a, _ = encoder.encode_memory(mem)
        fdm_text_b, _ = encoder.encode_memory(mem)

        ch_names = [tb.CHANNEL_NAMES[k] for k in ALL_CH]
        question = f"Report values for: {', '.join(ch_names)}."
        prompt = (f"[MEMORY]BLOCK_A {fdm_text_a}[/MEMORY]"
                  f"[MEMORY]BLOCK_B {fdm_text_b}[/MEMORY]"
                  f"\nQuestion: {question}\nAnswer:")

        ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=350, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        sample_correct = 0
        for k in ALL_CH:
            pat = rf"\b{re.escape(tb.CHANNEL_NAMES[k])}={re.escape(mem[k])}\b"
            if re.search(pat, text):
                sample_correct += 1
                slot_correct += 1
            slot_total += 1
        if sample_correct == 32:
            n_sample_all += 1

    return {
        "n_samples":    n_eval,
        "slot_correct": slot_correct,
        "slot_total":   slot_total,
        "n_sample_all": n_sample_all,
    }


def main():
    print(f"Loading host from {HOST}...")
    tokenizer = AutoTokenizer.from_pretrained(HOST, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HOST, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    encoder = tb.make_encoder(tokenizer)

    print("\n--- Round 1: seed=999 ---")
    r1 = evaluate(model, tokenizer, encoder, device, N_EVAL, 999)
    print(f"  Slot:  {100*r1['slot_correct']/r1['slot_total']:.2f}%  ({r1['slot_correct']}/{r1['slot_total']})")
    print(f"  Joint: {100*r1['n_sample_all']/r1['n_samples']:.2f}%  ({r1['n_sample_all']}/{r1['n_samples']})")

    print("\n--- Round 2: seed=2024 ---")
    r2 = evaluate(model, tokenizer, encoder, device, N_EVAL, 2024)
    print(f"  Slot:  {100*r2['slot_correct']/r2['slot_total']:.2f}%  ({r2['slot_correct']}/{r2['slot_total']})")
    print(f"  Joint: {100*r2['n_sample_all']/r2['n_samples']:.2f}%  ({r2['n_sample_all']}/{r2['n_samples']})")

    n   = r1['n_samples']     + r2['n_samples']
    sc  = r1['slot_correct']  + r2['slot_correct']
    st  = r1['slot_total']    + r2['slot_total']
    nj  = r1['n_sample_all']  + r2['n_sample_all']

    print(f"\n=== COMBINED (n={n}) ===")
    print(f"  Slot:   {100*sc/st:.2f}% +/- {100*wilson_hw(sc, st):.2f}pp  ({sc}/{st})")
    print(f"  Joint:  {100*nj/n:.2f}% +/- {100*wilson_hw(nj, n):.2f}pp  ({nj}/{n})")

    out = {
        "round_999": r1, "round_2024": r2,
        "combined": {
            "n_samples":     n,
            "slot":          sc/st,
            "slot_ci_pp":    100*wilson_hw(sc, st),
            "joint":         nj/n,
            "joint_ci_pp":   100*wilson_hw(nj, n),
        }
    }
    out_path = os.path.join(HOST, f"eval_two_block_n{N_EVAL}x2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
