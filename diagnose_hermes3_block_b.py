#!/usr/bin/env python3
"""
diagnose_hermes3_block_b.py (v2 - simplified)

The Block A/B "split" is just a labeling convention -- both blocks during
two-block training contain the SAME full 40-channel FDM signal. So the
diagnostic is simpler than I first thought.

Three tests, all using the same memory and same FDM signal text:

TEST 1: Two-block training format (host's training distribution)
  [MEMORY]BLOCK_A <fdm>[/MEMORY][MEMORY]BLOCK_B <fdm>[/MEMORY]Q...A:
  Expected: ~100% on all 32 channels (this is what host saw)

TEST 2: Block-B-only format (split-source eval format minus parametric KV)
  [MEMORY]BLOCK_B <fdm>[/MEMORY]Q...A:
  Expected: should be ~100% if host learned per-block decoding
            ~30% if host needs both BLOCK_A and BLOCK_B markers

TEST 3: Block-A-only format
  [MEMORY]BLOCK_A <fdm>[/MEMORY]Q...A:
  Expected: similar to TEST 2 -- shows symmetry

If TEST 1 = 100% and TEST 2 < 100%: the split-source format is incompatible
with the host's training distribution -> need [MEMORY]BLOCK_A scaffold.

If TEST 2 = 100%: split-source's parametric KV is interfering -> debug
the write head outputs or position_ids.
"""

import sys, os, json, re
import torch
sys.path.insert(0, '/workspace/FDM_IN_WEIGHTS')

import importlib.util
spec = importlib.util.spec_from_file_location(
    'fdm_ss_h', '/workspace/FDM_IN_WEIGHTS/fdm_split_source_hybrid_hermes3.py'
)
fdm_ss_h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdm_ss_h)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
HOST = "/workspace/FDM_IN_WEIGHTS/two_block_model_hermes3"
N_TEST = 20

random_memory = fdm_ss_h.random_memory
make_encoder = fdm_ss_h.make_encoder
BLOCK_A_CHANNELS = fdm_ss_h.BLOCK_A_CHANNELS
BLOCK_B_CHANNELS = fdm_ss_h.BLOCK_B_CHANNELS
ALL_CHANNELS = fdm_ss_h.ALL_CHANNELS
CHANNEL_NAMES = fdm_ss_h.CHANNEL_NAMES


def generate_simple(model, tokenizer, prompt, max_new=350):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def score_answer(answer, memory, channels):
    correct = 0
    total = 0
    for k in channels:
        pat = rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(memory[k])}\b"
        if re.search(pat, answer):
            correct += 1
        total += 1
    return correct, total


def build_prompt(memory, encoder, mode):
    """mode in {'two_block', 'b_only', 'a_only'}"""
    fdm_text, _ = encoder.encode_memory(memory)  # full 40-channel signal
    ch_names = [CHANNEL_NAMES[k] for k in ALL_CHANNELS]
    question = f"Report values for: {', '.join(ch_names)}."
    if mode == 'two_block':
        return (f"[MEMORY]BLOCK_A {fdm_text}[/MEMORY]"
                f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]\n"
                f"Question: {question}\nAnswer:")
    elif mode == 'b_only':
        return (f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]\n"
                f"Question: {question}\nAnswer:")
    elif mode == 'a_only':
        return (f"[MEMORY]BLOCK_A {fdm_text}[/MEMORY]\n"
                f"Question: {question}\nAnswer:")


def run_test(name, mode, model, tokenizer, encoder, n_test=N_TEST):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    a_correct = a_total = 0
    b_correct = b_total = 0

    for i in range(n_test):
        memory = random_memory()
        prompt = build_prompt(memory, encoder, mode)
        answer = generate_simple(model, tokenizer, prompt)

        ac, at = score_answer(answer, memory, BLOCK_A_CHANNELS)
        bc, bt = score_answer(answer, memory, BLOCK_B_CHANNELS)
        a_correct += ac
        a_total += at
        b_correct += bc
        b_total += bt

        if i == 0:
            print(f"  Sample 1 prompt[0:200]: {prompt[:200]}...")
            print(f"  Sample 1 answer[0:200]: {answer[:200]}")

    a_acc = a_correct / a_total
    b_acc = b_correct / b_total
    print(f"\n  Block A (ch8-23):  {a_acc*100:.1f}% ({a_correct}/{a_total})")
    print(f"  Block B (ch24-39): {b_acc*100:.1f}% ({b_correct}/{b_total})")
    return a_acc, b_acc


def main():
    print("Loading Hermes3 host (no parametric KV in this test)...")
    tokenizer = AutoTokenizer.from_pretrained(HOST, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HOST, dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()
    encoder = make_encoder(tokenizer)
    print(f"  Loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    results = {}
    a, b = run_test("TEST 1: Two-block format (host's training format)",
                    'two_block', model, tokenizer, encoder)
    results['two_block'] = {'A': a, 'B': b}

    a, b = run_test("TEST 2: Block-B-only format (split-source eval, no parametric)",
                    'b_only', model, tokenizer, encoder)
    results['b_only'] = {'A': a, 'B': b}

    a, b = run_test("TEST 3: Block-A-only format (symmetry check)",
                    'a_only', model, tokenizer, encoder)
    results['a_only'] = {'A': a, 'B': b}

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  TEST 1 (two-block training format): A={results['two_block']['A']*100:.1f}%, B={results['two_block']['B']*100:.1f}%")
    print(f"  TEST 2 (B-only):                    A={results['b_only']['A']*100:.1f}%, B={results['b_only']['B']*100:.1f}%")
    print(f"  TEST 3 (A-only):                    A={results['a_only']['A']*100:.1f}%, B={results['a_only']['B']*100:.1f}%")
    print()
    print("DIAGNOSIS:")
    if results['two_block']['A'] > 0.85 and results['two_block']['B'] > 0.85:
        print("  TEST 1 PASSES: Host is healthy in training format.")
        if results['b_only']['B'] < 0.5:
            print("  TEST 2 FAILS: Host needs BOTH [MEMORY]BLOCK_A and [MEMORY]BLOCK_B markers.")
            print("  -> ROOT CAUSE: split-source eval prompt format mismatch.")
            print("  -> FIX: Add [MEMORY]BLOCK_A [/MEMORY] placeholder to split-source prompt.")
            print("     The parametric KV is supposed to substitute for it, but the host")
            print("     also needs the textual marker.")
        else:
            print("  TEST 2 PASSES: Host reads from B-only format too.")
            print("  -> ROOT CAUSE: parametric KV at positions 0-512 is interfering.")
            print("  -> FIX: Investigate write head output magnitudes, or position_ids alignment.")
    else:
        print("  TEST 1 FAILS: Host is broken even in its training format.")
        print("  -> Re-investigate two_block_model_hermes3 training quality.")

    with open('/workspace/FDM_IN_WEIGHTS/diagnose_hermes3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to diagnose_hermes3_results.json")


if __name__ == "__main__":
    main()
