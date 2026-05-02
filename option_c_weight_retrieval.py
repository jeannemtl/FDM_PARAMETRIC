#!/usr/bin/env python3
"""
Option C: Weight-based retrieval fine-tuning.
=============================================

Tests whether the host transformer can learn to retrieve facts from its
weights (no FDM signal, no KV injection) for a single fixed memory M,
when fine-tuned on (question, answer) pairs for that memory.

Saves the fine-tuned model as a SEPARATE artifact at the specified output
path. The base model is never modified on disk.

Usage:
    python option_c_weight_retrieval.py \\
        --base_model prompterminal/fdm-40ch-two-block-qwen3 \\
        --output_model_dir /workspace/FDM_IN_WEIGHTS/two_block_weight_retrieval \\
        --n_steps 2000 --lr 5e-5
"""

import os, sys, json, random, time, argparse, shutil
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/root/FDM_IN_WEIGHTS/scripts')
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]
QUERY_CHANNELS = list(range(8, 40))


TRAIN_TEMPLATES = [
    "Question: What is {ch}? Answer:",
    "Question: Report {ch}. Answer:",
    "Question: Tell me {ch}. Answer:",
    "Question: Current {ch}? Answer:",
    "Question: Status of {ch}? Answer:",
]
EVAL_HELDOUT_TEMPLATES = [
    "Question: What's the {ch}? Answer:",
    "Question: Provide {ch} value. Answer:",
]


def make_fixed_memory(seed=42):
    rng = random.Random(seed)
    M = {}
    for ch in range(NUM_CHANNELS):
        M[ch] = rng.choice(MEMORY_SCHEMAS[ch][1])
    return M


def make_pairs(M, templates):
    pairs = []
    for ch in QUERY_CHANNELS:
        for tpl in templates:
            prompt = tpl.format(ch=CHANNEL_NAMES[ch])
            answer = f" {CHANNEL_NAMES[ch]}={M[ch]}."
            pairs.append((prompt, answer, CHANNEL_NAMES[ch], M[ch]))
    return pairs


def train_step(model, tokenizer, prompt, answer, optimizer, device):
    full_text = prompt + answer
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    prefix_len = len(full_ids) - len(answer_ids)

    input_tensor = torch.tensor([full_ids], device=device)
    labels = [-100] * prefix_len + full_ids[prefix_len:]
    label_tensor = torch.tensor([labels], device=device)

    outputs = model(input_ids=input_tensor, labels=label_tensor)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train_loop(model, tokenizer, pairs, n_steps, lr, device, log_every=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr/20)
    model.train()
    losses = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        prompt, answer, _, _ = random.choice(pairs)
        loss = train_step(model, tokenizer, prompt, answer, optimizer, device)
        losses.append(loss)
        scheduler.step()
        if step % log_every == 0:
            avg = np.mean(losses[-log_every:])
            print(f"    Step {step:5d}/{n_steps} | loss {avg:.4f} | {time.time()-t0:.0f}s")
    return losses


@torch.no_grad()
def eval_pairs(model, tokenizer, pairs, device, max_new_tokens=20):
    model.eval()
    correct = 0
    total = 0
    examples = []
    for prompt, _, ch_name, correct_value in pairs:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        target = f"{ch_name}={correct_value}"
        is_correct = target in response
        if is_correct:
            correct += 1
        total += 1
        if len(examples) < 5:
            examples.append({
                "prompt": prompt, "expected": target,
                "response": response[:120], "correct": is_correct,
            })
    return correct, total, examples


@torch.no_grad()
def eval_fdm(model, tokenizer, encoder, M, device, n_eval=20):
    model.eval()
    fdm_text, _ = encoder.encode_memory(M)
    correct = 0
    total = 0
    test_channels = random.sample(QUERY_CHANNELS, 5)
    for _ in range(n_eval):
        ch = random.choice(test_channels)
        ch_name = CHANNEL_NAMES[ch]
        expected = M[ch]
        prompt = (f"[MEMORY]{fdm_text}[/MEMORY]\n"
                  f"Question: What is {ch_name}? Answer:")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        if f"{ch_name}={expected}" in response:
            correct += 1
        total += 1
    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",
                        default="prompterminal/fdm-40ch-two-block-qwen3",
                        help="HF repo or local path. NEVER modified on disk.")
    parser.add_argument("--output_model_dir", required=True,
                        help="Where to save the fine-tuned model. "
                             "Must NOT match base_model if base_model is a local path.")
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_overwrite", action="store_true",
                        help="Allow overwriting an existing output directory.")
    args = parser.parse_args()

    # Safety: never overwrite the base model
    out_real = os.path.realpath(args.output_model_dir)
    if os.path.exists(args.base_model):
        base_real = os.path.realpath(args.base_model)
        if base_real == out_real:
            raise ValueError(
                f"Refusing to run: --output_model_dir resolves to the same path as "
                f"--base_model ({base_real}). Pick a different output directory."
            )

    if os.path.exists(args.output_model_dir):
        if not args.allow_overwrite:
            raise ValueError(
                f"Output directory exists: {args.output_model_dir}. "
                f"Pass --allow_overwrite to replace its contents, or pick a new path."
            )
        print(f"[!] Overwriting existing directory: {args.output_model_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("OPTION C: WEIGHT-BASED RETRIEVAL FINE-TUNING")
    print("=" * 70)
    print(f"Base model:         {args.base_model}")
    print(f"Output model dir:   {args.output_model_dir}")
    print(f"Steps:              {args.n_steps}")
    print(f"Learning rate:      {args.lr}")

    print(f"\nLoading model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, trust_remote_code=True
    ).to(device)
    print(f"  Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    encoder = TurboFDMSignalEncoder(
        vocab_size=151936, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    M = make_fixed_memory(seed=42)
    print(f"\nFixed memory M (channels 8-15 sample):")
    for ch in range(8, 16):
        print(f"  ch{ch:2d} {CHANNEL_NAMES[ch]:>10s} = {M[ch]}")

    training_pairs = make_pairs(M, TRAIN_TEMPLATES)
    heldout_pairs = make_pairs(M, EVAL_HELDOUT_TEMPLATES)
    print(f"\n{len(training_pairs)} training pairs, {len(heldout_pairs)} held-out paraphrases")

    print("\n" + "-" * 70)
    print("Phase 0: Baseline eval (before fine-tuning)")
    print("-" * 70)
    base_ho, base_ho_total, _ = eval_pairs(model, tokenizer, heldout_pairs, device)
    base_fdm, base_fdm_total = eval_fdm(model, tokenizer, encoder, M, device, 20)
    print(f"  Held-out paraphrase: {base_ho}/{base_ho_total} = "
          f"{100*base_ho/base_ho_total:.1f}% (expected near 0%)")
    print(f"  FDM signal:          {base_fdm}/{base_fdm_total} = "
          f"{100*base_fdm/base_fdm_total:.1f}% (expected high)")

    print("\n" + "-" * 70)
    print(f"Phase 1: Fine-tuning {args.n_steps} steps at lr={args.lr}")
    print("-" * 70)
    losses = train_loop(model, tokenizer, training_pairs, args.n_steps, args.lr, device)

    print("\n" + "-" * 70)
    print("Phase 2: Held-out paraphrase eval")
    print("-" * 70)
    ho_correct, ho_total, ho_ex = eval_pairs(model, tokenizer, heldout_pairs, device)
    print(f"  Held-out: {ho_correct}/{ho_total} = {100*ho_correct/ho_total:.1f}%")
    for ex in ho_ex[:3]:
        mark = "OK" if ex["correct"] else "X "
        print(f"    [{mark}] {ex['prompt']}")
        print(f"         expected: {ex['expected']}")
        print(f"         got:      {ex['response']}")

    print("\n" + "-" * 70)
    print("Phase 3: Training-set sanity (memorization check)")
    print("-" * 70)
    tr_correct, tr_total, _ = eval_pairs(model, tokenizer, training_pairs[:64], device)
    print(f"  Training subset: {tr_correct}/{tr_total} = "
          f"{100*tr_correct/tr_total:.1f}%")

    print("\n" + "-" * 70)
    print("Phase 4: FDM-signal capability check (catastrophic forgetting)")
    print("-" * 70)
    fdm_correct, fdm_total = eval_fdm(model, tokenizer, encoder, M, device, 20)
    print(f"  FDM signal: {fdm_correct}/{fdm_total} = "
          f"{100*fdm_correct/fdm_total:.1f}% (was {100*base_fdm/base_fdm_total:.1f}%)")

    print("\n" + "-" * 70)
    print("Phase 5: Saving fine-tuned model")
    print("-" * 70)
    if os.path.exists(args.output_model_dir):
        shutil.rmtree(args.output_model_dir)
    os.makedirs(args.output_model_dir, exist_ok=True)
    model.save_pretrained(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)
    print(f"  Saved to {args.output_model_dir}")

    findings = {
        "config": {
            "base_model": args.base_model,
            "output_model_dir": args.output_model_dir,
            "n_steps": args.n_steps,
            "lr": args.lr,
            "seed": args.seed,
        },
        "fixed_memory": {str(k): v for k, v in M.items()},
        "baseline_heldout_accuracy": base_ho / base_ho_total,
        "baseline_fdm_accuracy": base_fdm / base_fdm_total,
        "training_accuracy": tr_correct / tr_total,
        "heldout_accuracy": ho_correct / ho_total,
        "fdm_accuracy_after": fdm_correct / fdm_total,
        "training_loss_final": float(np.mean(losses[-100:])),
        "heldout_examples": ho_ex,
    }
    with open(os.path.join(args.output_model_dir, "results.json"), "w") as f:
        json.dump(findings, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Training-set:        {100*tr_correct/tr_total:.1f}%")
    print(f"  Held-out paraphrase: {100*ho_correct/ho_total:.1f}%")
    print(f"  FDM signal: before {100*base_fdm/base_fdm_total:.1f}% -> "
          f"after {100*fdm_correct/fdm_total:.1f}%")
    print()

    ho_r = ho_correct / ho_total
    fdm_r = fdm_correct / fdm_total
    fdm_base_r = base_fdm / base_fdm_total
    fdm_drop = fdm_base_r - fdm_r

    if ho_r > 0.9 and fdm_drop < 0.1:
        verdict = "GENUINE WEIGHT-BASED RETRIEVAL"
        msg = "Generalized to held-out phrasings AND retained FDM capability."
    elif ho_r > 0.9 and fdm_drop >= 0.1:
        verdict = "WEIGHT RETRIEVAL WITH CATASTROPHIC FORGETTING"
        msg = "Learned weight retrieval but degraded FDM reader."
    elif tr_correct / tr_total > 0.9 and ho_r < 0.5:
        verdict = "TRIVIAL MEMORIZATION"
        msg = "Memorized training inputs, did not generalize to held-out paraphrases."
    elif tr_correct / tr_total < 0.5:
        verdict = "DID NOT CONVERGE"
        msg = "Fine-tuning did not learn even the training pairs."
    else:
        verdict = "MIXED"
        msg = "Partial success on both axes; needs closer interpretation."

    print(f"  VERDICT: {verdict}")
    print(f"  -> {msg}")
    print()
    print(f"  Model artifact:  {args.output_model_dir}")
    print(f"  Findings JSON:   {args.output_model_dir}/results.json")
    print()
    print("Reload with:")
    print(f"  AutoModelForCausalLM.from_pretrained('{args.output_model_dir}')")


if __name__ == "__main__":
    main()
