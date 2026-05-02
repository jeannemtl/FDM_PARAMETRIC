#!/usr/bin/env python3
"""
Option C v2: Weight-based retrieval fine-tuning, with FIXED FDM eval.

KEY FIX from v1: The FDM-capability check now uses a DIFFERENT memory
than the one used for fine-tuning. Otherwise the eval cannot distinguish
"model reads FDM signal" from "model returns the baked-in default value
that happens to match."

Defaults are also tuned for the catastrophic-forgetting-avoiding regime:
  --lr 1e-5      (10x smaller than v1's 5e-5)
  --n_steps 500  (4x fewer than v1's 2000)

Saves the fine-tuned model as a SEPARATE artifact at the specified path.

Usage:
    python option_c_v2_weight_retrieval.py \\
        --base_model prompterminal/fdm-40ch-two-block-qwen3 \\
        --output_model_dir /workspace/FDM_IN_WEIGHTS/two_block_qwen3_low_lr \\
        --n_steps 500 --lr 1e-5
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


def make_memory(seed):
    rng = random.Random(seed)
    return {ch: rng.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}


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


def train_loop(model, tokenizer, pairs, n_steps, lr, device,
               eval_callback=None, log_every=50):
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
            elapsed = time.time() - t0
            print(f"    Step {step:4d}/{n_steps} | loss {avg:.4f} | {elapsed:.0f}s")
            if eval_callback is not None and step % (n_steps // 5) == 0:
                eval_callback(step)
                model.train()
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
def eval_fdm(model, tokenizer, encoder, M_test, device, n_eval=20):
    """
    KEY FIX: M_test is the memory used for the FDM signal AND for the
    expected answer. This M_test must be DIFFERENT from the memory used
    for fine-tuning, otherwise the eval cannot distinguish signal-driven
    retrieval from baked-in default behavior.
    """
    model.eval()
    fdm_text, _ = encoder.encode_memory(M_test)
    correct = 0
    total = 0
    examples = []
    test_channels = random.sample(QUERY_CHANNELS, 5)
    for trial in range(n_eval):
        ch = random.choice(test_channels)
        ch_name = CHANNEL_NAMES[ch]
        expected = M_test[ch]
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
        if len(examples) < 3:
            examples.append({
                "channel": ch_name,
                "expected": f"{ch_name}={expected}",
                "response": response[:100],
                "correct": f"{ch_name}={expected}" in response,
            })
    return correct, total, examples


@torch.no_grad()
def eval_fdm_returns_finetune_memory(model, tokenizer, encoder, M_test,
                                       M_finetune, device, n_eval=20):
    """
    Diagnostic: when given FDM signal for M_test, does the model return
    M_finetune's values instead? This catches the false-positive case
    where the model has memorized M_finetune and ignores any FDM signal.
    """
    model.eval()
    fdm_text, _ = encoder.encode_memory(M_test)
    returns_finetune = 0
    total = 0
    test_channels = random.sample(QUERY_CHANNELS, 5)
    for trial in range(n_eval):
        ch = random.choice(test_channels)
        ch_name = CHANNEL_NAMES[ch]
        # Only count as "returns_finetune" if the two memories differ on this channel
        if M_test[ch] == M_finetune[ch]:
            continue
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
        # Did model return M_finetune's value (the "wrong" answer when signal is M_test)?
        if f"{ch_name}={M_finetune[ch]}" in response:
            returns_finetune += 1
        total += 1
    return returns_finetune, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",
                        default="prompterminal/fdm-40ch-two-block-qwen3")
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--n_steps", type=int, default=500,
                        help="Default 500: small to avoid catastrophic forgetting.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Default 1e-5: small to preserve FDM reader.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    out_real = os.path.realpath(args.output_model_dir)
    if os.path.exists(args.base_model):
        if os.path.realpath(args.base_model) == out_real:
            raise ValueError("output_model_dir must differ from base_model.")
    if os.path.exists(args.output_model_dir) and not args.allow_overwrite:
        raise ValueError(
            f"Output directory exists: {args.output_model_dir}. "
            f"Pass --allow_overwrite to replace, or pick a new path."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("OPTION C v2: WEIGHT-BASED RETRIEVAL FINE-TUNING (low LR, FDM eval fixed)")
    print("=" * 70)
    print(f"Base model:        {args.base_model}")
    print(f"Output:            {args.output_model_dir}")
    print(f"Steps / LR:        {args.n_steps} / {args.lr}")

    print(f"\nLoading {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, trust_remote_code=True
    ).to(device)
    encoder = TurboFDMSignalEncoder(
        vocab_size=151936, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    # Two distinct memories
    M_finetune = make_memory(seed=42)   # what we fine-tune the model to produce
    M_eval = make_memory(seed=999)      # what the FDM eval uses (DIFFERENT from M_finetune)

    print(f"\nM_finetune (training target):")
    print(f"  TEAM={M_finetune[8]}, REGION={M_finetune[9]}, "
          f"STATUS={M_finetune[3]}, PHASE={M_finetune[10]}")
    print(f"M_eval (FDM-signal test memory, NOT seen during fine-tuning):")
    print(f"  TEAM={M_eval[8]}, REGION={M_eval[9]}, "
          f"STATUS={M_eval[3]}, PHASE={M_eval[10]}")

    # Confirm the two memories actually differ on enough channels
    differing_channels = [ch for ch in QUERY_CHANNELS if M_finetune[ch] != M_eval[ch]]
    print(f"\n  Channels where M_finetune differs from M_eval: "
          f"{len(differing_channels)}/{len(QUERY_CHANNELS)}")
    if len(differing_channels) < 10:
        print("  WARNING: too few differing channels; results may not be conclusive")

    training_pairs = make_pairs(M_finetune, TRAIN_TEMPLATES)
    heldout_pairs = make_pairs(M_finetune, EVAL_HELDOUT_TEMPLATES)

    # ----- Phase 0 -----
    print("\n" + "-" * 70)
    print("Phase 0: Baseline (before fine-tuning)")
    print("-" * 70)
    base_ho, base_ho_total, _ = eval_pairs(model, tokenizer, heldout_pairs, device)
    base_fdm, base_fdm_total, base_fdm_ex = eval_fdm(
        model, tokenizer, encoder, M_eval, device, 20
    )
    print(f"  Held-out paraphrase (M_finetune): {100*base_ho/base_ho_total:.1f}% (expected ~0%)")
    print(f"  FDM signal for M_eval:           {100*base_fdm/base_fdm_total:.1f}% (expected high)")
    print(f"  Sample baseline FDM responses:")
    for ex in base_fdm_ex:
        mark = "OK" if ex["correct"] else "X "
        print(f"    [{mark}] {ex['expected']:>20s} <- {ex['response']}")

    if base_fdm / base_fdm_total < 0.5:
        print("\n  WARNING: baseline FDM accuracy is unexpectedly low.")
        print("  This may indicate an issue with the encoder or M_eval choice.")

    # ----- Phase 1 -----
    print("\n" + "-" * 70)
    print(f"Phase 1: Fine-tuning {args.n_steps} steps at lr={args.lr}")
    print("-" * 70)
    losses = train_loop(model, tokenizer, training_pairs, args.n_steps, args.lr, device)

    # ----- Phase 2 -----
    print("\n" + "-" * 70)
    print("Phase 2: Held-out paraphrase eval (M_finetune)")
    print("-" * 70)
    ho_correct, ho_total, ho_ex = eval_pairs(model, tokenizer, heldout_pairs, device)
    print(f"  Held-out: {100*ho_correct/ho_total:.1f}%")

    # ----- Phase 3 -----
    print("\n" + "-" * 70)
    print("Phase 3: Training-set sanity")
    print("-" * 70)
    tr_correct, tr_total, _ = eval_pairs(model, tokenizer, training_pairs[:64], device)
    print(f"  Training subset: {100*tr_correct/tr_total:.1f}%")

    # ----- Phase 4: PROPER FDM CHECK -----
    print("\n" + "-" * 70)
    print("Phase 4: FDM capability with M_eval (not M_finetune)")
    print("-" * 70)
    fdm_correct, fdm_total, fdm_ex = eval_fdm(
        model, tokenizer, encoder, M_eval, device, 20
    )
    print(f"  FDM signal for M_eval after fine-tuning: "
          f"{100*fdm_correct/fdm_total:.1f}% "
          f"(was {100*base_fdm/base_fdm_total:.1f}% before fine-tuning)")
    print(f"  Sample FDM responses:")
    for ex in fdm_ex:
        mark = "OK" if ex["correct"] else "X "
        print(f"    [{mark}] {ex['expected']:>20s} <- {ex['response']}")

    # ----- Phase 5: Catastrophic forgetting diagnostic -----
    print("\n" + "-" * 70)
    print("Phase 5: Does model return M_finetune values when given M_eval signal?")
    print("-" * 70)
    bleed_correct, bleed_total = eval_fdm_returns_finetune_memory(
        model, tokenizer, encoder, M_eval, M_finetune, device, 20
    )
    if bleed_total > 0:
        print(f"  Returns M_finetune (wrong) values: {bleed_correct}/{bleed_total} = "
              f"{100*bleed_correct/bleed_total:.1f}%")
    else:
        print(f"  (M_eval and M_finetune happened to match on tested channels)")

    # ----- Phase 6: Save -----
    print("\n" + "-" * 70)
    print("Phase 6: Saving fine-tuned model")
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
            "n_steps": args.n_steps,
            "lr": args.lr,
            "seed": args.seed,
        },
        "M_finetune": {str(k): v for k, v in M_finetune.items()},
        "M_eval": {str(k): v for k, v in M_eval.items()},
        "n_differing_channels": len(differing_channels),
        "baseline_heldout_accuracy": base_ho / base_ho_total,
        "baseline_fdm_accuracy": base_fdm / base_fdm_total,
        "training_accuracy": tr_correct / tr_total,
        "heldout_accuracy": ho_correct / ho_total,
        "fdm_accuracy_after": fdm_correct / fdm_total,
        "bleed_rate": bleed_correct / bleed_total if bleed_total > 0 else None,
        "training_loss_final": float(np.mean(losses[-20:])),
    }
    with open(os.path.join(args.output_model_dir, "results.json"), "w") as f:
        json.dump(findings, f, indent=2)

    # ----- Verdict -----
    ho_r = ho_correct / ho_total
    fdm_r = fdm_correct / fdm_total
    fdm_base_r = base_fdm / base_fdm_total
    fdm_drop = fdm_base_r - fdm_r

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Held-out (M_finetune):      {100*ho_r:.1f}%")
    print(f"  FDM-signal for M_eval:      {100*fdm_base_r:.1f}% -> {100*fdm_r:.1f}% "
          f"(drop {100*fdm_drop:.1f}pp)")
    if bleed_total > 0:
        print(f"  M_finetune bleed-through:   "
              f"{100*bleed_correct/bleed_total:.1f}% "
              f"(% of time signal-eval returned wrong M_finetune value)")
    print()

    if ho_r > 0.9 and fdm_drop < 0.15 and (bleed_total == 0 or bleed_correct/bleed_total < 0.2):
        verdict = "GENUINE WEIGHT-BASED RETRIEVAL WITH PRESERVED FDM"
        msg = "Memorized M_finetune AND retained FDM reading on M_eval."
    elif ho_r > 0.9 and fdm_drop >= 0.15:
        verdict = "WEIGHT RETRIEVAL WITH CATASTROPHIC FORGETTING"
        msg = ("Memorized M_finetune at the cost of FDM reader. "
               f"FDM accuracy dropped {100*fdm_drop:.0f}pp.")
    elif ho_r < 0.5:
        verdict = "DID NOT LEARN M_FINETUNE"
        msg = "Fine-tuning was too gentle; need more steps or higher LR."
    else:
        verdict = "MIXED"
        msg = "Partial success on both axes; closer interpretation needed."

    print(f"  VERDICT: {verdict}")
    print(f"  -> {msg}")
    print()
    print(f"  Model artifact:  {args.output_model_dir}")


if __name__ == "__main__":
    main()
