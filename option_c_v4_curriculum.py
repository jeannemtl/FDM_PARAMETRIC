#!/usr/bin/env python3
"""
Option C v4 (curriculum): Train weight-storage AND FDM-reading simultaneously.

Hypothesis: standard fine-tuning destroys FDM because the no-signal gradient
has no counterweight. A curriculum that mixes both signals during training
should let the model maintain both circuits.

Training mix (50/50):
  Type A: no-signal weight retrieval for M_finetune
  Type B: FDM signal for fresh random memory + question (FDM reading)

If the model can hold both, you have evidence that weight-storage and
FDM-decoding occupy independent circuits in this architecture.

Usage:
    python option_c_v4_curriculum.py \\
        --base_model prompterminal/fdm-40ch-two-block-qwen3 \\
        --output_model_dir /workspace/FDM_IN_WEIGHTS/two_block_qwen3_curriculum \\
        --n_steps 1000 --lr 1e-5
"""

import os, sys, json, random, time, argparse, shutil, re
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


def random_memory():
    return {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}


def make_pairs(M, templates):
    pairs = []
    for ch in QUERY_CHANNELS:
        for tpl in templates:
            prompt = tpl.format(ch=CHANNEL_NAMES[ch])
            answer = f" {CHANNEL_NAMES[ch]}={M[ch]}."
            pairs.append((prompt, answer, CHANNEL_NAMES[ch], M[ch]))
    return pairs


def make_no_signal_batch(M_finetune):
    """Type A: no-signal weight retrieval batch."""
    ch = random.choice(QUERY_CHANNELS)
    template = random.choice(TRAIN_TEMPLATES)
    prompt = template.format(ch=CHANNEL_NAMES[ch])
    answer = f" {CHANNEL_NAMES[ch]}={M_finetune[ch]}."
    return prompt, answer


def make_fdm_signal_batch(encoder):
    """Type B: FDM signal for fresh random memory."""
    M_random = random_memory()
    ch = random.choice(QUERY_CHANNELS)
    template = random.choice(TRAIN_TEMPLATES)
    fdm_text, _ = encoder.encode_memory(M_random)
    prompt = (f"[MEMORY]{fdm_text}[/MEMORY]\n"
              f"{template.format(ch=CHANNEL_NAMES[ch])}")
    answer = f" {CHANNEL_NAMES[ch]}={M_random[ch]}."
    return prompt, answer


def train_step(model, tokenizer, prompt, answer, optimizer, device, max_len=1536):
    full_text = prompt + answer
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    prefix_len = len(full_ids) - len(answer_ids)

    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
        prefix_len = min(prefix_len, max_len - 1)

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


def train_curriculum(model, tokenizer, encoder, M_finetune, n_steps, lr, device,
                      mix_ratio=0.5, log_every=50):
    """Curriculum training: mix_ratio is fraction of TYPE A (no-signal) batches.
    1-mix_ratio is the fraction of TYPE B (FDM-signal) batches."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr/20
    )
    model.train()
    losses_a = []
    losses_b = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        if random.random() < mix_ratio:
            # Type A: no-signal
            prompt, answer = make_no_signal_batch(M_finetune)
            loss = train_step(model, tokenizer, prompt, answer, optimizer, device)
            losses_a.append(loss)
        else:
            # Type B: FDM-signal
            prompt, answer = make_fdm_signal_batch(encoder)
            loss = train_step(model, tokenizer, prompt, answer, optimizer, device)
            losses_b.append(loss)
        scheduler.step()

        if step % log_every == 0:
            recent_a = losses_a[-log_every:] if losses_a else [0]
            recent_b = losses_b[-log_every:] if losses_b else [0]
            avg_a = np.mean(recent_a)
            avg_b = np.mean(recent_b)
            elapsed = time.time() - t0
            print(f"    Step {step:4d}/{n_steps} | "
                  f"loss_A {avg_a:.4f} (n={len(recent_a)}) | "
                  f"loss_B {avg_b:.4f} (n={len(recent_b)}) | "
                  f"{elapsed:.0f}s")

    return losses_a, losses_b


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
        if f"{ch_name}={correct_value}" in response:
            correct += 1
        total += 1
        if len(examples) < 3:
            examples.append({
                "expected": f"{ch_name}={correct_value}",
                "response": response[:100],
                "correct": f"{ch_name}={correct_value}" in response,
            })
    return correct, total, examples


@torch.no_grad()
def eval_fdm_proper(model, tokenizer, encoder, M_signal, M_label, device, n=10):
    """Partition-sweep-style: long output, all 32 channels checked."""
    model.eval()
    fdm_text, _ = encoder.encode_memory(M_signal)
    ch_names = [CHANNEL_NAMES[k] for k in QUERY_CHANNELS]
    question = f"Report values for: {', '.join(ch_names)}."
    prompt = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"

    slot_correct = 0
    slot_total = 0
    sample_all = 0
    sample_responses = []

    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=350, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        sample_responses.append(answer[:200])

        all_right = True
        for k in QUERY_CHANNELS:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(M_label[k]) + r"\b"
            if re.search(pat, answer):
                slot_correct += 1
            else:
                all_right = False
            slot_total += 1
        if all_right:
            sample_all += 1

    return {
        "slot_correct": slot_correct,
        "slot_total": slot_total,
        "sample_all": sample_all,
        "n_samples": n,
        "sample_responses": sample_responses[:2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",
                        default="prompterminal/fdm-40ch-two-block-qwen3")
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mix_ratio", type=float, default=0.5,
                        help="Fraction of no-signal (type A) batches. "
                             "0.5 = balanced; 0.7 = more weight retrieval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    out_real = os.path.realpath(args.output_model_dir)
    if os.path.exists(args.base_model):
        if os.path.realpath(args.base_model) == out_real:
            raise ValueError("output_model_dir must differ from base_model.")
    if os.path.exists(args.output_model_dir) and not args.allow_overwrite:
        raise ValueError(
            f"Output directory exists. Pass --allow_overwrite or pick a new path."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("OPTION C v4: CURRICULUM (weight retrieval + FDM reading)")
    print("=" * 70)
    print(f"Base:           {args.base_model}")
    print(f"Output:         {args.output_model_dir}")
    print(f"Steps / LR:     {args.n_steps} / {args.lr}")
    print(f"Mix ratio:      {args.mix_ratio:.2f} type-A (no-signal) / "
          f"{1-args.mix_ratio:.2f} type-B (FDM)")

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

    M_finetune = make_memory(seed=42)
    M_eval = make_memory(seed=999)

    differing = [ch for ch in QUERY_CHANNELS if M_finetune[ch] != M_eval[ch]]
    print(f"\nM_finetune: TEAM={M_finetune[8]}, REGION={M_finetune[9]}, "
          f"STATUS={M_finetune[3]}")
    print(f"M_eval:     TEAM={M_eval[8]}, REGION={M_eval[9]}, "
          f"STATUS={M_eval[3]}")
    print(f"Differing channels: {len(differing)}/{len(QUERY_CHANNELS)}")

    heldout_pairs = make_pairs(M_finetune, EVAL_HELDOUT_TEMPLATES)

    # ----- Phase 0: Baseline -----
    print("\n" + "-" * 70)
    print("Phase 0: Baseline (proper FDM eval)")
    print("-" * 70)
    base_ho_c, base_ho_t, _ = eval_pairs(model, tokenizer, heldout_pairs, device)
    base_fdm = eval_fdm_proper(model, tokenizer, encoder, M_eval, M_eval, device, 10)
    base_slot = base_fdm["slot_correct"] / base_fdm["slot_total"]
    base_joint = base_fdm["sample_all"] / base_fdm["n_samples"]
    print(f"  Held-out (M_finetune): {100*base_ho_c/base_ho_t:.1f}% "
          f"(expected ~0%)")
    print(f"  FDM slot for M_eval:   {100*base_slot:.1f}% (expected high)")
    print(f"  FDM joint for M_eval:  {100*base_joint:.1f}%")

    # ----- Phase 1: Curriculum training -----
    print("\n" + "-" * 70)
    print(f"Phase 1: Curriculum {args.n_steps} steps at lr={args.lr}")
    print("-" * 70)
    losses_a, losses_b = train_curriculum(
        model, tokenizer, encoder, M_finetune,
        args.n_steps, args.lr, device, args.mix_ratio
    )

    # ----- Phase 2: Held-out paraphrases (M_finetune, no signal) -----
    print("\n" + "-" * 70)
    print("Phase 2: Held-out paraphrases for M_finetune (no signal)")
    print("-" * 70)
    ho_correct, ho_total, ho_ex = eval_pairs(
        model, tokenizer, heldout_pairs, device
    )
    print(f"  Held-out: {100*ho_correct/ho_total:.1f}%")
    for ex in ho_ex[:3]:
        mark = "OK" if ex["correct"] else "X "
        print(f"    [{mark}] {ex['expected']:>20s} <- {ex['response']}")

    # ----- Phase 3: FDM capability for M_eval (proper eval) -----
    print("\n" + "-" * 70)
    print("Phase 3: FDM capability for M_eval (after curriculum)")
    print("-" * 70)
    after_fdm = eval_fdm_proper(
        model, tokenizer, encoder, M_eval, M_eval, device, 10
    )
    after_slot = after_fdm["slot_correct"] / after_fdm["slot_total"]
    after_joint = after_fdm["sample_all"] / after_fdm["n_samples"]
    print(f"  Slot:  {100*after_slot:.1f}% (was {100*base_slot:.1f}%)")
    print(f"  Joint: {100*after_joint:.1f}% (was {100*base_joint:.1f}%)")
    print(f"  Sample response: {after_fdm['sample_responses'][0][:200]}")

    # ----- Phase 4: Bleed -----
    print("\n" + "-" * 70)
    print("Phase 4: Bleed (M_eval signal, looking for M_finetune values)")
    print("-" * 70)
    bleed = eval_fdm_proper(
        model, tokenizer, encoder, M_eval, M_finetune, device, 10
    )
    bleed_rate = bleed["slot_correct"] / bleed["slot_total"]
    print(f"  Returns M_finetune slots: {100*bleed_rate:.1f}% "
          f"(should be near M_finetune/M_eval agreement rate, "
          f"~{100*(32-len(differing))/32:.0f}%)")

    # ----- Phase 5: Save -----
    print("\n" + "-" * 70)
    print("Phase 5: Saving model")
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
            "mix_ratio": args.mix_ratio,
            "seed": args.seed,
        },
        "M_finetune": {str(k): v for k, v in M_finetune.items()},
        "M_eval": {str(k): v for k, v in M_eval.items()},
        "n_differing_channels": len(differing),
        "baseline_fdm_slot": base_slot,
        "baseline_fdm_joint": base_joint,
        "heldout_accuracy_after": ho_correct / ho_total,
        "fdm_slot_after": after_slot,
        "fdm_joint_after": after_joint,
        "bleed_rate": bleed_rate,
        "bleed_expected_baseline": (32 - len(differing)) / 32,
        "training_loss_final_typeA": float(np.mean(losses_a[-20:])) if losses_a else None,
        "training_loss_final_typeB": float(np.mean(losses_b[-20:])) if losses_b else None,
    }
    with open(os.path.join(args.output_model_dir, "results.json"), "w") as f:
        json.dump(findings, f, indent=2)

    # ----- Verdict -----
    ho_r = ho_correct / ho_total
    fdm_drop = base_slot - after_slot
    # If bleed_rate ~ 25% (= channels where M_finetune == M_eval / 32),
    # that's just coincidental agreement, not actual bleed.
    expected_bleed = (32 - len(differing)) / 32
    bleed_excess = bleed_rate - expected_bleed

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Held-out (M_finetune):       {100*ho_r:.1f}%")
    print(f"  FDM slot (M_eval):           "
          f"{100*base_slot:.1f}% -> {100*after_slot:.1f}%  "
          f"(drop {100*fdm_drop:.1f}pp)")
    print(f"  Bleed:                       "
          f"{100*bleed_rate:.1f}% (expected "
          f"{100*expected_bleed:.1f}% from coincidence)")
    print(f"  Bleed excess:                {100*bleed_excess:.1f}pp")
    print()

    if ho_r > 0.9 and fdm_drop < 0.15 and bleed_excess < 0.15:
        verdict = "CURRICULUM SUCCESS: BOTH CIRCUITS COEXIST"
        msg = ("Model holds M_finetune in weights AND retains FDM reading. "
               "Independent circuits.")
    elif ho_r > 0.9 and fdm_drop >= 0.15:
        verdict = "CURRICULUM PARTIAL: WEIGHT WINS"
        msg = ("Weight retrieval learned but FDM degraded despite mixing. "
               "Try lower LR or higher mix_ratio toward FDM.")
    elif ho_r < 0.5 and fdm_drop < 0.15:
        verdict = "CURRICULUM PARTIAL: FDM WINS"
        msg = ("FDM preserved but weight retrieval not learned. "
               "Try higher LR, more steps, or higher mix_ratio toward A.")
    elif bleed_excess >= 0.15:
        verdict = "BLEEDING"
        msg = ("Model is treating both inputs the same way. "
               "Circuits are coupled.")
    else:
        verdict = "MIXED"
        msg = "Closer interpretation needed."

    print(f"  VERDICT: {verdict}")
    print(f"  -> {msg}")


if __name__ == "__main__":
    main()
