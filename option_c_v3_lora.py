#!/usr/bin/env python3
"""
Option C v3 (LoRA): Weight-based retrieval via low-rank adapters.

Hypothesis: full fine-tuning destroys the FDM reader because both
capabilities share weight subspaces. A LoRA adapter at low rank only
modifies a small subspace, possibly leaving the FDM reader intact.

Setup:
  - Apply rank-8 LoRA to all attention projections (q_proj, k_proj, v_proj, o_proj)
  - Freeze the host
  - Train ONLY the LoRA params
  - Test: did we learn M_finetune AND retain FDM reading on M_eval?

Same dual-memory eval as v2: M_finetune (seed 42) for fine-tuning,
M_eval (seed 999) for FDM-capability check.

Usage:
    python option_c_v3_lora.py \\
        --base_model prompterminal/fdm-40ch-two-block-qwen3 \\
        --output_model_dir /workspace/FDM_IN_WEIGHTS/two_block_qwen3_lora \\
        --lora_rank 8 --lr 1e-4 --n_steps 500
"""

import os, sys, json, random, time, argparse, shutil, re
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/root/FDM_IN_WEIGHTS/scripts')
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("ERROR: peft library not installed. Run: pip install peft")
    sys.exit(1)

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
    nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()
    return loss.item()


def train_loop(model, tokenizer, pairs, n_steps, lr, device, log_every=50):
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: "
          f"{sum(p.numel() for p in trainable)/1e6:.2f}M")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr/20
    )
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
            print(f"    Step {step:4d}/{n_steps} | loss {avg:.4f} | "
                  f"{time.time()-t0:.0f}s")
    return losses


@torch.no_grad()
def eval_pairs(model, tokenizer, pairs, device, max_new_tokens=20):
    model.eval()
    correct = 0
    total = 0
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
    return correct, total


@torch.no_grad()
def eval_fdm_proper(model, tokenizer, encoder, M_signal, M_label, device, n=10):
    """Partition-sweep-style eval: long output, all 32 channels checked."""
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
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="LoRA tolerates higher LR than full fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
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
            f"Pass --allow_overwrite or pick a new path."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("OPTION C v3: LoRA WEIGHT-BASED RETRIEVAL")
    print("=" * 70)
    print(f"Base model:        {args.base_model}")
    print(f"Output:            {args.output_model_dir}")
    print(f"LoRA rank/alpha:   {args.lora_rank} / {args.lora_alpha}")
    print(f"LR / steps:        {args.lr} / {args.n_steps}")

    print(f"\nLoading {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, trust_remote_code=True
    ).to(device)

    # Apply LoRA to attention projections only
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    encoder = TurboFDMSignalEncoder(
        vocab_size=151936, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    M_finetune = make_memory(seed=42)
    M_eval = make_memory(seed=999)

    differing_channels = [ch for ch in QUERY_CHANNELS if M_finetune[ch] != M_eval[ch]]
    print(f"\nM_finetune (training target):")
    print(f"  TEAM={M_finetune[8]}, REGION={M_finetune[9]}, "
          f"STATUS={M_finetune[3]}, PHASE={M_finetune[10]}")
    print(f"M_eval (FDM eval target):")
    print(f"  TEAM={M_eval[8]}, REGION={M_eval[9]}, "
          f"STATUS={M_eval[3]}, PHASE={M_eval[10]}")
    print(f"  Differing channels: {len(differing_channels)}/{len(QUERY_CHANNELS)}")

    training_pairs = make_pairs(M_finetune, TRAIN_TEMPLATES)
    heldout_pairs = make_pairs(M_finetune, EVAL_HELDOUT_TEMPLATES)

    # ----- Phase 0: Baseline FDM (proper eval) -----
    print("\n" + "-" * 70)
    print("Phase 0: Baseline FDM eval (proper, long-output)")
    print("-" * 70)
    base_fdm_eval = eval_fdm_proper(
        model, tokenizer, encoder, M_eval, M_eval, device, n=10
    )
    base_slot = base_fdm_eval["slot_correct"] / base_fdm_eval["slot_total"]
    base_joint = base_fdm_eval["sample_all"] / base_fdm_eval["n_samples"]
    print(f"  FDM signal=M_eval, label=M_eval:")
    print(f"    Slot accuracy: {100*base_slot:.1f}%")
    print(f"    Sample joint:  {100*base_joint:.1f}%")
    print(f"  Sample response: {base_fdm_eval['sample_responses'][0][:150]}")

    # ----- Phase 1: Train LoRA -----
    print("\n" + "-" * 70)
    print(f"Phase 1: LoRA fine-tuning {args.n_steps} steps at lr={args.lr}")
    print("-" * 70)
    losses = train_loop(model, tokenizer, training_pairs, args.n_steps, args.lr, device)

    # ----- Phase 2: Held-out paraphrases (M_finetune) -----
    print("\n" + "-" * 70)
    print("Phase 2: Held-out paraphrases (M_finetune)")
    print("-" * 70)
    ho_correct, ho_total = eval_pairs(model, tokenizer, heldout_pairs, device)
    print(f"  Held-out: {ho_correct}/{ho_total} = {100*ho_correct/ho_total:.1f}%")

    # ----- Phase 3: FDM capability for M_eval -----
    print("\n" + "-" * 70)
    print("Phase 3: FDM capability for M_eval (after LoRA)")
    print("-" * 70)
    after_fdm_eval = eval_fdm_proper(
        model, tokenizer, encoder, M_eval, M_eval, device, n=10
    )
    after_slot = after_fdm_eval["slot_correct"] / after_fdm_eval["slot_total"]
    after_joint = after_fdm_eval["sample_all"] / after_fdm_eval["n_samples"]
    print(f"  Slot:  {100*after_slot:.1f}% (was {100*base_slot:.1f}%)")
    print(f"  Joint: {100*after_joint:.1f}% (was {100*base_joint:.1f}%)")
    print(f"  Sample response: {after_fdm_eval['sample_responses'][0][:150]}")

    # ----- Phase 4: Bleed check -----
    print("\n" + "-" * 70)
    print("Phase 4: Bleed check (FDM signal=M_eval, looking for M_finetune values)")
    print("-" * 70)
    bleed_eval = eval_fdm_proper(
        model, tokenizer, encoder, M_eval, M_finetune, device, n=10
    )
    bleed_rate = bleed_eval["slot_correct"] / bleed_eval["slot_total"]
    print(f"  Returns M_finetune slots: {100*bleed_rate:.1f}% "
          f"(should be <30% for non-bleed)")

    # ----- Phase 5: Save -----
    print("\n" + "-" * 70)
    print("Phase 5: Saving LoRA adapter + merged model")
    print("-" * 70)
    if os.path.exists(args.output_model_dir):
        shutil.rmtree(args.output_model_dir)
    os.makedirs(args.output_model_dir, exist_ok=True)
    # Save LoRA adapter
    model.save_pretrained(os.path.join(args.output_model_dir, "lora_adapter"))
    tokenizer.save_pretrained(args.output_model_dir)
    # Also save merged model for easy loading
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_model_dir)
    print(f"  LoRA adapter saved to {args.output_model_dir}/lora_adapter")
    print(f"  Merged model saved to {args.output_model_dir}")

    findings = {
        "config": {
            "base_model": args.base_model,
            "n_steps": args.n_steps,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "seed": args.seed,
        },
        "M_finetune": {str(k): v for k, v in M_finetune.items()},
        "M_eval": {str(k): v for k, v in M_eval.items()},
        "n_differing_channels": len(differing_channels),
        "baseline_fdm_slot": base_slot,
        "baseline_fdm_joint": base_joint,
        "heldout_accuracy_after": ho_correct / ho_total,
        "fdm_slot_after": after_slot,
        "fdm_joint_after": after_joint,
        "bleed_rate": bleed_rate,
        "training_loss_final": float(np.mean(losses[-20:])),
    }
    with open(os.path.join(args.output_model_dir, "results.json"), "w") as f:
        json.dump(findings, f, indent=2)

    # ----- Verdict -----
    ho_r = ho_correct / ho_total
    fdm_drop = base_slot - after_slot

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Held-out (M_finetune):          {100*ho_r:.1f}%")
    print(f"  FDM slot for M_eval:            {100*base_slot:.1f}% -> "
          f"{100*after_slot:.1f}% (drop {100*fdm_drop:.1f}pp)")
    print(f"  FDM joint for M_eval:           {100*base_joint:.1f}% -> "
          f"{100*after_joint:.1f}%")
    print(f"  Bleed rate (M_finetune in M_eval signal):  {100*bleed_rate:.1f}%")
    print()

    if ho_r > 0.9 and fdm_drop < 0.15 and bleed_rate < 0.3:
        verdict = "LoRA SUCCESS: WEIGHT RETRIEVAL + FDM PRESERVED"
        msg = "Memorized M_finetune via LoRA AND retained FDM reading."
    elif ho_r > 0.9 and fdm_drop >= 0.15:
        verdict = "LoRA INSUFFICIENT: FDM still degraded"
        msg = ("Even with LoRA, FDM reader is damaged. May need lower rank "
               "or more targeted module selection.")
    elif ho_r < 0.5:
        verdict = "LoRA UNDER-TRAINED"
        msg = "Try higher LR, more steps, or higher rank."
    else:
        verdict = "MIXED"
        msg = "Partial; needs closer analysis."

    print(f"  VERDICT: {verdict}")
    print(f"  -> {msg}")


if __name__ == "__main__":
    main()
