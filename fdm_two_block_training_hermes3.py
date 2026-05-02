"""
Two-Block FDM Base Model Fine-Tuning (Fixed)
=============================================

Fine-tunes the FDM model to read from TWO memory blocks.
No new tokens — uses existing [MEMORY]...[/MEMORY] with text prefixes.
Stays in fp16 with autocast for training (no .float() conversion).

Format:
  [MEMORY]BLOCK_A <FDM signal A>[/MEMORY]
  [MEMORY]BLOCK_B <FDM signal B>[/MEMORY]
  Question: Report all values for channels 8-39.
  Answer: Context: TEAM=RED_TEAM, REGION=EAST, ...

Block A carries channels 8-23, Block B carries channels 24-39.
The model learns to read from both position ranges.

Usage:
    python fdm_two_block_training.py --n_steps 5000 --eval_every 1000
"""

import sys, os, re, json, random, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/root/FDM_IN_WEIGHTS/scripts')
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

try:
    from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS
except ImportError:
    print("ERROR: nhop_source.py not found")
    print("Run: python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('prompterminal/fdm-parametric-weights', 'scripts/nhop_source.py', local_dir='/root/FDM_IN_WEIGHTS')\"")
    sys.exit(1)

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]


def make_encoder(tokenizer):
    return TurboFDMSignalEncoder(
        vocab_size=128256, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )


def random_memory():
    return {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}


# ============================================================
# Training data generation
# ============================================================

def generate_two_block_sample(encoder, tokenizer, memory,
                               block_a_channels, block_b_channels):
    fdm_text_a, _ = encoder.encode_memory(memory)
    fdm_text_b, _ = encoder.encode_memory(memory)

    all_channels = sorted(set(block_a_channels) | set(block_b_channels))
    ch_names = [CHANNEL_NAMES[k] for k in all_channels]
    question = f"Report values for: {', '.join(ch_names)}."
    parts = [f"{CHANNEL_NAMES[k]}={memory[k]}" for k in all_channels]
    answer = "Context: " + ", ".join(parts) + "."

    q_text = f"\nQuestion: {question}\nAnswer:"
    a_text = f" {answer}"

    prompt = f"[MEMORY]BLOCK_A {fdm_text_a}[/MEMORY][MEMORY]BLOCK_B {fdm_text_b}[/MEMORY]{q_text}{a_text}"

    full_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(a_text, add_special_tokens=False)
    prefix_len = len(full_ids) - len(a_ids)
    labels = [-100] * prefix_len + full_ids[prefix_len:]

    return full_ids, labels


def generate_single_block_sample(encoder, tokenizer, memory, channels):
    fdm_text, _ = encoder.encode_memory(memory)

    ch_names = [CHANNEL_NAMES[k] for k in channels]
    question = f"Report values for: {', '.join(ch_names)}."
    parts = [f"{CHANNEL_NAMES[k]}={memory[k]}" for k in channels]
    answer = "Context: " + ", ".join(parts) + "."

    q_text = f"\nQuestion: {question}\nAnswer:"
    a_text = f" {answer}"
    prompt = f"[MEMORY]{fdm_text}[/MEMORY]{q_text}{a_text}"

    full_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(a_text, add_special_tokens=False)
    prefix_len = len(full_ids) - len(a_ids)
    labels = [-100] * prefix_len + full_ids[prefix_len:]

    return full_ids, labels


# ============================================================
# Training
# ============================================================

def train(model, tokenizer, encoder, device, n_steps, lr, eval_every, n_eval,
          mix_ratio=0.3):
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr / 20)

    SPLITS = [
        (list(range(8, 24)), list(range(24, 40))),
        (list(range(8, 20)), list(range(20, 40))),
        (list(range(8, 28)), list(range(28, 40))),
    ]

    t0 = time.time()
    losses = []

    for step in range(1, n_steps + 1):
        mem = random_memory()

        if random.random() < mix_ratio:
            channels = list(range(8, 40))
            input_ids, labels = generate_single_block_sample(
                encoder, tokenizer, mem, channels)
        else:
            split = random.choice(SPLITS)
            input_ids, labels = generate_two_block_sample(
                encoder, tokenizer, mem, split[0], split[1])

        max_len = 2048
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        label_tensor = torch.tensor([labels], dtype=torch.long).to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            outputs = model(input_tensor, labels=label_tensor)
            loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 500 == 0 or step == n_steps:
            avg = np.mean(losses[-500:])
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step:5d}/{n_steps} | loss {avg:.4f} | "
                  f"lr {cur_lr:.1e} | {elapsed:.0f}s", flush=True)

        if eval_every > 0 and step % eval_every == 0:
            evaluate(model, tokenizer, encoder, device, n_eval)
            model.train()


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, tokenizer, encoder, device, n_eval=20):
    model.eval()

    single_correct = 0; single_total = 0
    two_correct = 0; two_total = 0

    all_ch = list(range(8, 40))

    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()

        # Single-block
        fdm_text, _ = encoder.encode_memory(mem)
        ch_names = [CHANNEL_NAMES[k] for k in all_ch]
        question = f"Report values for: {', '.join(ch_names)}."
        prompt_s = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        ids_s = tokenizer.encode(prompt_s, return_tensors='pt').to(device)
        with torch.no_grad():
            out_s = model.generate(ids_s, max_new_tokens=350, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        text_s = tokenizer.decode(out_s[0][ids_s.shape[1]:], skip_special_tokens=True)
        for k in all_ch:
            if re.search(rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b", text_s):
                single_correct += 1
            single_total += 1

        # Two-block
        fdm_text_a, _ = encoder.encode_memory(mem)
        fdm_text_b, _ = encoder.encode_memory(mem)
        prompt_2 = (f"[MEMORY]BLOCK_A {fdm_text_a}[/MEMORY]"
                     f"[MEMORY]BLOCK_B {fdm_text_b}[/MEMORY]"
                     f"\nQuestion: {question}\nAnswer:")
        ids_2 = tokenizer.encode(prompt_2, return_tensors='pt').to(device)
        with torch.no_grad():
            out_2 = model.generate(ids_2, max_new_tokens=350, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        text_2 = tokenizer.decode(out_2[0][ids_2.shape[1]:], skip_special_tokens=True)
        for k in all_ch:
            if re.search(rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b", text_2):
                two_correct += 1
            two_total += 1

    s_acc = single_correct / single_total if single_total > 0 else 0
    t_acc = two_correct / two_total if two_total > 0 else 0

    print(f"\n    Single-block: {s_acc*100:.1f}% (should stay ~93%+)", flush=True)
    print(f"    Two-block:    {t_acc*100:.1f}% (target: >80%)", flush=True)

    return s_acc, t_acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--mix_ratio", type=float, default=0.3)
    parser.add_argument("--output_dir", default="/workspace/FDM_IN_WEIGHTS/two_block_model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  TWO-BLOCK FDM FINE-TUNING (fp16, no new tokens)")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.n_steps}")
    print(f"  LR: {args.lr}")
    print(f"  Mix ratio: {args.mix_ratio}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32).to(device)

    trainable = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} ({trainable/1e6:.0f}M)")

    encoder = make_encoder(tokenizer)

    print("\n  Initial evaluation:", flush=True)
    model.eval()
    evaluate(model, tokenizer, encoder, device, n_eval=args.n_eval)

    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60, flush=True)
    train(model, tokenizer, encoder, device,
          n_steps=args.n_steps, lr=args.lr,
          eval_every=args.eval_every, n_eval=args.n_eval,
          mix_ratio=args.mix_ratio)

    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60, flush=True)
    model.eval()
    s_acc, t_acc = evaluate(model, tokenizer, encoder, device, n_eval=args.n_eval)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    results = {
        "single_block_acc": float(s_acc),
        "two_block_acc": float(t_acc),
        "config": {
            "n_steps": args.n_steps, "lr": args.lr,
            "mix_ratio": args.mix_ratio, "base_model": args.model,
        },
    }
    json.dump(results, open(os.path.join(args.output_dir, "results.json"), "w"), indent=2)
    print(f"\n  Saved to {args.output_dir}")
    print(f"  Single-block: {s_acc*100:.1f}%")
    print(f"  Two-block:    {t_acc*100:.1f}%")


if __name__ == "__main__":
    main()
