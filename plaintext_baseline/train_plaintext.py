"""
Train a causal LM on Plaintext-40 data.

Loss is masked to answer tokens only (everything after "Answer:"), matching
the FDM training protocol from Section 3 of paper draft 18.

Usage:
    python train_plaintext.py \
        --model_name Qwen/Qwen3-0.6B \
        --train_path data/train.jsonl \
        --output_dir models/qwen3_plaintext40 \
        --batch_size 8 \
        --grad_accum 1 \
        --lr 1e-4
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def build_training_text(sample, channel_names):
    """Build the full sequence: prefix + answer + extras. Returns (text, answer_start_char)."""
    prefix = (
        f"[MEMORY]\n{sample['memory_block']}\n[/MEMORY]\n"
        f"Question: {sample['question']}\nAnswer:"
    )
    # The boundary between "loss-masked" and "loss-active" is right after "Answer:"
    # We emit a leading space + answer + extras as the loss-active region.
    extras = " ".join(
        f"{channel_names[int(i)]}={sample['channel_values'][str(i)]}"
        for i in sample["extra_channels"]
    )
    answer_part = f" {sample['answer']} [EXTRA] {extras}"
    full_text = prefix + answer_part
    return prefix, full_text


class PlaintextDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, channel_names, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.channel_names = channel_names
        self.max_length = max_length
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prefix, full_text = build_training_text(sample, self.channel_names)

        # Tokenize the prefix to find where answer tokens start
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False,
                                   truncation=True, max_length=self.max_length)["input_ids"]

        # Add EOS if room
        if self.tokenizer.eos_token_id is not None and len(full_ids) < self.max_length:
            full_ids = full_ids + [self.tokenizer.eos_token_id]

        # Labels: -100 for prefix tokens (no loss), real ids for answer tokens
        labels = [-100] * min(len(prefix_ids), len(full_ids)) + full_ids[len(prefix_ids):]
        labels = labels[:len(full_ids)]

        return {
            "input_ids": full_ids,
            "labels": labels,
        }


def collate(batch, pad_id):
    """Pad to the max length in the batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        attention_mask[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--train_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every_steps", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_ckpt", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Channel names — imported lazily so train_plaintext.py works as a standalone file
    from channels import CHANNEL_NAMES

    # Tokenizer + model
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        ).cuda()
        print("  flash_attention_2 enabled")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).cuda()
        print("  flash_attention_2 unavailable; using sdpa")

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # Data
    dataset = PlaintextDataset(args.train_path, tokenizer, CHANNEL_NAMES,
                                max_length=args.max_length)
    print(f"  loaded {len(dataset):,} training samples")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        pin_memory=True, drop_last=True,
    )

    # Optimizer + scheduler
    total_steps = (len(loader) * args.epochs) // args.grad_accum
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    print(f"\n  Training plan:")
    print(f"    epochs:        {args.epochs}")
    print(f"    samples/epoch: {len(dataset):,}")
    print(f"    batch:         {args.batch_size} x grad_accum {args.grad_accum} = "
          f"effective {args.batch_size * args.grad_accum}")
    print(f"    optim steps:   {total_steps:,}")
    print(f"    lr:            {args.lr} (cosine, warmup {args.warmup_steps})\n")

    # Train
    model.train()
    step = 0
    t_start = time.perf_counter()
    accum_loss = 0.0
    accum_count = 0
    for epoch in range(args.epochs):
        for it, batch in enumerate(loader):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item() * args.grad_accum
            accum_count += 1

            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    avg_loss = accum_loss / accum_count
                    elapsed = time.perf_counter() - t_start
                    samples_done = step * args.batch_size * args.grad_accum
                    samples_per_sec = samples_done / elapsed
                    eta_h = (total_steps - step) * (elapsed / step) / 3600
                    print(f"  step {step:>6}/{total_steps} "
                          f"| loss {avg_loss:.4f} "
                          f"| {samples_per_sec:.1f} samp/s "
                          f"| ETA {eta_h:.2f}h")
                    accum_loss = 0.0
                    accum_count = 0

                if step % args.save_every_steps == 0:
                    ckpt = output_dir / f"step_{step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"  saved checkpoint: {ckpt}")

    # Final save
    print("\n  Training done; saving final checkpoint...")
    final = output_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    elapsed_h = (time.perf_counter() - t_start) / 3600
    print(f"  Total wall time: {elapsed_h:.2f}h")
    print(f"  Final model: {final}")


if __name__ == "__main__":
    main()
