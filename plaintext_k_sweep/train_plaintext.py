"""
Train a causal LM on Plaintext-K data for any K.

Reads K from the dataset (each sample has a "K" field) and builds the
matching channel vocabulary. Auto-scales max_length based on K unless
overridden.

Usage:
    python train_plaintext.py \
        --model_name Qwen/Qwen3-0.6B \
        --train_path data_K160/train.jsonl \
        --output_dir models/qwen3_plaintext160 \
        --batch_size 8 --grad_accum 1 --lr 1e-4
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


# Auto-scale max_length based on K. Each NAME=VALUE pair is roughly 8 chars
# so ~2 tokens, plus the [EXTRA] block in the answer adds another ~2 tokens
# per channel. So total tokens scale as ~4-5 * K. Add 100 for question/wrappers.
def auto_max_length(K: int) -> int:
    # Empirical: K=40 ~280 tokens, K=160 ~1118, K=320 ~2240.
    # Roughly 7 tokens per channel total ([MEMORY] entry + [EXTRA] entry).
    estimated = 7 * K + 200
    # round up to nice powers of 2 for batching efficiency
    for cap in (512, 1024, 2048, 4096, 8192):
        if estimated <= cap:
            return cap
    return 16384


def detect_K(jsonl_path: str) -> int:
    """Read the first sample to discover K."""
    with open(jsonl_path) as f:
        first = json.loads(f.readline())
    if "K" in first:
        return first["K"]
    # Backwards-compat: infer K from extra_channels
    return max(first["extra_channels"]) + 1


def build_training_text(sample, channel_names):
    prefix = (
        f"[MEMORY]\n{sample['memory_block']}\n[/MEMORY]\n"
        f"Question: {sample['question']}\nAnswer:"
    )
    extras = " ".join(
        f"{channel_names[int(i)]}={sample['channel_values'][str(i)]}"
        for i in sample["extra_channels"]
    )
    answer_part = f" {sample['answer']} [EXTRA] {extras}"
    full_text = prefix + answer_part
    return prefix, full_text


class PlaintextDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, channel_names, max_length):
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

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False,
                                   truncation=True, max_length=self.max_length)["input_ids"]

        if self.tokenizer.eos_token_id is not None and len(full_ids) < self.max_length:
            full_ids = full_ids + [self.tokenizer.eos_token_id]

        labels = [-100] * min(len(prefix_ids), len(full_ids)) + full_ids[len(prefix_ids):]
        labels = labels[:len(full_ids)]

        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
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
    p.add_argument("--max_length", type=int, default=None,
                   help="Override auto-scaled max_length")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every_steps", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_ckpt", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect K from data
    K = detect_K(args.train_path)
    if args.max_length is None:
        args.max_length = auto_max_length(K)
    print(f"  K = {K}, max_length = {args.max_length}")

    # Build the K-specific channel vocabulary
    from channels import build_channel_vocab
    vocab = build_channel_vocab(K)
    channel_names = {idx: name for idx, (name, _) in vocab.items()}

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

    dataset = PlaintextDataset(args.train_path, tokenizer, channel_names, args.max_length)
    print(f"  loaded {len(dataset):,} training samples")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        pin_memory=True, drop_last=True,
    )

    total_steps = (len(loader) * args.epochs) // args.grad_accum
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    print(f"\n  Training plan:")
    print(f"    K:             {K}")
    print(f"    epochs:        {args.epochs}")
    print(f"    samples/epoch: {len(dataset):,}")
    print(f"    batch:         {args.batch_size} x grad_accum {args.grad_accum} = "
          f"effective {args.batch_size * args.grad_accum}")
    print(f"    optim steps:   {total_steps:,}")
    print(f"    max_length:    {args.max_length}")
    print(f"    lr:            {args.lr} (cosine, warmup {args.warmup_steps})\n")

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

                if step <= 10 or step % args.log_every == 0:
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

    print("\n  Training done; saving final checkpoint...")
    final = output_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    elapsed_h = (time.perf_counter() - t_start) / 3600
    print(f"  Total wall time: {elapsed_h:.2f}h")
    print(f"  Final model: {final}")


if __name__ == "__main__":
    main()
