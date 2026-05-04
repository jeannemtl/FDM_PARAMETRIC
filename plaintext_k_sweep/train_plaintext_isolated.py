"""
Train Qwen3-0.6B on Plaintext-K=960 with isolated single-channel queries.

Mirrors fdm_K960_source.py's training format exactly:
  Input:  [MEMORY]<plaintext NAME=VALUE block>[/MEMORY]
          Question: What is the value of CHXXX?
          Answer:
  Target: <single value>

Same hyperparams as FDM training so the comparison is apples-to-apples.

Usage:
    python -u train_plaintext_isolated.py 2>&1 | tee logs/plaintext_K960_isolated_train.log
"""

import sys, os, json, time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# Hyperparameters — match fdm_K960_source.py for apples-to-apples comparison
MAX_LENGTH = 4352
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
EPOCHS = 1

DATA_DIR = "data_K960_isolated"
MODEL_DIR = "models/qwen3_0p6b_plaintext960_isolated"
LOGS_DIR = "logs"

QWEN3_BASE = "Qwen/Qwen3-0.6B-Base"


class IsolatedDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=MAX_LENGTH):
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prefix = (
            f"[MEMORY]\n{s['memory_block']}\n[/MEMORY]\n"
            f"Question: What is the value of {s['query_name']}?\n"
            f"Answer:"
        )
        full = prefix + f" {s['answer']}"

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False,
                                   truncation=True, max_length=self.max_length)["input_ids"]
        if self.tokenizer.eos_token_id is not None and len(full_ids) < self.max_length:
            full_ids = full_ids + [self.tokenizer.eos_token_id]

        # Defensive: if prefix is itself too long (shouldn't happen at K=960 but just in case),
        # the answer span gets truncated and labels become all -100, causing NaN loss.
        # Detect this and fall back to a non-truncated representation.
        if len(prefix_ids) >= len(full_ids) - 1:
            # Answer was truncated. Reduce prefix to fit answer.
            answer_ids = self.tokenizer(f" {s['answer']}", add_special_tokens=False)["input_ids"]
            # Keep room for answer + EOS
            keep = self.max_length - len(answer_ids) - 1
            prefix_ids = prefix_ids[:keep]
            full_ids = prefix_ids + answer_ids
            if self.tokenizer.eos_token_id is not None:
                full_ids = full_ids + [self.tokenizer.eos_token_id]

        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
        labels = labels[:len(full_ids)]
        # Sanity check: at least one label must be != -100
        assert any(l != -100 for l in labels), \
            f"ALL LABELS MASKED for idx={idx}, prefix_ids_len={len(prefix_ids)}, full_ids_len={len(full_ids)}"

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
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"Loading {QWEN3_BASE}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN3_BASE, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", trust_remote_code=True,
        ).cuda()
        print("  flash_attention_2 enabled")
    except Exception as e:
        print(f"  flash_attention_2 unavailable ({e}); using sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            QWEN3_BASE, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).cuda()

    train_path = os.path.join(DATA_DIR, "train.jsonl")
    dataset = IsolatedDataset(train_path, tokenizer)
    print(f"  loaded {len(dataset):,} training samples")

    # Sanity-check first sample to catch labeling issues immediately
    s0 = dataset[0]
    n_loss_active = sum(1 for l in s0["labels"] if l != -100)
    n_total = len(s0["labels"])
    print(f"  sanity: sample 0 has {n_loss_active}/{n_total} loss-active tokens")
    assert n_loss_active > 0, "Sample 0 has zero loss-active tokens — labels are broken"

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        pin_memory=True, drop_last=True,
    )

    total_steps = (len(loader) * EPOCHS) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps,
    )

    print(f"\n  Training plan:")
    print(f"    K              = 960 (isolated single-channel queries)")
    print(f"    samples/epoch  = {len(dataset):,}")
    print(f"    batch/grad_acc = {BATCH_SIZE} x {GRAD_ACCUM} = effective {BATCH_SIZE*GRAD_ACCUM}")
    print(f"    optim steps    = {total_steps:,}")
    print(f"    max_length     = {MAX_LENGTH}")
    print(f"    lr             = {LR} (cosine, warmup {WARMUP_STEPS})\n")

    model.train()
    step = 0
    t_start = time.perf_counter()
    accum_loss = 0.0
    accum_count = 0
    for epoch in range(EPOCHS):
        for it, batch in enumerate(loader):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item() * GRAD_ACCUM
            accum_count += 1

            if (it + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # Early-step logging then back off
                if step <= 10 or step % 50 == 0:
                    avg_loss = accum_loss / accum_count
                    elapsed = time.perf_counter() - t_start
                    samples_done = step * BATCH_SIZE * GRAD_ACCUM
                    samples_per_sec = samples_done / elapsed
                    eta_h = (total_steps - step) * (elapsed / step) / 3600
                    print(f"  step {step:>6}/{total_steps} "
                          f"| loss {avg_loss:.4f} "
                          f"| {samples_per_sec:.1f} samp/s "
                          f"| ETA {eta_h:.2f}h")
                    accum_loss = 0.0
                    accum_count = 0

                if step % 2000 == 0:
                    ckpt = os.path.join(MODEL_DIR, f"step_{step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"  saved checkpoint: {ckpt}")

    print("\n  Training done; saving final checkpoint...")
    final = os.path.join(MODEL_DIR, "final")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    elapsed_h = (time.perf_counter() - t_start) / 3600
    print(f"  Total wall time: {elapsed_h:.2f}h")
    print(f"  Final model: {final}")


if __name__ == "__main__":
    main()
