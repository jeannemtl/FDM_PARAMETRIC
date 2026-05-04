"""
FDM training at K=960 channels.

Self-contained: encoder + dataset + Stage 1 training loop, mirroring the
TurboFDMSignalEncoder math from nhop_source.py but parameterized for K=960.

Encoder parameters chosen to satisfy Nyquist with margin:
  K              = 960
  f_s            = 2000.0 Hz       (>= 2*K=1920, with safety margin)
  N              = 2048 samples    (per encoder, ensures Δf=f_s/N < 1Hz carrier spacing)
  W              = 4096 tokens     (dual encoding: 2N)
  carrier_freqs  = [1, 2, ..., 960] Hz
  num_levels     = 64 (unchanged from K=40 baseline)

Channel schema: channels 0-39 use hand-named MEMORY_SCHEMAS from nhop_source.py
(SECRET, LOCATION, ..., COMMS), channels 40-959 use synthetic CH{idx:03d}/CH{idx:03d}V{0..7}.

Training: 1-hop questions only (single-channel retrieval) with random query
channel per sample. This produces a clean retrieval-trained model directly
comparable to plaintext-K=960 on the isolated retrieval task.

Usage:
    # 1. Generate dataset
    python fdm_K960_source.py generate --n_train 115000 --n_eval 1500

    # 2. Train (in tmux)
    python fdm_K960_source.py train

    # 3. Eval
    python fdm_K960_source.py eval
"""

import sys, os, json, random, time, argparse, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ============================================================
# Hyperparameters for K=960
# ============================================================

K = 960                          # number of channels
F_S = 2000.0                     # sampling rate (Hz) — Nyquist for K=960 with margin
N_PER_ENCODER = 2048             # samples per encoder pass; Δf = f_s/N = 0.977 Hz < 1Hz spacing
W = 2 * N_PER_ENCODER            # total token budget = 4096 (dual encoding)
NUM_LEVELS = 64                  # quantization levels (unchanged from K=40)
A_HIGH = 1.0
A_LOW = 0.25
SEED = 42

# Training hyperparameters
MAX_LENGTH = 4352                # W=4096 + ~256 for question/answer
BATCH_SIZE = 2                   # fits with W=4096 on H200; effective batch via grad_accum
GRAD_ACCUM = 8                   # effective batch = 16
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
EPOCHS = 1

# Paths
DATA_DIR = "data_fdm_K960"
MODEL_DIR = "models/fdm_K960_qwen3"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

QWEN3_BASE = "Qwen/Qwen3-0.6B-Base"
VOCAB_SIZE_QWEN3 = 151936


# ============================================================
# Channel schema: 0-39 hand-named, 40-959 synthetic
# ============================================================

# Hand-named (matching nhop_source.py / paper Table 12)
HAND_NAMED_SCHEMAS = {
    0:  ("SECRET",     ["RED", "BLUE", "GREEN", "GOLD"]),
    1:  ("LOCATION",   ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2:  ("AGENT",      ["ALICE", "BOB", "CAROL", "DAVE"]),
    3:  ("STATUS",     ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4:  ("PRIORITY",   ["HIGH", "MEDIUM", "LOW"]),
    5:  ("BACKUP",     ["AVAILABLE", "UNAVAILABLE"]),
    6:  ("RULE",       ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7:  ("META",       ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY",
                         "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",       ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",     ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",      ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",       ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",      ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",     ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",      ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",    ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",     ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",    ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",    ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",    ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",     ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",       ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",    ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",      ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION",   ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",    ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",       ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE",   ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",      ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION",  ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",      ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",     ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",     ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",     ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",      ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",       ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2",   ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE",   ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",      ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}


def build_schemas(K):
    """Build channel schemas for any K >= 40. 0-39 hand-named; 40+ synthetic."""
    schemas = dict(HAND_NAMED_SCHEMAS)
    for idx in range(40, K):
        name = f"CH{idx:03d}"
        values = [f"{name}V{v}" for v in range(8)]
        schemas[idx] = (name, values)
    return schemas


MEMORY_SCHEMAS = build_schemas(K)
CHANNEL_NAMES = {idx: name for idx, (name, _) in MEMORY_SCHEMAS.items()}


# ============================================================
# FDM Encoder — direct port of TurboFDMSignalEncoder, parameterized for K=960
# ============================================================

class FDMSignalEncoder:
    def __init__(self, vocab_size, tokenizer, K=K,
                 num_tokens_per_encoder=N_PER_ENCODER, sample_rate=F_S,
                 a_high=A_HIGH, a_low=A_LOW, num_levels=NUM_LEVELS, seed=SEED,
                 schemas=MEMORY_SCHEMAS):
        self.K = K
        self.num_tokens_per_encoder = num_tokens_per_encoder
        self.total_tokens = num_tokens_per_encoder * 2
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        self.tokenizer = tokenizer
        self.schemas = schemas
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(K)]

        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(num_tokens_per_encoder, S)

        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(vocab_size, size=num_levels, replace=False)

        # Sanity: check Nyquist
        max_carrier = self.carrier_freqs[-1]
        assert sample_rate >= 2 * max_carrier + 1, \
            f"Nyquist violation: f_s={sample_rate} but max carrier={max_carrier}"
        delta_f = sample_rate / num_tokens_per_encoder
        carrier_spacing = 1.0
        assert delta_f < carrier_spacing, \
            f"Resolution violation: Δf={delta_f:.3f} >= spacing={carrier_spacing}"
        print(f"  FDM encoder ready: K={K}, f_s={sample_rate}, N={num_tokens_per_encoder}, "
              f"Δf={delta_f:.3f}Hz, K_max={int(sample_rate/2 - 1)}")

    def _generate_s_random_interleaver(self, length, S):
        import random as rnd
        rnd.seed(42)
        interleaver = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                valid = True
                for k in range(max(0, i - S + 1), i):
                    if abs(interleaver[j] - interleaver[k]) < S:
                        valid = False
                        break
                if valid:
                    interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                    break
        return interleaver

    def value_to_bits(self, channel_id, value):
        _, values = self.schemas[channel_id]
        idx = values.index(value)
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits

    def encode_memory(self, memory):
        all_bits = {}
        max_bits = 0
        for ch in range(self.K):
            bits, nb = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = bits
            max_bits = max(max_bits, nb)
        for ch in range(self.K):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        num_message_bits = max_bits
        t = np.arange(self.num_tokens_per_encoder) / self.sample_rate
        samples_per_bit = self.num_tokens_per_encoder // num_message_bits

        composite = np.zeros(self.num_tokens_per_encoder)
        for ch in range(self.K):
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens_per_encoder)
                amp = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amp * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[start:end]
                )

        sig_min, sig_max = composite.min(), composite.max()
        sig_range = sig_max - sig_min + 1e-10

        norm1 = (composite - sig_min) / sig_range
        q1 = np.floor(norm1 * (self.num_levels - 1) + 0.5).astype(int)
        q1 = np.clip(q1, 0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]

        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int)
        q2 = np.clip(q2, 0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]

        all_tokens = tokens1 + tokens2
        fdm_text = self.tokenizer.decode(all_tokens)
        return fdm_text, all_tokens


# ============================================================
# Data generation: 1-hop single-channel queries
# ============================================================

def generate_data(args):
    """Generate train + eval datasets. Each sample is one query for one random channel."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Generating FDM-K={K} dataset to {DATA_DIR}/")
    print(f"  n_train: {args.n_train:,}")
    print(f"  n_eval:  {args.n_eval}")

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder = FDMSignalEncoder(
        vocab_size=VOCAB_SIZE_QWEN3, tokenizer=tokenizer,
    )

    rng = random.Random(SEED)

    splits = {"train": args.n_train, "eval": args.n_eval}
    for split_name, n_samples in splits.items():
        out_path = os.path.join(DATA_DIR, f"{split_name}.jsonl")
        print(f"\nGenerating {split_name} ({n_samples:,} samples)...")
        with open(out_path, "w") as f:
            for i in tqdm(range(n_samples)):
                # sample channel values for all K channels
                memory = {ch: rng.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(K)}

                # encode the memory once
                fdm_text, token_ids = encoder.encode_memory(memory)

                # pick a random channel to query
                query_ch = rng.randrange(K)
                query_name = CHANNEL_NAMES[query_ch]
                answer = memory[query_ch]

                sample = {
                    "fdm_text": fdm_text,
                    "query_channel": query_ch,
                    "query_name": query_name,
                    "answer": answer,
                    "memory": {str(ch): memory[ch] for ch in range(K)},
                }
                f.write(json.dumps(sample) + "\n")
        print(f"  -> {out_path}")

    # Print one sample for sanity check
    with open(os.path.join(DATA_DIR, "train.jsonl")) as f:
        sample = json.loads(f.readline())
    prefix = f"[MEMORY]{sample['fdm_text']}[/MEMORY]\nQuestion: What is the value of {sample['query_name']}?\nAnswer: {sample['answer']}"
    enc = tokenizer(prefix, add_special_tokens=False)
    print(f"\nSAMPLE PROMPT (first 200 chars):")
    print(prefix[:200] + "..." if len(prefix) > 200 else prefix)
    print(f"\n  total tokens: {len(enc['input_ids'])}")
    print(f"  query channel: {sample['query_channel']} ({sample['query_name']})")
    print(f"  answer: {sample['answer']}")


# ============================================================
# Training
# ============================================================

class FDMDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=MAX_LENGTH):
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prefix = (
            f"[MEMORY]{s['fdm_text']}[/MEMORY]\n"
            f"Question: What is the value of {s['query_name']}?\n"
            f"Answer:"
        )
        full = prefix + f" {s['answer']}"

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False,
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


def train_model(args):
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
    dataset = FDMDataset(train_path, tokenizer)
    print(f"  loaded {len(dataset):,} training samples")

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
    print(f"    K              = {K}")
    print(f"    f_s            = {F_S}")
    print(f"    N_per_encoder  = {N_PER_ENCODER}")
    print(f"    W (total)      = {W}")
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

                if step % 50 == 0:
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


# ============================================================
# Evaluation: isolated single-channel retrieval
# ============================================================

def wilson_halfwidth(p, n, z=1.96):
    if n == 0:
        return float("nan")
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    delta = (z/denom) * ((p*(1-p)/n + z*z/(4*n*n))**0.5)
    return max(centre + delta - p, p - (centre - delta))


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens=12):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=False).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    responses = []
    for i in range(out.shape[0]):
        prompt_len = inputs["input_ids"][i].shape[0]
        new_tokens = out[i, prompt_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return responses


def parse_value(response):
    """Extract single value token."""
    s = response.strip()
    if not s:
        return ""
    return s.split()[0].rstrip(".,;:[]")


def evaluate_model(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    final_dir = os.path.join(MODEL_DIR, "final")
    print(f"Loading {final_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(final_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            final_dir, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", trust_remote_code=True,
        ).cuda()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            final_dir, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).cuda()
    model.eval()

    eval_path = os.path.join(DATA_DIR, "eval.jsonl")
    with open(eval_path) as f:
        samples = [json.loads(l) for l in f]
    print(f"  loaded {len(samples)} eval samples")

    correct = 0
    by_channel = defaultdict(lambda: [0, 0])
    by_channel_band = defaultdict(lambda: [0, 0])

    t0 = time.perf_counter()
    for batch_start in range(0, len(samples), args.batch_size):
        batch = samples[batch_start:batch_start + args.batch_size]
        prompts = [
            f"[MEMORY]{s['fdm_text']}[/MEMORY]\n"
            f"Question: What is the value of {s['query_name']}?\nAnswer:"
            for s in batch
        ]
        responses = generate_batch(model, tokenizer, prompts)
        for s, resp in zip(batch, responses):
            predicted = parse_value(resp)
            ok = (predicted == s["answer"])
            correct += int(ok)
            ch = s["query_channel"]
            by_channel[ch][0] += int(ok)
            by_channel[ch][1] += 1
            band = ch // (K // 10)
            by_channel_band[band][0] += int(ok)
            by_channel_band[band][1] += 1

        if (batch_start // args.batch_size) % 20 == 0:
            done = batch_start + len(batch)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(samples) - done) / rate if rate > 0 else 0
            print(f"    [{done}/{len(samples)}] {rate:.1f} samp/s, ETA {eta/60:.1f}min")

    p = correct / len(samples)
    ci = wilson_halfwidth(p, len(samples))

    print(f"\n  Eval done in {(time.perf_counter()-t0)/60:.1f} min")
    print(f"  Single-channel retrieval (K=960): {p*100:.2f}% ± {ci*100:.2f}  (n={len(samples)})")
    print(f"\n  Per-channel-band (10 bands of 96 channels each):")
    for b in sorted(by_channel_band.keys()):
        c, t = by_channel_band[b]
        bp = c/t if t else 0
        bar = "#" * int(bp * 40)
        ch_lo, ch_hi = b * (K//10), (b+1)*(K//10) - 1
        print(f"    ch{ch_lo:>3}-{ch_hi:<3}  {bp*100:5.1f}%  (n={t})  {bar}")

    out = {
        "K": K, "f_s": F_S, "N_per_encoder": N_PER_ENCODER, "W": W,
        "n_samples": len(samples),
        "single_channel_retrieval": {
            "p": p, "n_correct": correct, "n": len(samples), "ci": ci,
        },
        "by_channel_band": {
            str(b): {"p": v[0]/v[1], "n_correct": v[0], "n": v[1]}
            for b, v in sorted(by_channel_band.items())
        },
        "by_channel": {
            str(ch): {"p": v[0]/v[1], "n_correct": v[0], "n": v[1]}
            for ch, v in sorted(by_channel.items())
        },
    }
    out_path = os.path.join(RESULTS_DIR, "fdm_K960_qwen3_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--n_train", type=int, default=115_000)
    g.add_argument("--n_eval", type=int, default=1500)

    sub.add_parser("train")

    e = sub.add_parser("eval")
    e.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    if args.cmd == "generate":
        generate_data(args)
    elif args.cmd == "train":
        train_model(args)
    elif args.cmd == "eval":
        evaluate_model(args)


if __name__ == "__main__":
    main()
