"""
FDM-V3 Adaptive Sampling
=========================

Scales n_samples with channel count during curriculum training:
  Stage 0 (5ch):  n_samples=256,  fs=100  → Δf=0.39 Hz
  Stage 1 (10ch): n_samples=256,  fs=100  → Δf=0.39 Hz
  Stage 2 (20ch): n_samples=512,  fs=200  → Δf=0.39 Hz
  Stage 3 (32ch): n_samples=1024, fs=400  → Δf=0.39 Hz

Frequency resolution stays constant. Higher channel counts get
more carrier cycles, maintaining the same separation quality
per channel regardless of spectral crowding.

This is the write-head analog of the paper's density scaling:
doubling channels requires doubling samples to maintain accuracy.

The carrier_pool layer uses a fixed max size (2*1024=2048) and
zero-pads shorter signals, so the same network handles all resolutions.

Usage:
    python fdm_v3_adaptive.py --n_steps 40000 --n_eval 30
"""

import sys, os, re, json, random, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, '/root/FDM_IN_WEIGHTS')
from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]


def make_encoder(tokenizer):
    return TurboFDMSignalEncoder(
        vocab_size=151936, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

def random_memory():
    return {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}

def memory_to_indices(memory):
    ch_ids = list(range(NUM_CHANNELS))
    val_ids = []
    for ch in range(NUM_CHANNELS):
        _, values = MEMORY_SCHEMAS[ch]
        val_ids.append(values.index(memory[ch]))
    return ch_ids, val_ids

def generate_s_random_interleaver(length, S, seed=42):
    import random as rnd
    rnd.seed(seed)
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


# ============================================================
# FDM-V3 with Adaptive Sampling
# ============================================================

class FDMAdaptiveWriteHead(nn.Module):
    """
    FDM-Struct V3 with adaptive sampling per forward call.
    
    The carrier_pool accepts variable-length dual signals by using
    a fixed max_dual_n input size with zero-padding. This lets the
    same network process 256, 512, or 1024 samples.
    
    The carrier matrix and S-random permutation are computed on-the-fly
    for each n_samples setting (cached for efficiency).
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len,
                 n_channels=40, max_values=5, embed_dim=384,
                 max_samples=1024):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.max_samples = max_samples
        self.max_dual_n = 2 * max_samples

        # Channel + value modulation
        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)
        self.modulation_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Carrier pool: fixed size input (max_dual_n), zero-padded for shorter signals
        self.carrier_pool = nn.Sequential(
            nn.Linear(self.max_dual_n, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Self-attention over carrier features
        feat_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu')
        self.feature_attn = nn.TransformerEncoder(feat_layer, num_layers=2)

        # Cross-attention: positions attend to carrier features
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # KV projection
        kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, kv_per_pos))

        # Cache for carrier matrices and interleavers at different n_samples
        self._carrier_cache = {}

    def _get_carriers(self, n_samples, fs, device):
        """Get or compute carrier matrix and interleaver for given n_samples."""
        key = (n_samples, fs)
        if key not in self._carrier_cache:
            freqs = torch.arange(1, self.n_channels + 1, dtype=torch.float32)
            t = torch.linspace(0, n_samples / fs, n_samples)
            carriers = torch.sin(2 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0))
            carriers = carriers.to(device)

            S = int(math.sqrt(n_samples / 2))
            s_perm = generate_s_random_interleaver(n_samples, S, seed=43)
            s_perm = torch.tensor(s_perm, dtype=torch.long, device=device)

            self._carrier_cache[key] = (carriers, s_perm)
        return self._carrier_cache[key]

    def forward(self, ch_ids, val_ids, device, n_samples=256, fs=100.0):
        """
        n_samples and fs can vary per call (per curriculum stage).
        """
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)

        # Rich modulation envelope per channel
        raw = self.ch_embed(ch_t) + self.val_embed(val_t)
        modulation = self.modulation_proj(raw)  # (1, K, E)

        # Get carriers for this n_samples
        carriers, s_perm = self._get_carriers(n_samples, fs, device)

        # Build per-carrier dual signals
        dual_carriers = []
        for k in range(self.n_channels):
            carrier = carriers[k]              # (n_samples,)
            interleaved = carrier[s_perm]       # (n_samples,)
            dual = torch.cat([carrier, interleaved])  # (2*n_samples,)
            dual_carriers.append(dual)
        dual_stack = torch.stack(dual_carriers, dim=0)  # (K, 2*n_samples)

        # Zero-pad to max_dual_n for the fixed-size carrier_pool
        if dual_stack.shape[1] < self.max_dual_n:
            pad_size = self.max_dual_n - dual_stack.shape[1]
            dual_stack = F.pad(dual_stack, (0, pad_size))  # (K, max_dual_n)

        # Pool all carriers: (K, max_dual_n) -> (K, E)
        pooled = self.carrier_pool(dual_stack)  # (K, E)
        pooled = pooled.unsqueeze(0)            # (1, K, E)

        # Scale by modulation
        carrier_features = pooled * modulation  # (1, K, E)

        # Self-attention over carrier features
        carrier_features = self.feature_attn(carrier_features)

        # Cross-attention: positions query carrier features
        pos = self.pos_embed
        attended, _ = self.cross_attn(pos, carrier_features, carrier_features)
        pos_context = self.cross_norm(pos + attended)

        kv_flat = self.pos_to_kv(pos_context)
        kv = kv_flat.view(1, self.seq_len, self.n_layers, 2,
                          self.n_kv_heads, self.head_dim)
        kv = kv.permute(2, 3, 0, 4, 1, 5)
        return kv

    def make_cache(self, ch_ids, val_ids, device, n_samples=256, fs=100.0):
        kv = self.forward(ch_ids, val_ids, device, n_samples, fs)
        cache = DynamicCache()
        for li in range(self.n_layers):
            cache.update(kv[li, 0].half(), kv[li, 1].half(), li)
        return cache, kv.shape[4]


# ============================================================
# Training and eval (shared code)
# ============================================================

def build_answer_for_channels(memory, active_channels):
    parts = [f"{CHANNEL_NAMES[k]}={memory[k]}" for k in active_channels]
    return "Context: " + ", ".join(parts) + "."

def build_question_for_channels(active_channels):
    ch_names = [CHANNEL_NAMES[k] for k in active_channels]
    return f"Report values for: {', '.join(ch_names)}."

def e2e_train_step(model, tokenizer, write_head, memory, active_channels,
                   device, n_samples=256, fs=100.0):
    question = build_question_for_channels(active_channels)
    answer = build_answer_for_channels(memory, active_channels)
    q_text = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
    a_text = f" {answer}"
    full_text = q_text + a_text
    q_ids = tokenizer.encode(q_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    full_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)
    ch_ids, val_ids = memory_to_indices(memory)
    kv_raw = write_head(ch_ids, val_ids, device, n_samples=n_samples, fs=fs)
    fdm_seq_len = kv_raw.shape[4]
    cache = DynamicCache()
    for li in range(write_head.n_layers):
        cache.update(kv_raw[li, 0], kv_raw[li, 1], li)
    seq_len = len(full_ids)
    position_ids = torch.arange(fdm_seq_len, fdm_seq_len + seq_len,
                                 device=device).unsqueeze(0)
    attn_mask = torch.ones(1, fdm_seq_len + seq_len, device=device, dtype=torch.long)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(full_tensor, past_key_values=cache,
                       position_ids=position_ids, attention_mask=attn_mask,
                       use_cache=False)
    labels = torch.full((1, seq_len), -100, dtype=torch.long, device=device)
    labels[0, len(q_ids):] = full_tensor[0, len(q_ids):]
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1), ignore_index=-100)
    return loss

def generate_with_head(model, tokenizer, write_head, memory, active_channels,
                       device, n_samples=256, fs=100.0):
    write_head.eval()
    ch_ids, val_ids = memory_to_indices(memory)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        cache, fdm_seq_len = write_head.make_cache(ch_ids, val_ids, device,
                                                     n_samples=n_samples, fs=fs)
    question = build_question_for_channels(active_channels)
    rest = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
    rest_ids = tokenizer.encode(rest, add_special_tokens=False)
    rest_t = torch.tensor([rest_ids], dtype=torch.long).to(device)
    pos = torch.arange(fdm_seq_len, fdm_seq_len + len(rest_ids),
                        device=device).unsqueeze(0)
    attn = torch.ones(1, fdm_seq_len + len(rest_ids), device=device, dtype=torch.long)
    generated_ids = []
    past_kv = cache
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(rest_t, past_key_values=past_kv,
                   position_ids=pos, attention_mask=attn, use_cache=True)
        past_kv = out.past_key_values
        nt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated_ids.append(nt.item())
        tl = fdm_seq_len + len(rest_ids) + 1
        for s in range(249):
            p = torch.tensor([[tl - 1 + s]], device=device)
            a = torch.ones(1, tl + s, device=device, dtype=torch.long)
            o = model(nt, past_key_values=past_kv,
                     position_ids=p, attention_mask=a, use_cache=True)
            past_kv = o.past_key_values
            nt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            tid = nt.item()
            generated_ids.append(tid)
            if tid == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def evaluate_head(model, tokenizer, write_head, active_channels, device,
                  n_eval=30, n_samples=256, fs=100.0):
    write_head.eval()
    channel_correct = {CHANNEL_NAMES[k]: 0 for k in active_channels}
    total = 0
    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        text = generate_with_head(model, tokenizer, write_head, mem,
                                  active_channels, device,
                                  n_samples=n_samples, fs=fs)
        for k in active_channels:
            name = CHANNEL_NAMES[k]
            expected = mem[k]
            pattern = rf"\b{re.escape(name)}={re.escape(expected)}\b"
            if re.search(pattern, text):
                channel_correct[name] += 1
        total += 1
    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    return overall, per_ch


# ============================================================
# Curriculum with adaptive sampling
# ============================================================

CURRICULUM = [
    {"name": "5 ch",  "channels": list(range(8, 13)),
     "lr_mult": 1.0, "n_samples": 256,  "fs": 100.0},
    {"name": "10 ch", "channels": list(range(8, 18)),
     "lr_mult": 0.8, "n_samples": 256,  "fs": 100.0},
    {"name": "20 ch", "channels": list(range(8, 28)),
     "lr_mult": 0.6, "n_samples": 512,  "fs": 200.0},
    {"name": "32 ch", "channels": list(range(8, 40)),
     "lr_mult": 0.4, "n_samples": 1024, "fs": 400.0},
]


def train_adaptive(model, tokenizer, write_head, device,
                   total_steps, lr, eval_every, n_eval):
    """Train with per-stage sampling frequency."""
    # Non-uniform: 1:2:3:4 ratio
    ratios = [1, 2, 3, 4]
    total_ratio = sum(ratios)
    stage_steps = [total_steps * r // total_ratio for r in ratios]

    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=1e-4)
    global_step = 0
    t0 = time.time()
    stage_results = []

    for stage_idx, stage in enumerate(CURRICULUM):
        active = stage["channels"]
        n_steps = stage_steps[stage_idx]
        stage_lr = lr * stage["lr_mult"]
        n_samp = stage["n_samples"]
        fs = stage["fs"]
        warmup_steps = min(500, n_steps // 10)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps - warmup_steps, eta_min=stage_lr / 20)
        for pg in optimizer.param_groups:
            pg['lr'] = stage_lr

        print(f"\n  Stage {stage_idx}: {stage['name']} | "
              f"n_samples={n_samp}, fs={fs:.0f} Hz | "
              f"LR: {stage_lr:.1e} | Steps: {n_steps}")
        print(f"    Carrier cycles at max freq ({len(active)+8} Hz): "
              f"{(len(active)+8) * n_samp / fs:.1f}")
        print(f"    Δf = {fs/n_samp:.2f} Hz")

        losses = []
        write_head.train()

        for step in range(1, n_steps + 1):
            global_step += 1
            if step <= warmup_steps:
                for pg in optimizer.param_groups:
                    pg['lr'] = stage_lr * step / warmup_steps
            else:
                scheduler.step()

            mem = random_memory()
            loss = e2e_train_step(model, tokenizer, write_head, mem,
                                  active, device, n_samples=n_samp, fs=fs)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if step % 2000 == 0 or step == n_steps:
                avg = np.mean(losses[-2000:])
                elapsed = time.time() - t0
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"    Step {step:6d}/{n_steps} (g{global_step}) | "
                      f"loss {avg:.4f} | lr {cur_lr:.1e} | {elapsed:.0f}s")

            if eval_every > 0 and global_step % eval_every == 0:
                acc, _ = evaluate_head(model, tokenizer, write_head,
                                       active, device, n_eval=n_eval,
                                       n_samples=n_samp, fs=fs)
                print(f"    *** Eval ({len(active)} ch, {n_samp} samp): {acc*100:.1f}% ***")
                write_head.train()

        acc, per_ch = evaluate_head(model, tokenizer, write_head,
                                     active, device, n_eval=n_eval,
                                     n_samples=n_samp, fs=fs)
        print(f"    Stage {stage_idx} final: {acc*100:.1f}%")
        worst = sorted(per_ch, key=per_ch.get)[:3]
        for n in worst:
            print(f"      {n}: {per_ch[n]*100:.1f}%")
        stage_results.append({
            "stage": stage_idx, "channels": len(active),
            "n_samples": n_samp, "fs": fs, "acc": float(acc),
        })

    return stage_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_steps", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--embed_dim", type=int, default=384)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    encoder = make_encoder(tokenizer)
    mem = random_memory()
    _, fdm_tokens = encoder.encode_memory(mem)
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    seq_len = len(memory_start) + len(fdm_tokens)

    n_kv_heads = model.config.num_key_value_heads
    head_dim = 128
    max_values = max(len(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS))

    write_head = FDMAdaptiveWriteHead(
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len,
        n_channels=NUM_CHANNELS, max_values=max_values,
        embed_dim=args.embed_dim, max_samples=1024,
    ).to(device)

    n_params = sum(p.numel() for p in write_head.parameters())
    print(f"  FDM-V3 Adaptive: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Seq len: {seq_len}")
    print(f"\n  Curriculum sampling schedule:")
    for stage in CURRICULUM:
        print(f"    {stage['name']:8s}: n_samples={stage['n_samples']}, "
              f"fs={stage['fs']:.0f} Hz, Δf={stage['fs']/stage['n_samples']:.2f} Hz")

    print(f"\n  Training: {args.n_steps} total steps (non-uniform 1:2:3:4)")

    # Train
    results = train_adaptive(
        model, tokenizer, write_head, device,
        total_steps=args.n_steps, lr=args.lr,
        eval_every=args.eval_every, n_eval=args.n_eval,
    )

    # Summary
    print("\n" + "=" * 60)
    print("ADAPTIVE SAMPLING RESULTS")
    print("=" * 60)
    for r in results:
        print(f"  {r['channels']:2d} ch ({r['n_samples']} samp, {r['fs']:.0f} Hz): "
              f"{r['acc']*100:.1f}%")

    print(f"\n  Comparison (same 40K budget, fixed 256 samples):")
    print(f"    FDM-V3 fixed:     5ch=100%, 10ch=92%, 20ch=69%, 32ch=54%")
    print(f"    Turbo V2:         5ch=100%, 10ch=76%, 20ch=64%, 32ch=54%")
    print(f"    Turbo V2 (75K):   5ch=100%, 10ch=88%, 20ch=68%, 32ch=64%")
    print(f"\n  Reference:")
    print(f"    FDM in context: 93.7%")
    print(f"    Frozen KV:      89.1%")

    out = {
        "stages": results,
        "config": {
            "n_steps": args.n_steps, "embed_dim": args.embed_dim,
            "curriculum": CURRICULUM,
        },
    }
    json.dump(out, open("fdm_adaptive_results.json", "w"), indent=2, default=float)
    torch.save(write_head.state_dict(), "fdm_v3_adaptive.pt")
    print("\nSaved results and checkpoint")


if __name__ == "__main__":
    main()
