"""
FDM Write Head: Turbo Extended + FDM-Struct V3
===============================================

Two experiments:

1. TURBO EXTENDED: Non-uniform stage budgets (5K/10K/20K/40K = 75K total)
   to push Turbo V2 toward the 89.1% frozen KV ceiling.

2. FDM-STRUCT V3: Fixes the amplitude bottleneck that caused V2 to plateau.
   Instead of 200 scalar amplitudes controlling 95M parameters, each channel
   gets a full embedding that modulates the carrier via LEARNED FREQUENCY
   RESPONSE — the carrier structure provides orthogonality, but the
   modulation is rich enough for gradients to flow.

   V2 bottleneck: amp_embed(200 scalars) → signal(512) → chunk → project(95M)
   V3 fix:        channel_embed(40×384) → frequency-modulated signal(40×N) 
                  → per-carrier features → cross-attend → KV

   The key insight: keep carrier orthogonality (sin at integer Hz) but
   let each carrier carry a RICH modulation learned end-to-end, not just
   a 3-bit amplitude pattern.

Usage:
    python fdm_turbo_extended.py --mode turbo --n_steps 75000
    python fdm_turbo_extended.py --mode fdm_v3 --n_steps 40000
    python fdm_turbo_extended.py --mode both --n_steps 40000
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
# Turbo V2 (same as before, proven architecture)
# ============================================================

class TurboWriteHeadV2(nn.Module):
    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len,
                 n_channels=40, max_values=5, embed_dim=384,
                 n_attn_layers=2, turbo_iters=3):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.turbo_iters = turbo_iters
        self.embed_dim = embed_dim

        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)

        S = int(math.sqrt(n_channels / 2))
        self.interleaver = generate_s_random_interleaver(n_channels, S, seed=42)
        self.deinterleaver = [0] * n_channels
        for i, j in enumerate(self.interleaver):
            self.deinterleaver[j] = i

        class CompDecoder(nn.Module):
            def __init__(self, embed_dim, n_attn_layers):
                super().__init__()
                layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=4,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1, batch_first=True, activation='gelu')
                self.attn = nn.TransformerEncoder(layer, num_layers=n_attn_layers)
                self.ext_gate = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
                self.ext_fuse = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim), nn.GELU())
                self.confidence = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
            def forward(self, x, ext_in=None):
                if ext_in is not None:
                    gate = self.ext_gate(torch.cat([x, ext_in], -1))
                    fused = self.ext_fuse(torch.cat([x, ext_in], -1))
                    x = x + gate * fused
                out = self.attn(x)
                conf = self.confidence(out)
                ext_out = conf * (out - x)
                return out, ext_out

        self.decoder_a = CompDecoder(embed_dim, n_attn_layers)
        self.decoder_b = CompDecoder(embed_dim, n_attn_layers)
        self.iter_weights = nn.Parameter(torch.ones(turbo_iters, 2) * 0.5)

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(embed_dim)

        kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, kv_per_pos))

    def interleave(self, x):
        return x[:, torch.tensor(self.interleaver, device=x.device), :]
    def deinterleave(self, x):
        return x[:, torch.tensor(self.deinterleaver, device=x.device), :]

    def forward(self, ch_ids, val_ids, device):
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)
        raw = self.ch_embed(ch_t) + self.val_embed(val_t)
        emb_nat = raw
        emb_int = self.interleave(raw)
        ext_a, ext_b = None, None
        for it in range(self.turbo_iters):
            ext_for_a = self.deinterleave(ext_b) if ext_b is not None else None
            out_a, ext_a = self.decoder_a(emb_nat, ext_for_a)
            ext_for_b = self.interleave(ext_a)
            out_b, ext_b = self.decoder_b(emb_int, ext_for_b)
        channel_context = out_a
        pos = self.pos_embed
        attended, _ = self.cross_attn(pos, channel_context, channel_context)
        pos_context = self.cross_norm(pos + attended)
        kv_flat = self.pos_to_kv(pos_context)
        kv = kv_flat.view(1, self.seq_len, self.n_layers, 2,
                          self.n_kv_heads, self.head_dim)
        kv = kv.permute(2, 3, 0, 4, 1, 5)
        return kv

    def make_cache(self, ch_ids, val_ids, device):
        kv = self.forward(ch_ids, val_ids, device)
        cache = DynamicCache()
        for li in range(self.n_layers):
            cache.update(kv[li, 0].half(), kv[li, 1].half(), li)
        return cache, kv.shape[4]


# ============================================================
# FDM-Struct V3: Wide carrier modulation (fixes amplitude bottleneck)
# ============================================================

class FDMStructWriteHeadV3(nn.Module):
    """
    The V2 failure was: 200 amplitude scalars → 512-dim signal → project.
    The amplitude bottleneck starved the gradients.
    
    V3 fix: each channel produces a FULL embed_dim-dimensional modulation
    envelope that multiplies the carrier. The carrier provides orthogonality,
    the modulation provides expressiveness.
    
    For each channel k with value v:
      1. Embed (k, v) → modulation vector m_k of dim embed_dim
      2. Carrier matrix C_k[i,j] = sin(2π f_k t_j) for each embed dim i
         (same frequency, different learned phase/amplitude per dim)
      3. Channel contribution = m_k ⊙ C_k  (element-wise, broadcast over time)
      4. Sum all channel contributions → (embed_dim, N) feature map
      5. Cross-attend: positions query the feature map → KV
    
    This gives 40 × 384 = 15,360 learnable modulation parameters feeding
    into the signal (vs 200 in V2), matching Turbo's input dimensionality,
    while keeping carrier orthogonality.
    
    The flatness across channel counts should be preserved because the
    carriers are still orthogonal — sin(2π·9·t) ⊥ sin(2π·10·t) regardless
    of what modulation envelope is applied.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len,
                 n_channels=40, max_values=5, embed_dim=384,
                 n_samples=256, fs=100.0):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.embed_dim = embed_dim

        # Channel + value → modulation envelope (FULL embed_dim, not 3 scalars)
        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)
        self.modulation_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Fixed carrier matrix: (n_channels, n_samples)
        freqs = torch.arange(1, n_channels + 1, dtype=torch.float32)
        t = torch.linspace(0, n_samples / fs, n_samples)
        carriers = torch.sin(2 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0))
        self.register_buffer("carriers", carriers)  # (K, N)

        # S-random interleaver on time samples
        S = int(math.sqrt(n_samples / 2))
        s_perm = generate_s_random_interleaver(n_samples, S, seed=43)
        self.register_buffer("s_perm", torch.tensor(s_perm, dtype=torch.long))

        # Per-carrier feature projection
        # Each carrier's modulated signal → embed_dim features over N positions
        # We treat (embed_dim × dual_N) as a 2D feature map
        # Then reshape into a sequence for cross-attention
        dual_n = 2 * n_samples  # natural + interleaved
        self.n_feature_positions = n_channels  # one feature per carrier

        # Carrier → feature: aggregate each carrier's contribution over time
        # (N,) → (embed_dim,) per carrier, producing (K, embed_dim) feature sequence
        self.carrier_pool = nn.Sequential(
            nn.Linear(dual_n, embed_dim),
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

    def forward(self, ch_ids, val_ids, device):
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)

        # Rich modulation envelope per channel
        raw = self.ch_embed(ch_t) + self.val_embed(val_t)  # (1, K, E)
        modulation = self.modulation_proj(raw)               # (1, K, E)

        # Generate per-carrier modulated signals
        # For each carrier k, multiply modulation[k] (E-dim) with carrier[k] (N-dim)
        # Result: per-carrier signal of shape (E, N) per channel
        # We compute: modulation[k] outer-product carrier[k], then pool over time

        # Efficient: for each carrier, compute modulated dual signal
        # Build per-carrier dual signals, pool each, scale by modulation
        dual_carriers = []
        for k in range(self.n_channels):
            carrier = self.carriers[k]
            interleaved = carrier[self.s_perm]
            dual = torch.cat([carrier, interleaved])
            dual_carriers.append(dual)
        dual_stack = torch.stack(dual_carriers, dim=0)    # (K, 2N)
        pooled = self.carrier_pool(dual_stack)            # (K, E)
        pooled = pooled.unsqueeze(0)                      # (1, K, E)
        carrier_features = pooled * modulation            # (1, K, E)

        # Self-attention over carrier features (captures inter-carrier structure)
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

    def make_cache(self, ch_ids, val_ids, device):
        kv = self.forward(ch_ids, val_ids, device)
        cache = DynamicCache()
        for li in range(self.n_layers):
            cache.update(kv[li, 0].half(), kv[li, 1].half(), li)
        return cache, kv.shape[4]


# ============================================================
# Training and eval (shared)
# ============================================================

def build_answer_for_channels(memory, active_channels):
    parts = [f"{CHANNEL_NAMES[k]}={memory[k]}" for k in active_channels]
    return "Context: " + ", ".join(parts) + "."

def build_question_for_channels(active_channels):
    ch_names = [CHANNEL_NAMES[k] for k in active_channels]
    return f"Report values for: {', '.join(ch_names)}."

def e2e_train_step(model, tokenizer, write_head, memory, active_channels, device):
    question = build_question_for_channels(active_channels)
    answer = build_answer_for_channels(memory, active_channels)
    q_text = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
    a_text = f" {answer}"
    full_text = q_text + a_text
    q_ids = tokenizer.encode(q_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    full_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)
    ch_ids, val_ids = memory_to_indices(memory)
    kv_raw = write_head(ch_ids, val_ids, device)
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

def generate_with_head(model, tokenizer, write_head, memory, active_channels, device):
    write_head.eval()
    ch_ids, val_ids = memory_to_indices(memory)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        cache, fdm_seq_len = write_head.make_cache(ch_ids, val_ids, device)
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

def evaluate_head(model, tokenizer, write_head, active_channels, device, n_eval=30):
    write_head.eval()
    channel_correct = {CHANNEL_NAMES[k]: 0 for k in active_channels}
    total = 0
    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        text = generate_with_head(model, tokenizer, write_head, mem,
                                  active_channels, device)
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
# Training with non-uniform stage budgets
# ============================================================

def train_head_nonuniform(model, tokenizer, write_head, device, name,
                          stage_steps, lr, eval_every, n_eval):
    """
    stage_steps: list of steps per stage, e.g. [5000, 10000, 20000, 40000]
    """
    STAGES = [
        {"name": "5 channels",  "channels": list(range(8, 13)),  "lr_mult": 1.0},
        {"name": "10 channels", "channels": list(range(8, 18)),  "lr_mult": 0.8},
        {"name": "20 channels", "channels": list(range(8, 28)),  "lr_mult": 0.6},
        {"name": "32 channels", "channels": list(range(8, 40)),  "lr_mult": 0.4},
    ]

    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=1e-4)
    global_step = 0
    t0 = time.time()
    stage_results = []

    for stage_idx, stage in enumerate(STAGES):
        active = stage["channels"]
        n_steps = stage_steps[stage_idx]
        stage_lr = lr * stage["lr_mult"]
        warmup_steps = min(500, n_steps // 10)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps - warmup_steps, eta_min=stage_lr / 20)
        for pg in optimizer.param_groups:
            pg['lr'] = stage_lr

        print(f"\n  [{name}] Stage {stage_idx}: {stage['name']} | "
              f"LR: {stage_lr:.1e} | Steps: {n_steps}")

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
            loss = e2e_train_step(model, tokenizer, write_head, mem, active, device)
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
                                       active, device, n_eval=n_eval)
                print(f"    *** Eval ({len(active)} ch): {acc*100:.1f}% ***")
                write_head.train()

        acc, per_ch = evaluate_head(model, tokenizer, write_head,
                                     active, device, n_eval=n_eval)
        print(f"    Stage {stage_idx} final: {acc*100:.1f}%")
        worst = sorted(per_ch, key=per_ch.get)[:3]
        for n in worst:
            print(f"      {n}: {per_ch[n]*100:.1f}%")
        stage_results.append({"stage": stage_idx, "channels": len(active),
                              "acc": float(acc)})

    return stage_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--mode", choices=["turbo", "fdm_v3", "both"], default="both")
    parser.add_argument("--n_steps", type=int, default=40000,
                        help="For turbo: total steps (non-uniform split). "
                             "For fdm_v3: steps per stage (uniform).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--turbo_iters", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--fs", type=float, default=None)
    args = parser.parse_args()

    if args.fs is None:
        args.fs = 100.0 * args.n_samples / 256.0

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

    results = {}

    # ---- Turbo Extended ----
    if args.mode in ("turbo", "both"):
        turbo = TurboWriteHeadV2(
            n_layers=model.config.num_hidden_layers,
            n_kv_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len,
            n_channels=NUM_CHANNELS, max_values=max_values,
            embed_dim=args.embed_dim, n_attn_layers=2,
            turbo_iters=args.turbo_iters).to(device)
        tp = sum(p.numel() for p in turbo.parameters())
        print(f"\n  Turbo V2: {tp:,} ({tp/1e6:.1f}M)")

        # Non-uniform: more steps for harder stages
        # Split total steps as 1:2:3:4 ratio
        total = args.n_steps
        turbo_stage_steps = [
            total // 10,      # 5ch: easy, converges fast
            total * 2 // 10,  # 10ch
            total * 3 // 10,  # 20ch
            total * 4 // 10,  # 32ch: hardest, most steps
        ]
        print(f"  Stage steps: {turbo_stage_steps} = {sum(turbo_stage_steps)} total")

        print("\n" + "=" * 60)
        print("TURBO V2 EXTENDED (non-uniform stages)")
        print("=" * 60)
        turbo_results = train_head_nonuniform(
            model, tokenizer, turbo, device, "Turbo-V2",
            turbo_stage_steps, args.lr, args.eval_every, args.n_eval)
        results["turbo_v2"] = {"params": tp, "stages": turbo_results,
                               "stage_steps": turbo_stage_steps}
        torch.save(turbo.state_dict(), "turbo_v2_extended.pt")

    # ---- FDM-Struct V3 ----
    if args.mode in ("fdm_v3", "both"):
        fdm = FDMStructWriteHeadV3(
            n_layers=model.config.num_hidden_layers,
            n_kv_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len,
            n_channels=NUM_CHANNELS, max_values=max_values,
            embed_dim=args.embed_dim, n_samples=args.n_samples,
            fs=args.fs).to(device)
        fp = sum(p.numel() for p in fdm.parameters())
        print(f"\n  FDM-Struct V3: {fp:,} ({fp/1e6:.1f}M)")
        print(f"  Signal: n_samples={args.n_samples}, fs={args.fs:.1f}")

        # Uniform stages for FDM
        fdm_stage_steps = [args.n_steps // 4] * 4

        print("\n" + "=" * 60)
        print("FDM-STRUCT V3 (wide carrier modulation)")
        print("=" * 60)
        fdm_results = train_head_nonuniform(
            model, tokenizer, fdm, device, "FDM-V3",
            fdm_stage_steps, args.lr, args.eval_every, args.n_eval)
        results["fdm_v3"] = {"params": fp, "stages": fdm_results}
        torch.save(fdm.state_dict(), "fdm_v3_write_head.pt")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n  {name} ({r['params']/1e6:.1f}M):")
        for s in r["stages"]:
            print(f"    {s['channels']:2d} ch: {s['acc']*100:.1f}%")

    print(f"\n  Previous results:")
    print(f"    Turbo V2 (40K uniform):  5ch=100%, 10ch=76%, 20ch=64%, 32ch=54%")
    print(f"    FDM-Struct V2 (40K):     flat ~31-34% (amplitude bottleneck)")
    print(f"  Reference:")
    print(f"    FDM in context: 93.7%")
    print(f"    Frozen KV:      89.1%")

    json.dump(results, open("fdm_turbo_extended_results.json", "w"), indent=2, default=float)
    print("\nSaved results and checkpoints")


if __name__ == "__main__":
    main()
