"""
FDM Write Head Extended Training
=================================

Trains the two best architectures (Turbo and FDM-Struct) for 40K steps
with fixes for the projection bottleneck:

1. No context pooling — signal features project to KV per-position
   directly instead of pooling to a single 256-dim vector
2. Wider embedding (384 vs 256)
3. 10K steps per curriculum stage (vs 2K)
4. Warmup + cosine annealing

Usage:
    python fdm_extended_training.py --n_steps 40000 --eval_every 5000
    python fdm_extended_training.py --n_steps 80000 --eval_every 10000  # overnight
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


# ============================================================
# S-random interleaver
# ============================================================

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
# Improved Turbo Write Head (no bottleneck)
# ============================================================

class TurboWriteHeadV2(nn.Module):
    """
    Turbo write head with per-position KV projection (no pooling bottleneck).
    
    Instead of: channels → pool to 1 vector → expand to seq_len positions
    Now:        channels → per-channel context → cross-attend with positions → KV
    
    Each position in the KV cache gets its own projection conditioned on
    the full channel context, rather than all positions sharing a single
    pooled representation.
    """

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

        # Channel/value embeddings
        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)

        # S-random interleaver on channel indices
        S = int(math.sqrt(n_channels / 2))
        self.interleaver = generate_s_random_interleaver(n_channels, S, seed=42)
        self.deinterleaver = [0] * n_channels
        for i, j in enumerate(self.interleaver):
            self.deinterleaver[j] = i

        # Component decoders
        class CompDecoder(nn.Module):
            def __init__(self, embed_dim, n_attn_layers):
                super().__init__()
                layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=4,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1, batch_first=True, activation='gelu'
                )
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

        # Per-position KV projection (NO pooling bottleneck)
        # Position embeddings cross-attend to channel context
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        # Cross-attention: positions attend to channel context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # KV projection per position
        kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, kv_per_pos),
        )

    def interleave(self, x):
        idx = torch.tensor(self.interleaver, device=x.device)
        return x[:, idx, :]

    def deinterleave(self, x):
        idx = torch.tensor(self.deinterleaver, device=x.device)
        return x[:, idx, :]

    def forward(self, ch_ids, val_ids, device):
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)
        raw = self.ch_embed(ch_t) + self.val_embed(val_t)

        emb_nat = raw
        emb_int = self.interleave(raw)

        ext_a, ext_b = None, None
        contexts = []

        for it in range(self.turbo_iters):
            ext_for_a = self.deinterleave(ext_b) if ext_b is not None else None
            out_a, ext_a = self.decoder_a(emb_nat, ext_for_a)

            ext_for_b = self.interleave(ext_a)
            out_b, ext_b = self.decoder_b(emb_int, ext_for_b)

            w = torch.softmax(self.iter_weights[it], dim=0)
            ctx = w[0] * out_a.mean(1, keepdim=True) + w[1] * out_b.mean(1, keepdim=True)
            contexts.append(ctx)

        # Weighted final context (full channel representations, not just pooled)
        # Use last iteration's full output for cross-attention
        channel_context = out_a  # (1, n_channels, embed_dim)

        # Cross-attention: positions query channel context
        pos = self.pos_embed  # (1, seq_len, embed_dim)
        attended, _ = self.cross_attn(pos, channel_context, channel_context)
        pos_context = self.cross_norm(pos + attended)  # (1, seq_len, embed_dim)

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
# Improved FDM-Struct Write Head (no bottleneck)
# ============================================================

class FDMStructWriteHeadV2(nn.Module):
    """
    FDM-structured write head with per-position projection.
    
    Generates differentiable DSB-SC signal, then uses cross-attention
    between position embeddings and signal features (rather than
    pooling the signal to a single vector).
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

        # Fixed carriers
        freqs = torch.arange(1, n_channels + 1, dtype=torch.float32)
        t = torch.linspace(0, n_samples / fs, n_samples)
        carriers = torch.sin(2 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0))
        self.register_buffer("carriers", carriers)

        # S-random interleaver on time samples
        S = int(math.sqrt(n_samples / 2))
        s_perm = generate_s_random_interleaver(n_samples, S, seed=43)
        self.register_buffer("s_perm", torch.tensor(s_perm, dtype=torch.long))

        # Learned amplitudes per (channel, value)
        self.n_bits = 3
        self.amp_embed = nn.Embedding(n_channels * max_values, self.n_bits)
        with torch.no_grad():
            self.amp_embed.weight.fill_(0.5)

        # Signal → feature sequence (not pooled!)
        # Break dual signal (512 samples) into chunks, embed each chunk
        self.chunk_size = 16  # 512 / 16 = 32 feature positions
        n_chunks = (2 * n_samples) // self.chunk_size
        self.signal_chunk_proj = nn.Linear(self.chunk_size, embed_dim)
        self.signal_pos_embed = nn.Parameter(
            torch.randn(1, n_chunks, embed_dim) * 0.02
        )

        # Signal feature self-attention
        sig_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.signal_attn = nn.TransformerEncoder(sig_layer, num_layers=2)

        # Cross-attention: KV positions attend to signal features
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Per-position KV projection
        kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, kv_per_pos),
        )

    def forward(self, ch_ids, val_ids, device):
        n_ch = len(ch_ids)
        max_values = self.amp_embed.num_embeddings // self.n_channels
        flat_idx = torch.tensor(
            [ch * max_values + val for ch, val in zip(ch_ids, val_ids)],
            device=device
        )
        amps = self.amp_embed(flat_idx)

        # Generate composite signal
        samples_per_bit = self.n_samples // self.n_bits
        composite = torch.zeros(1, self.n_samples, device=device)
        for k in range(n_ch):
            carrier = self.carriers[k]
            for bi in range(self.n_bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.n_samples)
                composite[0, start:end] += amps[k, bi] * carrier[start:end]

        # Dual encoding
        interleaved = composite[:, self.s_perm]
        dual = torch.cat([composite, interleaved], dim=1)  # (1, 512)

        # Chunk the signal into feature sequence
        n_chunks = dual.shape[1] // self.chunk_size
        chunks = dual.view(1, n_chunks, self.chunk_size)  # (1, 32, 16)
        signal_features = self.signal_chunk_proj(chunks)   # (1, 32, embed_dim)
        signal_features = signal_features + self.signal_pos_embed
        signal_features = self.signal_attn(signal_features)  # (1, 32, embed_dim)

        # Cross-attention: positions attend to signal features
        pos = self.pos_embed  # (1, seq_len, embed_dim)
        attended, _ = self.cross_attn(pos, signal_features, signal_features)
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
# Curriculum with extended training
# ============================================================

CURRICULUM = [
    {"name": "5 channels",  "channels": list(range(8, 13)),  "lr_mult": 1.0},
    {"name": "10 channels", "channels": list(range(8, 18)),  "lr_mult": 0.8},
    {"name": "20 channels", "channels": list(range(8, 28)),  "lr_mult": 0.6},
    {"name": "32 channels", "channels": list(range(8, 40)),  "lr_mult": 0.4},
]


def train_head(model, tokenizer, write_head, device, name,
               total_steps, lr, eval_every, n_eval):
    steps_per_stage = total_steps // len(CURRICULUM)
    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine annealing
    warmup_steps = min(500, steps_per_stage // 10)

    global_step = 0
    t0 = time.time()
    stage_results = []

    for stage_idx, stage in enumerate(CURRICULUM):
        active = stage["channels"]
        stage_lr = lr * stage["lr_mult"]

        # Reset scheduler per stage with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps_per_stage - warmup_steps,
            eta_min=stage_lr / 20
        )

        for pg in optimizer.param_groups:
            pg['lr'] = stage_lr

        print(f"\n  [{name}] Stage {stage_idx}: {stage['name']} | "
              f"LR: {stage_lr:.1e} | Steps: {steps_per_stage}")

        losses = []
        write_head.train()

        for step in range(1, steps_per_stage + 1):
            global_step += 1

            # Linear warmup
            if step <= warmup_steps:
                warmup_lr = stage_lr * step / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr
            else:
                scheduler.step()

            mem = random_memory()
            loss = e2e_train_step(model, tokenizer, write_head, mem,
                                  active, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if step % 1000 == 0 or step == steps_per_stage:
                avg = np.mean(losses[-1000:])
                elapsed = time.time() - t0
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"    Step {step:6d}/{steps_per_stage} (global {global_step}) | "
                      f"loss {avg:.4f} | lr {cur_lr:.1e} | {elapsed:.0f}s")

            if eval_every > 0 and global_step % eval_every == 0:
                acc, _ = evaluate_head(model, tokenizer, write_head,
                                       active, device, n_eval=n_eval)
                print(f"    *** Eval ({len(active)} ch): {acc*100:.1f}% ***")
                write_head.train()

        # Stage final eval
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
    parser.add_argument("--n_steps", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--turbo_iters", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=384)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16
    ).to(device)
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

    print(f"  Seq len: {seq_len}, embed_dim: {args.embed_dim}")
    print(f"  Training: {args.n_steps} total steps ({args.n_steps // 4} per stage)")

    # Create both heads
    turbo = TurboWriteHeadV2(
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len,
        n_channels=NUM_CHANNELS, max_values=max_values,
        embed_dim=args.embed_dim, n_attn_layers=2,
        turbo_iters=args.turbo_iters,
    ).to(device)

    fdm_struct = FDMStructWriteHeadV2(
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len,
        n_channels=NUM_CHANNELS, max_values=max_values,
        embed_dim=args.embed_dim, n_samples=256, fs=100.0,
    ).to(device)

    tp = sum(p.numel() for p in turbo.parameters())
    fp = sum(p.numel() for p in fdm_struct.parameters())
    print(f"  Turbo V2:      {tp:,} ({tp/1e6:.1f}M)")
    print(f"  FDM-Struct V2: {fp:,} ({fp/1e6:.1f}M)")

    # Train both
    print("\n" + "=" * 60)
    print("1/2 TRAINING: Turbo V2 (cross-attention, no bottleneck)")
    print("=" * 60)
    turbo_results = train_head(model, tokenizer, turbo, device,
                                "Turbo-V2", args.n_steps, args.lr,
                                args.eval_every, args.n_eval)

    print("\n" + "=" * 60)
    print("2/2 TRAINING: FDM-Struct V2 (signal chunks + cross-attention)")
    print("=" * 60)
    fdm_results = train_head(model, tokenizer, fdm_struct, device,
                              "FDM-V2", args.n_steps, args.lr,
                              args.eval_every, args.n_eval)

    # Comparison
    print("\n" + "=" * 60)
    print("EXTENDED TRAINING RESULTS")
    print("=" * 60)
    print(f"\n  {'Stage':<15s} {'Turbo-V2':>10s} {'FDM-V2':>10s}")
    print(f"  {'-'*40}")
    for tr, fr in zip(turbo_results, fdm_results):
        print(f"  {tr['channels']:2d} ch        "
              f"{tr['acc']*100:9.1f}% {fr['acc']*100:9.1f}%")

    print(f"\n  Previous best (8K steps):")
    print(f"    Turbo:      39.2% at 32ch")
    print(f"    FDM-Struct: 36.2% at 32ch")
    print(f"\n  Reference:")
    print(f"    FDM in context: 93.7%")
    print(f"    Frozen KV:      89.1%")

    results = {
        "turbo_v2": {"params": tp, "stages": turbo_results},
        "fdm_struct_v2": {"params": fp, "stages": fdm_results},
        "config": {
            "n_steps": args.n_steps, "embed_dim": args.embed_dim,
            "turbo_iters": args.turbo_iters, "lr": args.lr,
        },
        "reference": {
            "baseline": 0.937, "frozen_kv": 0.891,
            "prev_turbo_32ch": 0.392, "prev_fdm_32ch": 0.362,
        },
    }
    json.dump(results, open("fdm_extended_results.json", "w"), indent=2, default=float)

    torch.save(turbo.state_dict(), "turbo_v2_write_head.pt")
    torch.save(fdm_struct.state_dict(), "fdm_struct_v2_write_head.pt")
    print("\nSaved checkpoints and results")


if __name__ == "__main__":
    main()
