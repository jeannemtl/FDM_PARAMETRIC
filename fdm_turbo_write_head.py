"""
FDM Turbo Write Head: Iterative Soft-Decision KV Generation
============================================================

Applies the full turbo code architecture to the write head:
  - Two component decoders (A and B) 
  - S-random interleaver between them
  - Iterative soft-information exchange (2-3 turbo iterations)
  - Each decoder refines its KV estimate using the other's confidence

Turbo encoding (already in your paper):
  Encoder 1: signal in natural order → 256 tokens
  Encoder 2: same signal, S-random permuted → 256 tokens

Turbo decoding (new — for the write head):
  Decoder A: processes channels in natural order → preliminary KV + confidence
  Decoder B: processes channels in S-random order + A's confidence → refined KV + confidence  
  Decoder A: re-processes with B's confidence → further refined KV
  ... iterate 2-3 times, then combine

The key insight from Berrou 1993: each component decoder produces
"extrinsic information" — what it learned that the OTHER decoder
didn't already know. By exchanging only extrinsic info (not the
full estimate), the decoders avoid reinforcing each other's errors
and converge on the correct solution.

For KV generation: each decoder produces a KV estimate plus a
per-position confidence score. The confidence from decoder A tells
decoder B "I'm sure about these positions, uncertain about those."
Decoder B focuses its capacity on the uncertain positions, producing
complementary information. After 2-3 iterations, the combined KV
is better than either decoder alone.

Usage:
    # Compare turbo vs non-turbo (same total params)
    python fdm_turbo_write_head.py --n_steps 8000 --turbo_iters 3
    
    # Quick test
    python fdm_turbo_write_head.py --n_steps 4000 --turbo_iters 2 --n_eval 20
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


def score_extra_channels(text, memory):
    results = {}
    for k in range(8, NUM_CHANNELS):
        name = CHANNEL_NAMES[k]
        expected = memory[k]
        pattern = rf"\b{re.escape(name)}={re.escape(expected)}\b"
        results[name] = bool(re.search(pattern, text))
    return results


# ============================================================
# S-Random Interleaver (matching the paper's encoder)
# ============================================================

def generate_s_random_interleaver(length, S, seed=42):
    """Generate S-random permutation for channel ordering."""
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
# Component Decoder (one half of the turbo pair)
# ============================================================

class ComponentDecoder(nn.Module):
    """
    One component decoder in the turbo pair.
    
    Takes: channel embeddings (in its own ordering) + extrinsic info from partner
    Produces: KV contribution + confidence (extrinsic info for partner)
    
    The extrinsic information is the key turbo concept:
    it represents what THIS decoder learned that the OTHER didn't know.
    """

    def __init__(self, n_channels, embed_dim, n_attn_layers=2):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim

        # Process channels with self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.channel_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=n_attn_layers
        )

        # Fuse extrinsic info from partner decoder
        self.extrinsic_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.extrinsic_fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

        # Produce context vector + confidence
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.confidence_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),  # confidence in [0, 1]
        )

    def forward(self, channel_embs, extrinsic_in=None):
        """
        channel_embs: (1, n_channels, embed_dim) — in this decoder's ordering
        extrinsic_in: (1, n_channels, embed_dim) or None — from partner decoder
        
        Returns:
            context: (1, 1, embed_dim) — pooled representation
            confidence: (1, n_channels, embed_dim) — per-channel confidence
            extrinsic_out: (1, n_channels, embed_dim) — info for partner
        """
        x = channel_embs

        # Fuse extrinsic info from partner (if available)
        if extrinsic_in is not None:
            gate = self.extrinsic_gate(torch.cat([x, extrinsic_in], dim=-1))
            fused = self.extrinsic_fuse(torch.cat([x, extrinsic_in], dim=-1))
            x = x + gate * fused  # gated residual fusion

        # Self-attention over channels
        x = self.channel_attn(x)

        # Context (pooled) and confidence
        context = self.context_proj(x.mean(dim=1, keepdim=True))
        confidence = self.confidence_proj(x)

        # Extrinsic info = what this decoder learned beyond the input
        # (the residual between output and input, gated by confidence)
        extrinsic_out = confidence * (x - channel_embs)

        return context, confidence, extrinsic_out


# ============================================================
# Turbo Write Head
# ============================================================

class TurboWriteHead(nn.Module):
    """
    Turbo-decoded write head with iterative soft-decision exchange.
    
    Two component decoders process channels in different orderings
    (natural and S-random permuted), exchanging extrinsic information
    iteratively. The combined output produces KV cache entries.
    
    Architecture mirrors turbo codes:
      - Parallel concatenated decoders (not serial)
      - S-random interleaver breaks error correlation
      - Iterative refinement converges on correct KV
      - Extrinsic info exchange (not full estimates) prevents
        decoders from reinforcing each other's errors
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len,
                 n_channels=40, max_values=5, embed_dim=256,
                 n_attn_layers=2, turbo_iters=3):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.turbo_iters = turbo_iters
        self.embed_dim = embed_dim

        # Shared channel/value embeddings (both decoders see same raw input)
        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)

        # S-random interleaver for channel ordering
        S = int(math.sqrt(n_channels / 2))
        self.interleaver = generate_s_random_interleaver(n_channels, S, seed=42)
        self.deinterleaver = [0] * n_channels
        for i, j in enumerate(self.interleaver):
            self.deinterleaver[j] = i

        # Two component decoders
        self.decoder_a = ComponentDecoder(n_channels, embed_dim, n_attn_layers)
        self.decoder_b = ComponentDecoder(n_channels, embed_dim, n_attn_layers)

        # Iteration-dependent mixing weights (learned)
        self.iter_weights = nn.Parameter(torch.ones(turbo_iters, 2) * 0.5)

        # KV projection (from combined context to KV cache)
        self.kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, self.kv_per_pos),
        )

    def interleave_channels(self, x):
        """Reorder channels using S-random interleaver. x: (1, n_ch, E)"""
        idx = torch.tensor(self.interleaver, device=x.device)
        return x[:, idx, :]

    def deinterleave_channels(self, x):
        """Reverse S-random interleaving. x: (1, n_ch, E)"""
        idx = torch.tensor(self.deinterleaver, device=x.device)
        return x[:, idx, :]

    def forward(self, ch_ids, val_ids, device):
        """
        Turbo decoding iterations:
          1. Embed channels
          2. Decoder A processes natural order → extrinsic A
          3. Interleave extrinsic A → feed to Decoder B
          4. Decoder B processes permuted order → extrinsic B
          5. Deinterleave extrinsic B → feed back to Decoder A
          6. Repeat for turbo_iters iterations
          7. Combine final contexts → project to KV
        """
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)

        # Raw channel embeddings
        ch_emb = self.ch_embed(ch_t)     # (1, n_ch, E)
        val_emb = self.val_embed(val_t)   # (1, n_ch, E)
        raw_emb = ch_emb + val_emb        # (1, n_ch, E)

        # Natural and interleaved views
        emb_natural = raw_emb
        emb_interleaved = self.interleave_channels(raw_emb)

        # Iterative turbo decoding
        extrinsic_a = None  # No extrinsic info on first iteration
        extrinsic_b = None
        contexts = []

        for it in range(self.turbo_iters):
            # Decoder A: natural order + extrinsic from B (deinterleaved)
            ext_for_a = None
            if extrinsic_b is not None:
                ext_for_a = self.deinterleave_channels(extrinsic_b)

            ctx_a, conf_a, extrinsic_a = self.decoder_a(
                emb_natural, ext_for_a
            )

            # Decoder B: interleaved order + extrinsic from A (interleaved)
            ext_for_b = self.interleave_channels(extrinsic_a)

            ctx_b, conf_b, extrinsic_b = self.decoder_b(
                emb_interleaved, ext_for_b
            )

            # Weighted combination of contexts for this iteration
            w = torch.softmax(self.iter_weights[it], dim=0)
            combined_ctx = w[0] * ctx_a + w[1] * ctx_b
            contexts.append(combined_ctx)

        # Final context: weighted sum across iterations
        # Later iterations should matter more (more refined)
        final_context = torch.zeros_like(contexts[0])
        for it, ctx in enumerate(contexts):
            # Exponential weighting: later iterations count more
            weight = 2.0 ** it
            final_context = final_context + weight * ctx
        final_context = final_context / sum(2.0 ** i for i in range(self.turbo_iters))

        # Project to KV cache
        pos = self.pos_embed  # (1, seq_len, E)
        pos_context = pos + self.context_proj(final_context)  # (1, seq_len, E)
        kv_flat = self.pos_to_kv(pos_context)  # (1, seq_len, kv_per_pos)

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
# Non-Turbo Write Head (same param budget, for fair comparison)
# ============================================================

class PlainWriteHead(nn.Module):
    """
    Same architecture as the E2E write head (no turbo structure).
    Param count matched to TurboWriteHead for fair comparison.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len,
                 n_channels=40, max_values=5, embed_dim=256, n_attn_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.n_channels = n_channels

        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.channel_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=n_attn_layers
        )

        self.kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.channel_to_pos = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, self.kv_per_pos),
        )

    def forward(self, ch_ids, val_ids, device):
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)
        ch_emb = self.ch_embed(ch_t)
        val_emb = self.val_embed(val_t)
        combined = ch_emb + val_emb
        coupled = self.channel_attn(combined)
        context = coupled.mean(dim=1, keepdim=True)
        pos = self.pos_embed
        pos_context = pos + self.channel_to_pos(context)
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
# E2E Training (shared by both architectures)
# ============================================================

def build_answer_for_channels(memory, active_channels):
    parts = [f"{CHANNEL_NAMES[k]}={memory[k]}" for k in active_channels]
    return "Context: " + ", ".join(parts) + "."


def build_question_for_channels(active_channels):
    ch_names = [CHANNEL_NAMES[k] for k in active_channels]
    return f"Report values for: {', '.join(ch_names)}."


def e2e_train_step(model, tokenizer, write_head, memory,
                   active_channels, device):
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
        shift_labels.view(-1), ignore_index=-100,
    )
    return loss


def generate_with_head(model, tokenizer, write_head, memory,
                       active_channels, device):
    write_head.eval()
    ch_ids, val_ids = memory_to_indices(memory)

    with torch.no_grad():
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

    with torch.no_grad():
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


def evaluate_head(model, tokenizer, write_head, active_channels,
                  device, n_eval=30):
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
# Curriculum
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

    global_step = 0
    t0 = time.time()
    stage_results = []

    for stage_idx, stage in enumerate(CURRICULUM):
        active = stage["channels"]
        stage_lr = lr * stage["lr_mult"]
        for pg in optimizer.param_groups:
            pg['lr'] = stage_lr

        print(f"\n  [{name}] Stage {stage_idx}: {stage['name']} | LR: {stage_lr:.1e}")

        losses = []
        write_head.train()

        for step in range(1, steps_per_stage + 1):
            global_step += 1
            mem = random_memory()
            loss = e2e_train_step(model, tokenizer, write_head, mem,
                                  active, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if step % 500 == 0 or step == steps_per_stage:
                avg = np.mean(losses[-500:])
                elapsed = time.time() - t0
                print(f"    Step {step:5d}/{steps_per_stage} | "
                      f"loss {avg:.4f} | {elapsed:.0f}s")

            if eval_every > 0 and global_step % eval_every == 0:
                acc, _ = evaluate_head(model, tokenizer, write_head,
                                       active, device, n_eval=n_eval)
                print(f"    *** Eval ({len(active)} ch): {acc*100:.1f}% ***")
                write_head.train()

        # Stage final eval
        acc, per_ch = evaluate_head(model, tokenizer, write_head,
                                     active, device, n_eval=n_eval)
        print(f"    Stage {stage_idx} final: {acc*100:.1f}%")
        stage_results.append({"stage": stage_idx, "channels": len(active), "acc": acc})

    return stage_results


# ============================================================
# Main: Head-to-head comparison
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--turbo_iters", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--n_eval", type=int, default=30)
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

    print(f"  Seq len: {seq_len}, KV heads: {n_kv_heads}, head dim: {head_dim}")

    # ---- Create both heads ----

    turbo_head = TurboWriteHead(
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len,
        n_channels=NUM_CHANNELS, max_values=max_values,
        embed_dim=256, n_attn_layers=2, turbo_iters=args.turbo_iters,
    ).to(device)

    # Plain head with MORE attn layers to match param count
    plain_head = PlainWriteHead(
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len,
        n_channels=NUM_CHANNELS, max_values=max_values,
        embed_dim=256, n_attn_layers=4,  # more layers to match params
    ).to(device)

    turbo_params = sum(p.numel() for p in turbo_head.parameters())
    plain_params = sum(p.numel() for p in plain_head.parameters())
    print(f"\n  Turbo write head: {turbo_params:,} params ({turbo_params/1e6:.1f}M)")
    print(f"  Plain write head: {plain_params:,} params ({plain_params/1e6:.1f}M)")
    print(f"  Turbo iterations: {args.turbo_iters}")

    # ---- Train both ----

    print("\n" + "=" * 60)
    print("TRAINING: Plain write head (baseline)")
    print("=" * 60)
    plain_results = train_head(model, tokenizer, plain_head, device,
                                "Plain", args.n_steps, args.lr,
                                args.eval_every, args.n_eval)

    print("\n" + "=" * 60)
    print("TRAINING: Turbo write head (S-interleaved iterative)")
    print("=" * 60)
    turbo_results = train_head(model, tokenizer, turbo_head, device,
                                "Turbo", args.n_steps, args.lr,
                                args.eval_every, args.n_eval)

    # ---- Final comparison ----

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    print(f"\n  {'Stage':<20s} {'Plain':>10s} {'Turbo':>10s} {'Delta':>10s}")
    print(f"  {'-'*50}")
    for pr, tr in zip(plain_results, turbo_results):
        delta = tr['acc'] - pr['acc']
        print(f"  {pr['channels']:2d} channels        "
              f"{pr['acc']*100:9.1f}% {tr['acc']*100:9.1f}% {delta*100:+9.1f}%")

    print(f"\n  Plain params:  {plain_params:,}")
    print(f"  Turbo params:  {turbo_params:,}")
    print(f"  Turbo iters:   {args.turbo_iters}")
    print(f"\n  Baseline (FDM in context): 93.7%")
    print(f"  Frozen KV injection:       89.1%")

    # Save
    results = {
        "plain": {"params": plain_params, "stages": plain_results},
        "turbo": {"params": turbo_params, "stages": turbo_results,
                  "turbo_iters": args.turbo_iters},
        "reference": {"baseline": 0.937, "frozen_kv": 0.891},
    }
    json.dump(results, open("fdm_turbo_comparison.json", "w"), indent=2, default=float)
    print(f"\nSaved to fdm_turbo_comparison.json")

    # Save checkpoints
    torch.save(plain_head.state_dict(), "plain_write_head.pt")
    torch.save(turbo_head.state_dict(), "turbo_write_head.pt")
    print("Saved checkpoints: plain_write_head.pt, turbo_write_head.pt")


if __name__ == "__main__":
    main()
