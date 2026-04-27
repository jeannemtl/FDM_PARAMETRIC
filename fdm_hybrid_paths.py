"""
FDM Hybrid: Two Paths
======================

Path 1 — FROZEN KV HYBRID (no write head, no training needed):
  Compute KV cache from FDM tokens encoding memory_A
  Put FDM tokens encoding memory_B in context
  Test: can the model read from both KV sources?
  This isolates FDM composability from write head issues.

Path 2 — HYBRID-AWARE WRITE HEAD (needs training):
  Train the write head with FDM context tokens PRESENT during training.
  The write head learns to produce KV entries that complement (not 
  duplicate) the context signal.

Both paths test the same question: can two FDM sources compose?

Usage:
    # Path 1 only (fast, no training)
    python fdm_hybrid_paths.py --path 1 --n_eval 50

    # Path 2 only (needs training)  
    python fdm_hybrid_paths.py --path 2 --n_train 10000 --n_eval 50

    # Both
    python fdm_hybrid_paths.py --path both --n_train 10000 --n_eval 50
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

def score_channels(text, memory, channel_range):
    results = {}
    for k in channel_range:
        name = CHANNEL_NAMES[k]
        expected = memory[k]
        pattern = rf"\b{re.escape(name)}={re.escape(expected)}\b"
        results[name] = bool(re.search(pattern, text))
    return results


# ============================================================
# Generation with KV cache
# ============================================================

def generate_with_kv(model, tokenizer, prompt_text, kv_cache, kv_seq_len,
                     device, max_tokens=350):
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    position_ids = torch.arange(kv_seq_len, kv_seq_len + len(prompt_ids),
                                 device=device).unsqueeze(0)
    attn_mask = torch.ones(1, kv_seq_len + len(prompt_ids),
                            device=device, dtype=torch.long)
    generated_ids = []
    past_kv = kv_cache

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(prompt_tensor, past_key_values=past_kv,
                   position_ids=position_ids, attention_mask=attn_mask,
                   use_cache=True)
        past_kv = out.past_key_values
        nt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated_ids.append(nt.item())
        tl = kv_seq_len + len(prompt_ids) + 1

        for s in range(max_tokens - 1):
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

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(prompt_ids)


def generate_plain(model, tokenizer, prompt, device, max_tokens=350):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text, input_ids.shape[1]


# ============================================================
# PATH 1: Frozen KV Hybrid
# ============================================================

def run_path1(model, tokenizer, encoder, n_eval, device):
    """
    Pure FDM composability test using frozen KV (no write head).
    
    Method:
      1. Encode memory into FDM tokens (512 tokens)
      2. Process the FULL token sequence through model: 
         [MEMORY] + fdm_tokens
      3. Freeze the KV cache from step 2
      4. Now run AGAIN with the SAME FDM tokens in context,
         prepended with the frozen KV
      5. The model sees the FDM info TWICE: once from KV, once from context
    
    This tests whether frozen KV + context FDM compose.
    
    Then the real test:
      A. Context only: [MEMORY] fdm_text [/MEMORY] Question → answer
      B. Frozen KV only: KV from fdm_tokens + [/MEMORY] Question → answer  
      C. Frozen KV + Context: KV from fdm_tokens_A + fdm_text_B [/MEMORY] Question
         where A and B encode the SAME memory → redundancy test
    """
    print("\n" + "=" * 70)
    print("  PATH 1: Frozen KV Hybrid (no write head)")
    print("=" * 70)

    # Accumulators
    results = {
        "context_only": {n: 0 for n in CHANNEL_NAMES[8:]},
        "kv_only": {n: 0 for n in CHANNEL_NAMES[8:]},
        "kv_plus_context": {n: 0 for n in CHANNEL_NAMES[8:]},
    }
    total = 0

    for i in tqdm(range(n_eval), desc="Path 1"):
        mem = random_memory()
        fdm_text, fdm_tokens = encoder.encode_memory(mem)
        memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)

        # --- A: Context only (baseline) ---
        question = "Report all context values for channels 8-39."
        prompt_a = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        text_a, _ = generate_plain(model, tokenizer, prompt_a, device)
        scores_a = score_channels(text_a, mem, range(8, 40))
        for name, correct in scores_a.items():
            results["context_only"][name] += int(correct)

        # --- B: Frozen KV only ---
        full_prefix_ids = memory_start + fdm_tokens
        prefix_tensor = torch.tensor([full_prefix_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(prefix_tensor, use_cache=True)
        frozen_kv = out.past_key_values
        kv_len = len(full_prefix_ids)

        prompt_b = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
        text_b, _ = generate_with_kv(model, tokenizer, prompt_b,
                                      frozen_kv, kv_len, device)
        scores_b = score_channels(text_b, mem, range(8, 40))
        for name, correct in scores_b.items():
            results["kv_only"][name] += int(correct)

        # --- C: Frozen KV + Context (redundant) ---
        # KV cache has: [MEMORY] + fdm_tokens (positions 0 to kv_len-1)
        # Context has: fdm_text + [/MEMORY] + question
        # The FDM signal appears in BOTH the KV cache and the context
        prompt_c = f"{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        text_c, _ = generate_with_kv(model, tokenizer, prompt_c,
                                      frozen_kv, kv_len, device)
        scores_c = score_channels(text_c, mem, range(8, 40))
        for name, correct in scores_c.items():
            results["kv_plus_context"][name] += int(correct)

        total += 1

    # Compute accuracies
    ctx_acc = np.mean([results["context_only"][n] / total for n in results["context_only"]])
    kv_acc = np.mean([results["kv_only"][n] / total for n in results["kv_only"]])
    both_acc = np.mean([results["kv_plus_context"][n] / total for n in results["kv_plus_context"]])

    print(f"\n  Results:")
    print(f"    A. Context only:         {ctx_acc*100:.1f}%")
    print(f"    B. Frozen KV only:       {kv_acc*100:.1f}%")
    print(f"    C. Frozen KV + Context:  {both_acc*100:.1f}%")

    delta_bc = both_acc - ctx_acc
    print(f"\n    Delta (C vs A): {delta_bc*100:+.1f}%")
    if both_acc > ctx_acc * 0.95:
        print(f"    ✓ FDM sources compose — redundant encoding preserves accuracy")
    elif both_acc > ctx_acc * 0.5:
        print(f"    ~ Partial composition — some interference from double encoding")
    else:
        print(f"    ✗ Sources don't compose — frozen KV disrupts context reading")

    return {"context_only": ctx_acc, "kv_only": kv_acc, "kv_plus_context": both_acc}


# ============================================================
# PATH 2: Hybrid-Aware Write Head Training
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


class HybridWriteHead(nn.Module):
    """
    Write head trained to work alongside FDM context tokens.
    
    Key difference from the standalone write head: during training,
    FDM tokens are ALWAYS present in context. The write head learns
    to produce KV entries that COMPLEMENT the context signal rather
    than duplicate it.
    
    Uses the Turbo V2 architecture (best proven architecture)
    with cross-attention to channel embeddings.
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


def train_hybrid_write_head(model, tokenizer, encoder, write_head,
                             device, n_steps, lr):
    """
    Train write head WITH FDM context tokens present.
    
    Training prompt structure:
      KV cache: write_head(channel_values)
      Context:  {fdm_text}[/MEMORY]\nQuestion: ...\nAnswer: {answer}
    
    The write head learns that FDM tokens will be in context,
    so it should produce COMPLEMENTARY KV entries, not duplicate ones.
    """
    print(f"\n  Training hybrid-aware write head ({n_steps} steps)...")

    model.eval()
    write_head.train()
    for p in model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr / 20)

    t0 = time.time()
    losses = []

    for step in range(1, n_steps + 1):
        mem = random_memory()
        fdm_text, fdm_tokens = encoder.encode_memory(mem)

        # Build answer
        extra_parts = [f"{CHANNEL_NAMES[k]}={mem[k]}" for k in range(8, 40)]
        answer = "Context: " + ", ".join(extra_parts) + "."

        # Build training prompt: FDM tokens in context + question + answer
        question = "Report all context values for channels 8-39."
        q_text = f"{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        a_text = f" {answer}"
        full_text = q_text + a_text

        q_ids = tokenizer.encode(q_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        full_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)

        # Write head produces KV cache
        ch_ids, val_ids = memory_to_indices(mem)
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

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 1000 == 0 or step == n_steps:
            avg = np.mean(losses[-1000:])
            elapsed = time.time() - t0
            print(f"    Step {step:5d}/{n_steps} | loss {avg:.4f} | {elapsed:.0f}s")

    return write_head


def eval_hybrid_write_head(model, tokenizer, encoder, write_head, n_eval, device):
    """Evaluate the hybrid-aware write head with FDM context present."""
    print(f"\n  Evaluating hybrid write head ({n_eval} samples)...")
    write_head.eval()

    results = {
        "hybrid": {n: 0 for n in CHANNEL_NAMES[8:]},
        "context_only": {n: 0 for n in CHANNEL_NAMES[8:]},
    }
    total = 0

    for i in tqdm(range(n_eval), desc="Path 2 eval"):
        mem = random_memory()
        fdm_text, _ = encoder.encode_memory(mem)
        question = "Report all context values for channels 8-39."

        # Context only (baseline)
        prompt_a = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        text_a, _ = generate_plain(model, tokenizer, prompt_a, device)
        scores_a = score_channels(text_a, mem, range(8, 40))
        for name, correct in scores_a.items():
            results["context_only"][name] += int(correct)

        # Hybrid: write head KV + FDM context
        ch_ids, val_ids = memory_to_indices(mem)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            cache, kv_len = write_head.make_cache(ch_ids, val_ids, device)

        prompt_h = f"{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        text_h, _ = generate_with_kv(model, tokenizer, prompt_h, cache, kv_len, device)
        scores_h = score_channels(text_h, mem, range(8, 40))
        for name, correct in scores_h.items():
            results["hybrid"][name] += int(correct)

        total += 1

    ctx_acc = np.mean([results["context_only"][n] / total for n in results["context_only"]])
    hyb_acc = np.mean([results["hybrid"][n] / total for n in results["hybrid"]])

    print(f"\n    Context only:              {ctx_acc*100:.1f}%")
    print(f"    Hybrid (write head + ctx):  {hyb_acc*100:.1f}%")
    print(f"    Delta:                      {(hyb_acc - ctx_acc)*100:+.1f}%")

    if hyb_acc > ctx_acc * 0.95:
        print(f"    ✓ Hybrid-aware write head complements context successfully")
    elif hyb_acc > ctx_acc * 0.5:
        print(f"    ~ Partial success — write head partially complements context")
    else:
        print(f"    ✗ Write head still disrupts context reading")

    return {"context_only": ctx_acc, "hybrid": hyb_acc}


def run_path2(model, tokenizer, encoder, device, n_train, n_eval, lr):
    """Train and evaluate a hybrid-aware write head."""
    print("\n" + "=" * 70)
    print("  PATH 2: Hybrid-Aware Write Head")
    print("=" * 70)

    mem = random_memory()
    _, fdm_tokens = encoder.encode_memory(mem)
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    seq_len = len(memory_start) + len(fdm_tokens)

    n_kv_heads = model.config.num_key_value_heads
    head_dim = 128
    max_values = max(len(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS))

    write_head = HybridWriteHead(
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads, head_dim=head_dim, seq_len=seq_len,
        n_channels=NUM_CHANNELS, max_values=max_values,
        embed_dim=384, n_attn_layers=2, turbo_iters=3,
    ).to(device)

    n_params = sum(p.numel() for p in write_head.parameters())
    print(f"  Hybrid write head: {n_params:,} ({n_params/1e6:.1f}M)")

    write_head = train_hybrid_write_head(
        model, tokenizer, encoder, write_head, device, n_train, lr)

    torch.save(write_head.state_dict(), "hybrid_write_head.pt")

    results = eval_hybrid_write_head(
        model, tokenizer, encoder, write_head, n_eval, device)

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--path", choices=["1", "2", "both"], default="both")
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--n_train", type=int, default=10000,
                        help="Training steps for Path 2 hybrid write head")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  FDM HYBRID: TWO PATHS")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16).to(device)
    model.eval()

    encoder = make_encoder(tokenizer)

    all_results = {}

    if args.path in ("1", "both"):
        p1 = run_path1(model, tokenizer, encoder, args.n_eval, device)
        all_results["path1_frozen_kv"] = p1

    if args.path in ("2", "both"):
        p2 = run_path2(model, tokenizer, encoder, device,
                        args.n_train, args.n_eval, args.lr)
        all_results["path2_hybrid_write_head"] = p2

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    if "path1_frozen_kv" in all_results:
        p1 = all_results["path1_frozen_kv"]
        print(f"\n  PATH 1 — Frozen KV Hybrid (no training):")
        print(f"    Context only:        {p1['context_only']*100:.1f}%")
        print(f"    KV only:             {p1['kv_only']*100:.1f}%")
        print(f"    KV + Context:        {p1['kv_plus_context']*100:.1f}%")

    if "path2_hybrid_write_head" in all_results:
        p2 = all_results["path2_hybrid_write_head"]
        print(f"\n  PATH 2 — Hybrid-Aware Write Head ({args.n_train} steps):")
        print(f"    Context only:        {p2['context_only']*100:.1f}%")
        print(f"    Write head + context:{p2['hybrid']*100:.1f}%")

    print(f"\n  If Path 1 KV+Context ≈ Context only:")
    print(f"    → Frozen FDM KV composes with context FDM tokens")
    print(f"    → The two substrates don't interfere")
    print(f"\n  If Path 2 Hybrid > Context only:")
    print(f"    → Write head learned to COMPLEMENT context")
    print(f"    → Parametric + context FDM is better than either alone")

    json.dump(all_results, open("fdm_hybrid_paths_results.json", "w"),
              indent=2, default=float)
    print(f"\n  Saved to fdm_hybrid_paths_results.json")


if __name__ == "__main__":
    main()
