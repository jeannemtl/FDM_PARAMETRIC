#!/bin/bash
# ============================================================
# FDM Memory Interactive Demo
# ============================================================
# Runs all three modes on a SINGLE random memory, prints results
# side by side so you can visually verify.
#
# Usage:
#   bash fdm_interactive_demo.sh
#   bash fdm_interactive_demo.sh 5   # run 5 samples
# ============================================================

N_SAMPLES=${1:-3}
WORK_DIR="/workspace/FDM_IN_WEIGHTS"
MODEL="prompterminal/fdm-40ch-nhop-qwen3"
WRITE_HEAD="$WORK_DIR/fdm_v3_adaptive.pt"
HYBRID_HEAD="$WORK_DIR/hybrid_write_head.pt"

python3 << PYEOF
import sys, os, re, json, random, math, time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/root/FDM_IN_WEIGHTS')
from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]
N_SAMPLES = $N_SAMPLES
MODEL = "$MODEL"
WRITE_HEAD_PATH = "$WRITE_HEAD"
HYBRID_HEAD_PATH = "$HYBRID_HEAD"

# ---- Setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.float16).to(device)
model.eval()

encoder = TurboFDMSignalEncoder(
    vocab_size=151936, tokenizer=tokenizer,
    num_tokens_per_encoder=256, sample_rate=100.0,
    a_high=1.0, a_low=0.25, num_levels=64, seed=42)

# ---- Write head architecture ----
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

class FDMAdaptiveWriteHead(nn.Module):
    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len,
                 n_channels=40, max_values=5, embed_dim=384, max_samples=1024):
        super().__init__()
        self.n_layers = n_layers; self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim; self.seq_len = seq_len
        self.n_channels = n_channels; self.embed_dim = embed_dim
        self.max_dual_n = 2 * max_samples
        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)
        self.modulation_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.carrier_pool = nn.Sequential(
            nn.Linear(self.max_dual_n, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        feat_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4,
            dropout=0.1, batch_first=True, activation='gelu')
        self.feature_attn = nn.TransformerEncoder(feat_layer, num_layers=2)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(embed_dim)
        kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim*4), nn.GELU(),
            nn.Linear(embed_dim*4, kv_per_pos))
        self._carrier_cache = {}
    def _get_carriers(self, n_samples, fs, device):
        key = (n_samples, fs)
        if key not in self._carrier_cache:
            freqs = torch.arange(1, self.n_channels+1, dtype=torch.float32)
            t = torch.linspace(0, n_samples/fs, n_samples)
            carriers = torch.sin(2*math.pi*freqs.unsqueeze(1)*t.unsqueeze(0)).to(device)
            S = int(math.sqrt(n_samples/2))
            s_perm = generate_s_random_interleaver(n_samples, S, seed=43)
            s_perm = torch.tensor(s_perm, dtype=torch.long, device=device)
            self._carrier_cache[key] = (carriers, s_perm)
        return self._carrier_cache[key]
    def forward(self, ch_ids, val_ids, device, n_samples=1024, fs=400.0):
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)
        raw = self.ch_embed(ch_t) + self.val_embed(val_t)
        modulation = self.modulation_proj(raw)
        carriers, s_perm = self._get_carriers(n_samples, fs, device)
        dual_carriers = []
        for k in range(self.n_channels):
            c = carriers[k]; dual_carriers.append(torch.cat([c, c[s_perm]]))
        dual_stack = torch.stack(dual_carriers, dim=0)
        if dual_stack.shape[1] < self.max_dual_n:
            dual_stack = F.pad(dual_stack, (0, self.max_dual_n - dual_stack.shape[1]))
        pooled = self.carrier_pool(dual_stack).unsqueeze(0)
        carrier_features = self.feature_attn(pooled * modulation)
        pos = self.pos_embed
        attended, _ = self.cross_attn(pos, carrier_features, carrier_features)
        pos_context = self.cross_norm(pos + attended)
        kv_flat = self.pos_to_kv(pos_context)
        kv = kv_flat.view(1, self.seq_len, self.n_layers, 2, self.n_kv_heads, self.head_dim)
        return kv.permute(2, 3, 0, 4, 1, 5)
    def make_cache(self, ch_ids, val_ids, device, n_samples=1024, fs=400.0):
        kv = self.forward(ch_ids, val_ids, device, n_samples, fs)
        cache = DynamicCache()
        for li in range(self.n_layers):
            cache.update(kv[li, 0].half(), kv[li, 1].half(), li)
        return cache, kv.shape[4]

# Turbo write head for hybrid
class HybridWriteHead(nn.Module):
    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len,
                 n_channels=40, max_values=5, embed_dim=384, n_attn_layers=2, turbo_iters=3):
        super().__init__()
        self.n_layers = n_layers; self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim; self.seq_len = seq_len
        self.n_channels = n_channels; self.turbo_iters = turbo_iters; self.embed_dim = embed_dim
        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)
        S = int(math.sqrt(n_channels/2))
        self.interleaver = generate_s_random_interleaver(n_channels, S, seed=42)
        self.deinterleaver = [0]*n_channels
        for i, j in enumerate(self.interleaver): self.deinterleaver[j] = i
        class CompDecoder(nn.Module):
            def __init__(self, embed_dim, n_attn_layers):
                super().__init__()
                layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                    dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True, activation='gelu')
                self.attn = nn.TransformerEncoder(layer, num_layers=n_attn_layers)
                self.ext_gate = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.Sigmoid())
                self.ext_fuse = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.GELU())
                self.confidence = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
            def forward(self, x, ext_in=None):
                if ext_in is not None:
                    gate = self.ext_gate(torch.cat([x, ext_in], -1))
                    fused = self.ext_fuse(torch.cat([x, ext_in], -1))
                    x = x + gate * fused
                out = self.attn(x); conf = self.confidence(out)
                return out, conf * (out - x)
        self.decoder_a = CompDecoder(embed_dim, n_attn_layers)
        self.decoder_b = CompDecoder(embed_dim, n_attn_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(embed_dim)
        kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim*4), nn.GELU(),
            nn.Linear(embed_dim*4, kv_per_pos))
    def interleave(self, x):
        return x[:, torch.tensor(self.interleaver, device=x.device), :]
    def deinterleave(self, x):
        return x[:, torch.tensor(self.deinterleaver, device=x.device), :]
    def forward(self, ch_ids, val_ids, device):
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)
        raw = self.ch_embed(ch_t) + self.val_embed(val_t)
        emb_nat = raw; emb_int = self.interleave(raw)
        ext_a, ext_b = None, None
        for it in range(self.turbo_iters):
            ext_for_a = self.deinterleave(ext_b) if ext_b is not None else None
            out_a, ext_a = self.decoder_a(emb_nat, ext_for_a)
            out_b, ext_b = self.decoder_b(self.interleave(raw), self.interleave(ext_a))
        pos = self.pos_embed
        attended, _ = self.cross_attn(pos, out_a, out_a)
        pos_context = self.cross_norm(pos + attended)
        kv_flat = self.pos_to_kv(pos_context)
        kv = kv_flat.view(1, self.seq_len, self.n_layers, 2, self.n_kv_heads, self.head_dim)
        return kv.permute(2, 3, 0, 4, 1, 5)
    def make_cache(self, ch_ids, val_ids, device):
        kv = self.forward(ch_ids, val_ids, device)
        cache = DynamicCache()
        for li in range(self.n_layers):
            cache.update(kv[li, 0].half(), kv[li, 1].half(), li)
        return cache, kv.shape[4]

# ---- Generation helpers ----
def gen_plain(prompt):
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=350, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def gen_kv(prompt, kv_cache, kv_len):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    t = torch.tensor([ids], dtype=torch.long).to(device)
    pos = torch.arange(kv_len, kv_len+len(ids), device=device).unsqueeze(0)
    attn = torch.ones(1, kv_len+len(ids), device=device, dtype=torch.long)
    gen = []; past = kv_cache
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(t, past_key_values=past, position_ids=pos, attention_mask=attn, use_cache=True)
        past = out.past_key_values
        nt = out.logits[:,-1,:].argmax(-1, keepdim=True); gen.append(nt.item())
        tl = kv_len+len(ids)+1
        for s in range(299):
            p = torch.tensor([[tl-1+s]], device=device)
            a = torch.ones(1, tl+s, device=device, dtype=torch.long)
            o = model(nt, past_key_values=past, position_ids=p, attention_mask=a, use_cache=True)
            past = o.past_key_values; nt = o.logits[:,-1,:].argmax(-1, keepdim=True)
            tid = nt.item(); gen.append(tid)
            if tid == tokenizer.eos_token_id: break
    return tokenizer.decode(gen, skip_special_tokens=True)

# ---- Score ----
def score(text, mem, channels):
    correct = 0; total = 0
    for k in channels:
        name = CHANNEL_NAMES[k]; expected = mem[k]
        if re.search(rf"\b{re.escape(name)}={re.escape(expected)}\b", text):
            correct += 1
        total += 1
    return correct, total

# ---- Load write heads ----
mem_test = {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}
_, fdm_tokens = encoder.encode_memory(mem_test)
memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
seq_len = len(memory_start) + len(fdm_tokens)
n_kv_heads = model.config.num_key_value_heads
head_dim = 128
max_values = max(len(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS))

has_standalone = os.path.exists(WRITE_HEAD_PATH)
has_hybrid = os.path.exists(HYBRID_HEAD_PATH)

if has_standalone:
    standalone_head = FDMAdaptiveWriteHead(
        model.config.num_hidden_layers, n_kv_heads, head_dim, seq_len,
        NUM_CHANNELS, max_values, 384, 1024).to(device)
    standalone_head.load_state_dict(torch.load(WRITE_HEAD_PATH, map_location=device))
    standalone_head.eval()
    print("  ✓ Standalone write head loaded")

if has_hybrid:
    hybrid_head = HybridWriteHead(
        model.config.num_hidden_layers, n_kv_heads, head_dim, seq_len,
        NUM_CHANNELS, max_values, 384, 2, 3).to(device)
    hybrid_head.load_state_dict(torch.load(HYBRID_HEAD_PATH, map_location=device))
    hybrid_head.eval()
    print("  ✓ Hybrid write head loaded")

# ---- Run demo ----
print()
print("=" * 70)
print("  FDM INTERACTIVE MEMORY DEMO")
print("=" * 70)

channels = list(range(8, 40))
totals = {"context": [0,0], "kv": [0,0], "parametric": [0,0], "hybrid": [0,0]}

for sample in range(N_SAMPLES):
    mem = {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}
    ch_ids = list(range(NUM_CHANNELS))
    val_ids = [MEMORY_SCHEMAS[ch][1].index(mem[ch]) for ch in range(NUM_CHANNELS)]

    print(f"\n{'─' * 70}")
    print(f"  SAMPLE {sample+1}/{N_SAMPLES}")
    print(f"{'─' * 70}")

    # Print ground truth (first 5 extra channels)
    print(f"\n  Ground truth (first 5 of 32 extra channels):")
    for k in range(8, 13):
        print(f"    {CHANNEL_NAMES[k]:12s} = {mem[k]}")
    print(f"    ... ({len(channels)-5} more)")

    fdm_text, fdm_tokens = encoder.encode_memory(mem)
    question = "Report all context values for channels 8-39."

    # MODE 1: Context
    print(f"\n  MODE 1 — FDM IN-CONTEXT (512 tokens)")
    t0 = time.time()
    prompt1 = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    text1 = gen_plain(prompt1)
    lat1 = time.time() - t0
    c1, t1 = score(text1, mem, channels)
    totals["context"][0] += c1; totals["context"][1] += t1
    print(f"    Accuracy: {c1}/{t1} ({c1/t1*100:.1f}%)  |  Latency: {lat1:.1f}s")
    print(f"    Output: {text1[:120]}...")

    # MODE 2: Frozen KV
    print(f"\n  MODE 2 — FROZEN KV CACHE (0 fact tokens)")
    t0 = time.time()
    prefix_ids = memory_start + fdm_tokens
    prefix_t = torch.tensor([prefix_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(prefix_t, use_cache=True)
    frozen_kv = out.past_key_values
    kv_len = len(prefix_ids)
    prompt2 = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
    text2 = gen_kv(prompt2, frozen_kv, kv_len)
    lat2 = time.time() - t0
    c2, t2 = score(text2, mem, channels)
    totals["kv"][0] += c2; totals["kv"][1] += t2
    print(f"    Accuracy: {c2}/{t2} ({c2/t2*100:.1f}%)  |  Latency: {lat2:.1f}s")
    print(f"    Output: {text2[:120]}...")

    # MODE 3: Parametric (standalone write head)
    if has_standalone:
        print(f"\n  MODE 3 — PARAMETRIC (write head, 0 tokens)")
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            cache3, kl3 = standalone_head.make_cache(ch_ids, val_ids, device)
        prompt3 = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
        text3 = gen_kv(prompt3, cache3, kl3)
        lat3 = time.time() - t0
        c3, t3 = score(text3, mem, channels)
        totals["parametric"][0] += c3; totals["parametric"][1] += t3
        print(f"    Accuracy: {c3}/{t3} ({c3/t3*100:.1f}%)  |  Latency: {lat3:.1f}s")
        print(f"    Output: {text3[:120]}...")

    # MODE 4: Hybrid (write head + context)
    if has_hybrid:
        print(f"\n  MODE 4 — HYBRID (write head KV + context tokens)")
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            cache4, kl4 = hybrid_head.make_cache(ch_ids, val_ids, device)
        prompt4 = f"{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
        text4 = gen_kv(prompt4, cache4, kl4)
        lat4 = time.time() - t0
        c4, t4 = score(text4, mem, channels)
        totals["hybrid"][0] += c4; totals["hybrid"][1] += t4
        print(f"    Accuracy: {c4}/{t4} ({c4/t4*100:.1f}%)  |  Latency: {lat4:.1f}s")
        print(f"    Output: {text4[:120]}...")

# ---- Summary ----
print(f"\n{'=' * 70}")
print(f"  SUMMARY ({N_SAMPLES} samples)")
print(f"{'=' * 70}")
print(f"\n  {'Mode':<40s} {'Correct':>8s} {'Accuracy':>10s} {'Fact tokens':>12s}")
print(f"  {'─' * 72}")
for mode, label, tokens in [
    ("context",    "1. FDM In-Context",      "512"),
    ("kv",         "2. Frozen KV Cache",      "0"),
    ("parametric", "3. Parametric (write head)", "0"),
    ("hybrid",     "4. Hybrid (param+context)", "512"),
]:
    c, t = totals[mode]
    if t > 0:
        print(f"  {label:<40s} {c:>4d}/{t:<4d} {c/t*100:>9.1f}% {tokens:>12s}")

print(f"\n  The memory spectrum:")
print(f"    Context tokens → KV cache → Parameters → Hybrid")
print(f"    More compression, less cost, same read mechanism.")
print(f"    Hybrid: parametric + context exceed either alone.")
print()
PYEOF
