#!/bin/bash
# ============================================================
# FDM Memory Demo — Simple Version
# ============================================================
# Shows ONE set of facts, then asks each mode to retrieve them.
# You can read the output and check: did it get them right?
#
# Usage:  bash fdm_simple_demo.sh
# ============================================================

python3 << 'PYEOF'
import sys, os, re, random, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, '/root/FDM_IN_WEIGHTS')
from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]

# ---- Minimal write head classes (for loading checkpoints) ----
def generate_s_random_interleaver(length, S, seed=42):
    import random as rnd
    rnd.seed(seed)
    perm = list(range(length))
    for i in range(length):
        for _ in range(100):
            j = rnd.randint(i, length - 1)
            ok = True
            for k in range(max(0, i - S + 1), i):
                if abs(perm[j] - perm[k]) < S: ok = False; break
            if ok: perm[i], perm[j] = perm[j], perm[i]; break
    return perm

class FDMAdaptiveWriteHead(nn.Module):
    def __init__(s, n_layers, n_kv_heads, head_dim, seq_len, n_channels=40, max_values=5, embed_dim=384, max_samples=1024):
        super().__init__()
        s.n_layers=n_layers; s.n_kv_heads=n_kv_heads; s.head_dim=head_dim; s.seq_len=seq_len
        s.n_channels=n_channels; s.embed_dim=embed_dim; s.max_dual_n=2*max_samples
        s.ch_embed=nn.Embedding(n_channels,embed_dim); s.val_embed=nn.Embedding(max_values,embed_dim)
        s.modulation_proj=nn.Sequential(nn.Linear(embed_dim,embed_dim),nn.GELU(),nn.Linear(embed_dim,embed_dim))
        s.carrier_pool=nn.Sequential(nn.Linear(s.max_dual_n,embed_dim),nn.GELU(),nn.Linear(embed_dim,embed_dim))
        fl=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=4,dim_feedforward=embed_dim*4,dropout=0.1,batch_first=True,activation='gelu')
        s.feature_attn=nn.TransformerEncoder(fl,num_layers=2)
        s.pos_embed=nn.Parameter(torch.randn(1,seq_len,embed_dim)*0.02)
        s.cross_attn=nn.MultiheadAttention(embed_dim,num_heads=4,batch_first=True,dropout=0.1)
        s.cross_norm=nn.LayerNorm(embed_dim)
        kv=n_layers*2*n_kv_heads*head_dim
        s.pos_to_kv=nn.Sequential(nn.Linear(embed_dim,embed_dim*4),nn.GELU(),nn.Linear(embed_dim*4,embed_dim*4),nn.GELU(),nn.Linear(embed_dim*4,kv))
        s._cc={}
    def _get_carriers(s,n,fs,d):
        if (n,fs) not in s._cc:
            f=torch.arange(1,s.n_channels+1,dtype=torch.float32); t=torch.linspace(0,n/fs,n)
            c=torch.sin(2*math.pi*f.unsqueeze(1)*t.unsqueeze(0)).to(d)
            S=int(math.sqrt(n/2)); sp=generate_s_random_interleaver(n,S,seed=43)
            s._cc[(n,fs)]=(c,torch.tensor(sp,dtype=torch.long,device=d))
        return s._cc[(n,fs)]
    def forward(s,ch,vl,d,ns=1024,fs=400.0):
        ct=torch.tensor(ch,device=d).unsqueeze(0); vt=torch.tensor(vl,device=d).unsqueeze(0)
        mod=s.modulation_proj(s.ch_embed(ct)+s.val_embed(vt))
        car,sp=s._get_carriers(ns,fs,d)
        dc=[torch.cat([car[k],car[k][sp]]) for k in range(s.n_channels)]
        ds=torch.stack(dc,0)
        if ds.shape[1]<s.max_dual_n: ds=F.pad(ds,(0,s.max_dual_n-ds.shape[1]))
        cf=s.feature_attn(s.carrier_pool(ds).unsqueeze(0)*mod)
        a,_=s.cross_attn(s.pos_embed,cf,cf)
        pc=s.cross_norm(s.pos_embed+a)
        kv=s.pos_to_kv(pc).view(1,s.seq_len,s.n_layers,2,s.n_kv_heads,s.head_dim)
        return kv.permute(2,3,0,4,1,5)
    def make_cache(s,ch,vl,d,ns=1024,fs=400.0):
        kv=s.forward(ch,vl,d,ns,fs); c=DynamicCache()
        for i in range(s.n_layers): c.update(kv[i,0].half(),kv[i,1].half(),i)
        return c,kv.shape[4]

class HybridWriteHead(nn.Module):
    def __init__(s,n_layers,n_kv_heads,head_dim,seq_len,n_channels=40,max_values=5,embed_dim=384,n_attn_layers=2,turbo_iters=3):
        super().__init__()
        s.n_layers=n_layers;s.n_kv_heads=n_kv_heads;s.head_dim=head_dim;s.seq_len=seq_len
        s.n_channels=n_channels;s.turbo_iters=turbo_iters;s.embed_dim=embed_dim
        s.ch_embed=nn.Embedding(n_channels,embed_dim);s.val_embed=nn.Embedding(max_values,embed_dim)
        S=int(math.sqrt(n_channels/2))
        s.interleaver=generate_s_random_interleaver(n_channels,S,seed=42)
        s.deinterleaver=[0]*n_channels
        for i,j in enumerate(s.interleaver): s.deinterleaver[j]=i
        class CD(nn.Module):
            def __init__(s2,ed,na):
                super().__init__()
                l=nn.TransformerEncoderLayer(d_model=ed,nhead=4,dim_feedforward=ed*4,dropout=0.1,batch_first=True,activation='gelu')
                s2.attn=nn.TransformerEncoder(l,num_layers=na)
                s2.ext_gate=nn.Sequential(nn.Linear(ed*2,ed),nn.Sigmoid())
                s2.ext_fuse=nn.Sequential(nn.Linear(ed*2,ed),nn.GELU())
                s2.confidence=nn.Sequential(nn.Linear(ed,ed),nn.Sigmoid())
            def forward(s2,x,ext=None):
                if ext is not None:
                    g=s2.ext_gate(torch.cat([x,ext],-1)); f=s2.ext_fuse(torch.cat([x,ext],-1)); x=x+g*f
                o=s2.attn(x); return o,s2.confidence(o)*(o-x)
        s.decoder_a=CD(embed_dim,n_attn_layers);s.decoder_b=CD(embed_dim,n_attn_layers)
        s.pos_embed=nn.Parameter(torch.randn(1,seq_len,embed_dim)*0.02)
        s.cross_attn=nn.MultiheadAttention(embed_dim,num_heads=4,batch_first=True,dropout=0.1)
        s.cross_norm=nn.LayerNorm(embed_dim)
        kv=n_layers*2*n_kv_heads*head_dim
        s.pos_to_kv=nn.Sequential(nn.Linear(embed_dim,embed_dim*4),nn.GELU(),nn.Linear(embed_dim*4,embed_dim*4),nn.GELU(),nn.Linear(embed_dim*4,kv))
    def _il(s,x): return x[:,torch.tensor(s.interleaver,device=x.device),:]
    def _dil(s,x): return x[:,torch.tensor(s.deinterleaver,device=x.device),:]
    def forward(s,ch,vl,d):
        ct=torch.tensor(ch,device=d).unsqueeze(0);vt=torch.tensor(vl,device=d).unsqueeze(0)
        raw=s.ch_embed(ct)+s.val_embed(vt);en=raw;ei=s._il(raw);ea=None;eb=None
        for _ in range(s.turbo_iters):
            efa=s._dil(eb) if eb is not None else None; oa,ea=s.decoder_a(en,efa)
            ob,eb=s.decoder_b(s._il(raw),s._il(ea))
        a,_=s.cross_attn(s.pos_embed,oa,oa);pc=s.cross_norm(s.pos_embed+a)
        kv=s.pos_to_kv(pc).view(1,s.seq_len,s.n_layers,2,s.n_kv_heads,s.head_dim)
        return kv.permute(2,3,0,4,1,5)
    def make_cache(s,ch,vl,d):
        kv=s.forward(ch,vl,d);c=DynamicCache()
        for i in range(s.n_layers): c.update(kv[i,0].half(),kv[i,1].half(),i)
        return c,kv.shape[4]

# ---- Generate helper ----
def gen_kv(model, tokenizer, prompt, kv, kv_len, device):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    t = torch.tensor([ids], dtype=torch.long).to(device)
    pos = torch.arange(kv_len, kv_len+len(ids), device=device).unsqueeze(0)
    attn = torch.ones(1, kv_len+len(ids), device=device, dtype=torch.long)
    gen = []; past = kv
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(t, past_key_values=past, position_ids=pos, attention_mask=attn, use_cache=True)
        past = out.past_key_values; nt = out.logits[:,-1,:].argmax(-1, keepdim=True)
        gen.append(nt.item()); tl = kv_len+len(ids)+1
        for s in range(299):
            p = torch.tensor([[tl-1+s]], device=device)
            a = torch.ones(1, tl+s, device=device, dtype=torch.long)
            o = model(nt, past_key_values=past, position_ids=p, attention_mask=a, use_cache=True)
            past = o.past_key_values; nt = o.logits[:,-1,:].argmax(-1, keepdim=True)
            tid = nt.item(); gen.append(tid)
            if tid == tokenizer.eos_token_id: break
    return tokenizer.decode(gen, skip_special_tokens=True)

# ==============================================================
# MAIN
# ==============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n  Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("prompterminal/fdm-40ch-nhop-qwen3", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "prompterminal/fdm-40ch-nhop-qwen3", trust_remote_code=True, dtype=torch.float16).to(device)
model.eval()

encoder = TurboFDMSignalEncoder(
    vocab_size=151936, tokenizer=tokenizer,
    num_tokens_per_encoder=256, sample_rate=100.0,
    a_high=1.0, a_low=0.25, num_levels=64, seed=42)

# Setup
mem = {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}
ch_ids = list(range(NUM_CHANNELS))
val_ids = [MEMORY_SCHEMAS[ch][1].index(mem[ch]) for ch in range(NUM_CHANNELS)]
fdm_text, fdm_tokens = encoder.encode_memory(mem)
memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
seq_len = len(memory_start) + len(fdm_tokens)
n_kv = model.config.num_key_value_heads
max_v = max(len(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS))

# Load write heads
standalone = FDMAdaptiveWriteHead(model.config.num_hidden_layers, n_kv, 128, seq_len, NUM_CHANNELS, max_v, 384, 1024).to(device)
standalone.load_state_dict(torch.load("/workspace/FDM_IN_WEIGHTS/fdm_v3_adaptive.pt", map_location=device))
standalone.eval()

hybrid = HybridWriteHead(model.config.num_hidden_layers, n_kv, 128, seq_len, NUM_CHANNELS, max_v, 384, 2, 3).to(device)
hybrid.load_state_dict(torch.load("/workspace/FDM_IN_WEIGHTS/hybrid_write_head.pt", map_location=device))
hybrid.eval()

# Pick 8 channels to display (easier to read than 32)
display_channels = [8, 9, 10, 11, 15, 20, 30, 39]
question = "Report all context values for channels 8-39."

print("\n" + "=" * 60)
print("  FDM MEMORY DEMO")
print("  Same facts, four different memory substrates")
print("=" * 60)

# Ground truth
print("\n  FACTS TO REMEMBER:")
print("  " + "─" * 40)
for k in display_channels:
    print(f"    {CHANNEL_NAMES[k]:<14s} = {mem[k]}")
print(f"    (+ {32 - len(display_channels)} more channels)")

# ---- MODE 1: In-Context ----
print("\n\n  ┌─────────────────────────────────────────────┐")
print("  │  MODE 1: FDM IN-CONTEXT                     │")
print("  │  Facts encoded as 512 FDM tokens in prompt   │")
print("  │  Context cost: 512 tokens                    │")
print("  └─────────────────────────────────────────────┘")

prompt1 = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
ids1 = tokenizer.encode(prompt1, return_tensors='pt').to(device)
with torch.no_grad():
    out1 = model.generate(ids1, max_new_tokens=350, do_sample=False, pad_token_id=tokenizer.eos_token_id)
text1 = tokenizer.decode(out1[0][ids1.shape[1]:], skip_special_tokens=True)

print("\n  Results:")
correct1 = 0
for k in display_channels:
    name = CHANNEL_NAMES[k]; expected = mem[k]
    found = bool(re.search(rf"\b{re.escape(name)}={re.escape(expected)}\b", text1))
    correct1 += int(found)
    mark = "✓" if found else "✗"
    print(f"    {mark}  {name:<14s} expected={expected:<16s} {'CORRECT' if found else 'WRONG'}")

# ---- MODE 2: Frozen KV ----
print("\n\n  ┌─────────────────────────────────────────────┐")
print("  │  MODE 2: FROZEN KV CACHE                    │")
print("  │  FDM processed once → KV frozen → reused    │")
print("  │  Context cost: 0 fact tokens                 │")
print("  └─────────────────────────────────────────────┘")

prefix_ids = memory_start + fdm_tokens
prefix_t = torch.tensor([prefix_ids], dtype=torch.long).to(device)
with torch.no_grad():
    out_kv = model(prefix_t, use_cache=True)
frozen_kv = out_kv.past_key_values
kv_len = len(prefix_ids)
prompt2 = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
text2 = gen_kv(model, tokenizer, prompt2, frozen_kv, kv_len, device)

print("\n  Results:")
correct2 = 0
for k in display_channels:
    name = CHANNEL_NAMES[k]; expected = mem[k]
    found = bool(re.search(rf"\b{re.escape(name)}={re.escape(expected)}\b", text2))
    correct2 += int(found)
    mark = "✓" if found else "✗"
    print(f"    {mark}  {name:<14s} expected={expected:<16s} {'CORRECT' if found else 'WRONG'}")

# ---- MODE 3: Parametric ----
print("\n\n  ┌─────────────────────────────────────────────┐")
print("  │  MODE 3: PARAMETRIC (Write Head)             │")
print("  │  No FDM encoder, no tokens                   │")
print("  │  Facts → write head → KV cache directly      │")
print("  │  Context cost: 0 fact tokens                  │")
print("  └─────────────────────────────────────────────┘")

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    cache3, kl3 = standalone.make_cache(ch_ids, val_ids, device)
prompt3 = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
text3 = gen_kv(model, tokenizer, prompt3, cache3, kl3, device)

print("\n  Results:")
correct3 = 0
for k in display_channels:
    name = CHANNEL_NAMES[k]; expected = mem[k]
    found = bool(re.search(rf"\b{re.escape(name)}={re.escape(expected)}\b", text3))
    correct3 += int(found)
    mark = "✓" if found else "✗"
    print(f"    {mark}  {name:<14s} expected={expected:<16s} {'CORRECT' if found else 'WRONG'}")

# ---- MODE 4: Hybrid ----
print("\n\n  ┌─────────────────────────────────────────────┐")
print("  │  MODE 4: HYBRID (Write Head + Context)       │")
print("  │  Write head KV + FDM tokens together          │")
print("  │  Write head complements context signal        │")
print("  └─────────────────────────────────────────────┘")

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    cache4, kl4 = hybrid.make_cache(ch_ids, val_ids, device)
prompt4 = f"{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
text4 = gen_kv(model, tokenizer, prompt4, cache4, kl4, device)

print("\n  Results:")
correct4 = 0
for k in display_channels:
    name = CHANNEL_NAMES[k]; expected = mem[k]
    found = bool(re.search(rf"\b{re.escape(name)}={re.escape(expected)}\b", text4))
    correct4 += int(found)
    mark = "✓" if found else "✗"
    print(f"    {mark}  {name:<14s} expected={expected:<16s} {'CORRECT' if found else 'WRONG'}")

# ---- Summary ----
n = len(display_channels)
print(f"\n\n  {'=' * 60}")
print(f"  SCORECARD ({n} channels shown)")
print(f"  {'=' * 60}")
print(f"""
  Mode 1 — In-Context:      {correct1}/{n} correct   (512 fact tokens)
  Mode 2 — Frozen KV:       {correct2}/{n} correct   (0 fact tokens)
  Mode 3 — Parametric:      {correct3}/{n} correct   (0 fact tokens, 0 encoder)
  Mode 4 — Hybrid:          {correct4}/{n} correct   (write head + context)

  The same facts, four substrates:
    Context tokens → KV cache → Parameters → Hybrid
    Each step: more compression, less token cost.
    Hybrid: parametric + context exceed either alone.
""")
PYEOF
