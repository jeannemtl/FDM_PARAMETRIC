"""
FDM-in-Weights Experiment 3: Learned Write Head
================================================

We proved that the model reads FDM at 89.1% from pre-computed KV cache
(vs 93.7% from live context tokens). Now we train a PERMANENT module
whose WEIGHTS learn to produce those KV entries from channel values.

Architecture:
  FDMMemoryLayer: (channel_ids, value_ids) → KV cache entries
  - Its weights are trained model parameters (nn.Parameter)
  - At inference: call with current channel values → get KV pairs → prepend
  - Facts are encoded in the module's learned weights

Training:
  - Generate random channel values
  - Run real FDM encoder → FDM tokens → model forward pass → target KV cache
  - FDMMemoryLayer(channel_ids, value_ids) → predicted KV cache
  - Loss: MSE between predicted and target KV entries
  - After training: the module's weights encode the mapping from
    channel values to the KV representations the model needs

Evaluation:
  - Use trained FDMMemoryLayer to produce KV cache (no FDM encoder, no tokens)
  - Generate with prepended KV cache
  - Compare accuracy to baseline (93.7%) and frozen injection (89.1%)

Usage:
  python fdm_write_head.py                    # train + eval
  python fdm_write_head.py --eval_only        # eval saved checkpoint
  python fdm_write_head.py --n_train 5000     # more training steps
"""

import sys, os, re, json, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Convert memory dict to (channel_ids, value_ids) tensors."""
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
# FDMMemoryLayer — the write head
# ============================================================

class FDMMemoryLayer(nn.Module):
    """
    Learned write head: maps channel values → KV cache entries.

    Architecture:
      1. Embed each (channel_id, value) pair independently
      2. Process through a small transformer to model inter-channel
         interactions (channels affect each other in the composite signal)
      3. Project to KV contributions for all layers
      4. Expand across the sequence length dimension

    The key insight: in real FDM, each time sample is a superposition
    of ALL channels. So the KV entry at position t depends on all 40
    channel values jointly, not independently. The self-attention in
    step 2 captures this coupling.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len=514,
                 n_channels=40, max_values=5, embed_dim=256, n_attn_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len  # [MEMORY] + 512 FDM tokens + [/MEMORY] ≈ 514
        self.n_channels = n_channels

        # Embed channels and values
        self.ch_embed = nn.Embedding(n_channels, embed_dim)
        self.val_embed = nn.Embedding(max_values, embed_dim)

        # Self-attention to model inter-channel coupling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
            dropout=0.0, batch_first=True, activation='gelu'
        )
        self.channel_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)

        # Project to per-position KV contributions
        # Each position needs: n_layers * 2 (K and V) * n_kv_heads * head_dim
        self.kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        
        # Two-stage projection: channels → position embeddings → KV
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.channel_to_pos = nn.Linear(embed_dim, embed_dim)
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.kv_per_pos),
        )

    def forward(self, ch_ids, val_ids, device):
        """
        ch_ids: list of ints (n_channels,)
        val_ids: list of ints (n_channels,)
        Returns: DynamicCache-compatible list for model.generate
        """
        ch_t = torch.tensor(ch_ids, device=device).unsqueeze(0)   # (1, n_ch)
        val_t = torch.tensor(val_ids, device=device).unsqueeze(0)  # (1, n_ch)

        # Embed and combine
        ch_emb = self.ch_embed(ch_t)    # (1, n_ch, E)
        val_emb = self.val_embed(val_t)  # (1, n_ch, E)
        combined = ch_emb + val_emb      # (1, n_ch, E)

        # Inter-channel attention (model the superposition coupling)
        coupled = self.channel_attn(combined)  # (1, n_ch, E)

        # Pool channels → single context vector
        context = coupled.mean(dim=1, keepdim=True)  # (1, 1, E)

        # Expand to positions via learned position embeddings
        pos = self.pos_embed  # (1, seq_len, E)
        pos_context = pos + self.channel_to_pos(context)  # (1, seq_len, E)

        # Project to KV for all layers
        kv_flat = self.pos_to_kv(pos_context)  # (1, seq_len, kv_per_pos)

        # Reshape to (n_layers, 2, batch, n_kv_heads, seq_len, head_dim)
        kv = kv_flat.view(1, self.seq_len, self.n_layers, 2,
                          self.n_kv_heads, self.head_dim)
        kv = kv.permute(2, 3, 0, 4, 1, 5)  # (n_layers, 2, 1, n_kv_heads, seq_len, head_dim)

        return kv

    def to_dynamic_cache(self, kv, device):
        """Convert our KV tensor to the format model.forward expects."""
        from transformers import DynamicCache
        cache = DynamicCache()
        for layer_idx in range(self.n_layers):
            k = kv[layer_idx, 0]  # (1, n_kv_heads, seq_len, head_dim)
            v = kv[layer_idx, 1]
            cache.update(k.half(), v.half(), layer_idx)
        return cache


# ============================================================
# Get target KV cache from real FDM tokens
# ============================================================

def get_target_kv(model, tokenizer, encoder, memory, device):
    """
    Run the real pipeline: encode → tokenize → forward → KV cache.
    Returns list of (K, V) tensors per layer.
    """
    fdm_text, fdm_tokens = encoder.encode_memory(memory)
    
    # Build full prefix: [MEMORY] + fdm_tokens (already includes both encoders)
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    full_ids = memory_start + fdm_tokens
    ids_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model(ids_tensor, use_cache=True)

    # Extract K, V from DynamicCache
    kv_cache = out.past_key_values
    kv_list = []
    for layer_idx in range(len(kv_cache.layers)):
        layer = kv_cache.layers[layer_idx]
        kv_list.append((layer.keys.detach(), layer.values.detach()))

    return kv_list, len(full_ids)


# ============================================================
# Training
# ============================================================

def train_write_head(model, tokenizer, encoder, write_head, device,
                     n_steps=3000, lr=1e-4):
    print("\n" + "=" * 60)
    print(f"TRAINING: FDMMemoryLayer ({n_steps} steps)")
    print("  Target: match model's KV cache from real FDM tokens")
    print("=" * 60)

    model.eval()
    write_head.train()
    
    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr / 10
    )

    t0 = time.time()
    losses = []

    for step in range(1, n_steps + 1):
        mem = random_memory()
        ch_ids, val_ids = memory_to_indices(mem)

        # Target: real KV from FDM tokens
        target_kv, seq_len = get_target_kv(model, tokenizer, encoder, mem, device)

        # Predicted: from write head
        pred_kv_raw = write_head(ch_ids, val_ids, device)

        # Ensure write head output matches target sequence length
        # (write head may produce slightly different seq_len)
        # Compute loss only on overlapping positions
        loss = 0.0
        n_layers = len(target_kv)
        for layer_idx in range(n_layers):
            tgt_k, tgt_v = target_kv[layer_idx]
            pred_k = pred_kv_raw[layer_idx, 0]  # (1, n_kv_heads, seq_len, head_dim)
            pred_v = pred_kv_raw[layer_idx, 1]

            # Match sequence lengths
            min_len = min(tgt_k.shape[2], pred_k.shape[2])
            loss += F.mse_loss(pred_k[:, :, :min_len, :], tgt_k[:, :, :min_len, :].float())
            loss += F.mse_loss(pred_v[:, :, :min_len, :], tgt_v[:, :, :min_len, :].float())

        loss = loss / (2 * n_layers)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 200 == 0 or step == n_steps:
            avg_loss = np.mean(losses[-200:])
            elapsed = time.time() - t0
            print(f"  Step {step:5d}/{n_steps} | loss {avg_loss:.6f} | {elapsed:.0f}s")

    return write_head


# ============================================================
# Evaluation with write head
# ============================================================

def run_write_head_eval(model, tokenizer, write_head, n_eval, device):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Learned Write Head (facts in parameters)")
    print("  No FDM encoder, no FDM tokens — just the write head's weights")
    print("=" * 60)

    model.eval()
    write_head.eval()

    channel_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0

    for i in tqdm(range(n_eval), desc="Write head eval"):
        mem = random_memory()
        ch_ids, val_ids = memory_to_indices(mem)

        # Get KV from write head (no FDM encoder involved!)
        with torch.no_grad():
            pred_kv_raw = write_head(ch_ids, val_ids, device)
            pred_cache = write_head.to_dynamic_cache(pred_kv_raw, device)

        fdm_seq_len = pred_kv_raw.shape[4]  # seq_len dimension

        # Prompt without FDM tokens
        question = "Report all context values for channels 8-39."
        rest_prompt = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
        rest_ids = tokenizer.encode(rest_prompt, add_special_tokens=False)
        rest_tensor = torch.tensor([rest_ids], dtype=torch.long).to(device)

        position_ids = torch.arange(fdm_seq_len, fdm_seq_len + len(rest_ids),
                                     device=device).unsqueeze(0)
        attn_mask = torch.ones(1, fdm_seq_len + len(rest_ids),
                                device=device, dtype=torch.long)

        # Manual generation loop
        generated_ids = []
        past_kv = pred_cache
        
        with torch.no_grad():
            out = model(rest_tensor, past_key_values=past_kv,
                       position_ids=position_ids, attention_mask=attn_mask,
                       use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

            total_len = fdm_seq_len + len(rest_ids) + 1

            for step in range(249):
                pos = torch.tensor([[total_len - 1 + step]], device=device)
                attn = torch.ones(1, total_len + step, device=device, dtype=torch.long)

                out = model(next_token, past_key_values=past_kv,
                           position_ids=pos, attention_mask=attn,
                           use_cache=True)
                past_kv = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tok_id = next_token.item()
                generated_ids.append(tok_id)

                if tok_id == tokenizer.eos_token_id:
                    break

        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        scores = score_extra_channels(generated, mem)
        for name, correct in scores.items():
            channel_correct[name] += int(correct)
        total += 1

    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    print(f"\n  Overall extra-channel accuracy: {overall*100:.1f}%")
    for name in sorted(per_ch, key=per_ch.get)[:5]:
        print(f"    {name}: {per_ch[name]*100:.1f}%")
    return per_ch, overall


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_train", type=int, default=3000)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default="fdm_write_head.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16
    ).to(device)
    model.eval()

    # Freeze the base model entirely
    for p in model.parameters():
        p.requires_grad = False

    print(f"  Loaded. Layers: {model.config.num_hidden_layers}, "
          f"KV heads: {model.config.num_key_value_heads}, "
          f"Head dim: {model.config.hidden_size // model.config.num_attention_heads}")

    encoder = make_encoder(tokenizer)

    # Figure out seq_len from a sample
    mem = random_memory()
    _, fdm_tokens = encoder.encode_memory(mem)
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    sample_seq_len = len(memory_start) + len(fdm_tokens)
    print(f"  FDM prefix seq_len: {sample_seq_len}")

    # Create write head
    n_kv_heads = model.config.num_key_value_heads
    head_dim = 128  # actual KV head dim from model output
    max_values = max(len(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS))

    write_head = FDMMemoryLayer(
        n_layers=model.config.num_hidden_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        seq_len=sample_seq_len,
        n_channels=NUM_CHANNELS,
        max_values=max_values,
        embed_dim=256,
        n_attn_layers=2,
    ).to(device)

    n_params = sum(p.numel() for p in write_head.parameters())
    print(f"  Write head parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    if args.eval_only and os.path.exists(args.checkpoint):
        print(f"  Loading checkpoint: {args.checkpoint}")
        write_head.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        write_head = train_write_head(
            model, tokenizer, encoder, write_head, device,
            n_steps=args.n_train, lr=args.lr,
        )
        torch.save(write_head.state_dict(), args.checkpoint)
        print(f"  Saved checkpoint: {args.checkpoint}")

    # Evaluate
    pc, ov = run_write_head_eval(model, tokenizer, write_head, args.n_eval, device)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Baseline (FDM in context):    93.7%")
    print(f"  Frozen KV injection:          89.1%")
    print(f"  Learned write head:           {ov*100:.1f}%")
    print()
    if ov > 0.85:
        print("  → Write head approaches frozen KV — facts live in parameters!")
    elif ov > 0.5:
        print("  → Partial success — write head captures some FDM structure.")
        print("    Try: more training steps (--n_train 10000), larger embed_dim")
    else:
        print("  → Write head needs more capacity or different architecture.")
        print("    The read path works (89.1% frozen KV proves it).")
        print("    The write path needs to produce better KV approximations.")

    results = {
        "baseline": 0.937,
        "frozen_kv": 0.891,
        "learned_write_head": float(ov),
        "write_head_params": n_params,
        "n_train_steps": args.n_train,
    }
    json.dump(results, open("fdm_write_head_results.json", "w"), indent=2)
    print(f"\nResults saved to fdm_write_head_results.json")


if __name__ == "__main__":
    main()
