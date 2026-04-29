"""
Split-Source Hybrid: Write Head (Block A) + FDM Context (Block B)
================================================================

Uses the two-block trained model as base.
- Block A (positions 0-512): Write head produces KV entries for channels 8-23
- Block B (positions 513+): FDM context tokens carry channels 24-39
- Each source carries INDEPENDENT facts (different channel subsets)
- The model must read both to answer correctly

This is the true parametric + in-context hybrid with independent facts.
"""

import sys, os, re, json, random, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, '/root/FDM_IN_WEIGHTS/scripts')
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]

# Block A channels: 8-23 (parametric write head)
# Block B channels: 24-39 (FDM context tokens)
BLOCK_A_CHANNELS = list(range(8, 24))  # 16 channels
BLOCK_B_CHANNELS = list(range(24, 40))  # 16 channels
ALL_CHANNELS = list(range(8, 40))       # 32 channels


def layer_kvs_to_cache(layer_kvs, n_layers):
    """Convert write head layer_kvs dict to a DynamicCache for HF transformers."""
    cache = DynamicCache()
    for l in range(n_layers):
        k, v = layer_kvs[l]
        # k, v: (B, n_pos, n_heads, kv_dim) -> (B, n_heads, n_pos, kv_dim)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        cache.update(k, v, l)
    return cache


def make_encoder(tokenizer):
    return TurboFDMSignalEncoder(
        vocab_size=151936, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )


def random_memory():
    return {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}


# ============================================================
# Write Head Architecture (same as hybrid write head)
# ============================================================

class CrossAttentionWriteHead(nn.Module):
    """Write head that maps channel-value pairs to KV entries for Block A positions."""
    
    def __init__(self, n_channels=16, n_values=64, embed_dim=384,
                 n_kv_positions=513, n_layers=28, kv_dim=128, n_kv_heads=8):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_positions = n_kv_positions
        self.kv_dim = kv_dim
        self.n_kv_heads = n_kv_heads
        
        # Channel and value embeddings
        self.channel_embed = nn.Embedding(n_channels, embed_dim)
        self.value_embed = nn.Embedding(n_values + 1, embed_dim)  # +1 for padding
        
        # Refinement: 2-layer transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=6, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.refiner = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Position embeddings for KV positions
        self.pos_embed = nn.Parameter(torch.randn(n_kv_positions, embed_dim) * 0.02)
        
        # Cross-attention: positions attend to channels
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )
        
        # Project to KV for all layers
        total_kv_dim = n_layers * 2 * n_kv_heads * kv_dim  # K and V for all layers
        self.kv_proj = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, total_kv_dim)
        )
    
    def forward(self, channel_ids, value_ids):
        """
        channel_ids: (B, K) channel indices 0..15
        value_ids: (B, K) value indices 0..63
        Returns: dict of {layer_idx: (key, value)} each (B, n_pos, n_heads, kv_dim)
        """
        B = channel_ids.shape[0]
        
        # Embed and combine
        ch_emb = self.channel_embed(channel_ids)   # (B, K, D)
        val_emb = self.value_embed(value_ids)       # (B, K, D)
        x = ch_emb + val_emb                        # (B, K, D)
        
        # Refine
        x = self.refiner(x)                          # (B, K, D)
        
        # Cross-attention: each position attends to all channels
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)
        attended, _ = self.cross_attn(pos, x, x)             # (B, P, D)
        
        # Project to KV
        kv_raw = self.kv_proj(attended)  # (B, P, n_layers*2*n_heads*kv_dim)
        
        # Reshape to per-layer K, V
        kv_raw = kv_raw.view(B, self.n_kv_positions, self.n_layers, 2,
                             self.n_kv_heads, self.kv_dim)
        
        layer_kvs = {}
        for l in range(self.n_layers):
            k = kv_raw[:, :, l, 0, :, :]  # (B, P, H, D)
            v = kv_raw[:, :, l, 1, :, :]
            layer_kvs[l] = (k, v)
        
        return layer_kvs


# ============================================================
# Forward pass with split-source hybrid
# ============================================================

def hybrid_forward(model, tokenizer, write_head, memory, encoder, device):
    """
    Block A: write head KV for channels 8-23
    Block B: FDM context tokens for channels 24-39
    Question asks for ALL channels 8-39
    """
    # Prepare write head input (Block A channels)
    # Map channel 8-23 to local indices 0-15
    ch_ids = torch.tensor([[c - 8 for c in BLOCK_A_CHANNELS]], device=device)
    
    # Get value indices for Block A channels
    val_ids = []
    for c in BLOCK_A_CHANNELS:
        val = memory[c]
        val_list = MEMORY_SCHEMAS[c][1]
        val_idx = val_list.index(val) if val in val_list else 0
        val_ids.append(val_idx)
    val_ids = torch.tensor([val_ids], device=device)
    
    # Generate write head KV entries
    layer_kvs = write_head(ch_ids, val_ids)
    
    # Generate Block B FDM context tokens (channels 24-39 only)
    fdm_text, _ = encoder.encode_memory(memory)
    
    # Build the prompt: Block B context + question
    ch_names = [CHANNEL_NAMES[k] for k in ALL_CHANNELS]
    question = f"Report values for: {', '.join(ch_names)}."
    
    # The context only has Block B as tokens
    prompt = f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]
    
    # Run model with write head KV prepended
    n_kv_pos = write_head.n_kv_positions
    
    # Build position ids: write head at 0..512, context at 513..513+seq_len
    position_ids = torch.arange(n_kv_pos, n_kv_pos + seq_len, device=device).unsqueeze(0)
    
    # Build past_key_values from write head
    past_key_values = layer_kvs_to_cache(layer_kvs, write_head.n_layers)
    
    with torch.no_grad():
        # First get the logits with KV prepended
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
    
    return input_ids, position_ids, past_key_values


def generate_with_hybrid(model, tokenizer, write_head, memory, encoder, device,
                          max_new_tokens=350):
    """Generate answer using split-source hybrid."""
    # Prepare write head input
    ch_ids = torch.tensor([[c - 8 for c in BLOCK_A_CHANNELS]], device=device)
    val_ids = []
    for c in BLOCK_A_CHANNELS:
        val = memory[c]
        val_list = MEMORY_SCHEMAS[c][1]
        val_idx = val_list.index(val) if val in val_list else 0
        val_ids.append(val_idx)
    val_ids = torch.tensor([val_ids], device=device)
    
    layer_kvs = write_head(ch_ids, val_ids)
    
    # Block B context
    fdm_text, _ = encoder.encode_memory(memory)
    ch_names = [CHANNEL_NAMES[k] for k in ALL_CHANNELS]
    question = f"Report values for: {', '.join(ch_names)}."
    prompt = f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]
    n_kv_pos = write_head.n_kv_positions
    
    position_ids = torch.arange(n_kv_pos, n_kv_pos + seq_len, device=device).unsqueeze(0)
    
    past_kv = layer_kvs_to_cache(layer_kvs, write_head.n_layers)
    
    # Generate token by token
    generated = input_ids
    cur_past = past_kv
    cur_pos = position_ids
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if cur_past is not None and generated.shape[1] > input_ids.shape[1]:
                # After first pass, only feed last token
                last_tok = generated[:, -1:]
                last_pos = cur_pos[:, -1:] + 1
                outputs = model(
                    input_ids=last_tok,
                    position_ids=last_pos,
                    past_key_values=cur_past,
                    use_cache=True
                )
                cur_pos = last_pos
            else:
                outputs = model(
                    input_ids=generated,
                    position_ids=cur_pos,
                    past_key_values=cur_past,
                    use_cache=True
                )
            
            cur_past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    answer = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return answer


# ============================================================
# Training
# ============================================================

def train_split_source(model, tokenizer, write_head, encoder, device,
                        n_steps=10000, lr=3e-4, eval_every=2000, n_eval=20):
    """Train write head for Block A with Block B context present."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    write_head.train()
    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr/20)
    
    t0 = time.time()
    losses = []
    
    for step in range(1, n_steps + 1):
        mem = random_memory()
        
        # Write head input: Block A channels (8-23), local indices 0-15
        ch_ids = torch.tensor([[c - 8 for c in BLOCK_A_CHANNELS]], device=device)
        val_ids = []
        for c in BLOCK_A_CHANNELS:
            val = mem[c]
            val_list = MEMORY_SCHEMAS[c][1]
            val_idx = val_list.index(val) if val in val_list else 0
            val_ids.append(val_idx)
        val_ids = torch.tensor([val_ids], device=device)
        
        # Get write head KV
        layer_kvs = write_head(ch_ids, val_ids)
        
        # Block B: FDM context tokens for channels 24-39
        fdm_text, _ = encoder.encode_memory(mem)
        
        ch_names = [CHANNEL_NAMES[k] for k in ALL_CHANNELS]
        question = f"Report values for: {', '.join(ch_names)}."
        parts = [f"{CHANNEL_NAMES[k]}={mem[k]}" for k in ALL_CHANNELS]
        answer = "Context: " + ", ".join(parts) + "."
        
        q_text = f"\nQuestion: {question}\nAnswer:"
        a_text = f" {answer}"
        prompt = f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]{q_text}{a_text}"
        
        full_ids = tokenizer.encode(prompt, add_special_tokens=False)
        a_ids = tokenizer.encode(a_text, add_special_tokens=False)
        prefix_len = len(full_ids) - len(a_ids)
        labels = [-100] * prefix_len + full_ids[prefix_len:]
        
        # Truncate if needed
        max_len = 1536
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
            labels = labels[:max_len]
        
        input_tensor = torch.tensor([full_ids], device=device)
        label_tensor = torch.tensor([labels], device=device)
        seq_len = input_tensor.shape[1]
        n_kv_pos = write_head.n_kv_positions
        
        # Position IDs: context starts after write head positions
        position_ids = torch.arange(n_kv_pos, n_kv_pos + seq_len, device=device).unsqueeze(0)
        
        # Build past_key_values as DynamicCache
        past_kv = layer_kvs_to_cache(layer_kvs, write_head.n_layers)
        
        # Forward with write head KV prepended
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            outputs = model(
                input_ids=input_tensor,
                position_ids=position_ids,
                past_key_values=past_kv,
                labels=label_tensor,
                use_cache=False
            )
            loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
        if step % 500 == 0 or step == 1:
            avg = np.mean(losses[-500:]) if len(losses) >= 500 else np.mean(losses)
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step:5d}/{n_steps} | loss {avg:.4f} | "
                  f"lr {cur_lr:.1e} | {elapsed:.0f}s", flush=True)
        
        if eval_every > 0 and step % eval_every == 0:
            evaluate_split_source(model, tokenizer, write_head, encoder, device, n_eval)
            write_head.train()


def evaluate_split_source(model, tokenizer, write_head, encoder, device, n_eval=20):
    """Evaluate split-source hybrid."""
    write_head.eval()
    
    block_a_correct = 0; block_a_total = 0
    block_b_correct = 0; block_b_total = 0
    all_correct = 0; all_total = 0
    
    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        answer = generate_with_hybrid(model, tokenizer, write_head, mem, encoder, device)
        
        # Check Block A channels (parametric)
        for k in BLOCK_A_CHANNELS:
            if re.search(rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b", answer):
                block_a_correct += 1
            block_a_total += 1
        
        # Check Block B channels (context)
        for k in BLOCK_B_CHANNELS:
            if re.search(rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b", answer):
                block_b_correct += 1
            block_b_total += 1
        
        # All channels
        for k in ALL_CHANNELS:
            if re.search(rf"\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\b", answer):
                all_correct += 1
            all_total += 1
    
    a_acc = block_a_correct / block_a_total if block_a_total > 0 else 0
    b_acc = block_b_correct / block_b_total if block_b_total > 0 else 0
    all_acc = all_correct / all_total if all_total > 0 else 0
    
    print(f"\n    Block A (parametric, ch8-23):  {a_acc*100:.1f}%")
    print(f"    Block B (context, ch24-39):    {b_acc*100:.1f}%")
    print(f"    ALL channels (8-39):           {all_acc*100:.1f}%")
    
    return a_acc, b_acc, all_acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-two-block-qwen3")
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--output_dir", default="/workspace/FDM_IN_WEIGHTS/split_source_hybrid")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("  SPLIT-SOURCE HYBRID: Write Head (A) + Context (B)")
    print("=" * 60)
    print(f"  Base model: {args.model}")
    print(f"  Block A: channels 8-23 (parametric write head)")
    print(f"  Block B: channels 24-39 (FDM context tokens)")
    print(f"  Steps: {args.n_steps}")
    print(f"  LR: {args.lr}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    encoder = make_encoder(tokenizer)
    
    # Create write head
    write_head = CrossAttentionWriteHead(
        n_channels=16,  # channels 8-23 mapped to 0-15
        n_values=64,
        embed_dim=384,
        n_kv_positions=513,
        n_layers=28,
        kv_dim=128,
        n_kv_heads=8
    ).to(device)
    
    wh_params = sum(p.numel() for p in write_head.parameters())
    print(f"  Write head params: {wh_params:,} ({wh_params/1e6:.1f}M)")
    
    # Initial eval
    print("\n  Initial evaluation (random write head):", flush=True)
    evaluate_split_source(model, tokenizer, write_head, encoder, device, n_eval=args.n_eval)
    
    # Train
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60, flush=True)
    train_split_source(model, tokenizer, write_head, encoder, device,
                        n_steps=args.n_steps, lr=args.lr,
                        eval_every=args.eval_every, n_eval=args.n_eval)
    
    # Final eval
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60, flush=True)
    a_acc, b_acc, all_acc = evaluate_split_source(
        model, tokenizer, write_head, encoder, device, n_eval=args.n_eval)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(write_head.state_dict(), os.path.join(args.output_dir, "split_source_write_head.pt"))
    
    results = {
        "block_a_acc": float(a_acc),
        "block_b_acc": float(b_acc),
        "all_acc": float(all_acc),
        "config": {
            "n_steps": args.n_steps, "lr": args.lr,
            "base_model": args.model,
            "block_a_channels": BLOCK_A_CHANNELS,
            "block_b_channels": BLOCK_B_CHANNELS,
        }
    }
    json.dump(results, open(os.path.join(args.output_dir, "results.json"), "w"), indent=2)
    print(f"\n  Saved to {args.output_dir}")
    print(f"  Block A (parametric): {a_acc*100:.1f}%")
    print(f"  Block B (context):    {b_acc*100:.1f}%")
    print(f"  ALL channels:         {all_acc*100:.1f}%")


if __name__ == "__main__":
    main()
