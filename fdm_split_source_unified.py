"""
Unified Split-Source Hybrid: Write Head (Block A) + FDM Context (Block B)
=========================================================================

Auto-detects model architecture (layers, KV heads, KV dim) from config.
Works across: GPT-2, Qwen3, LFM2.5, Hermes3, or any HF causal LM.

Prerequisites:
  1. Base model fine-tuned on single-block FDM (Paper 1)
  2. Base model fine-tuned on two-block FDM (fdm_two_block_training.py)

This script then trains a write head for Block A with Block B context present.

Usage:
    # Qwen3 (already have two-block model)
    python fdm_split_source_unified.py \\
        --model /workspace/FDM_IN_WEIGHTS/two_block_model \\
        --n_steps 10000 --tag qwen3

    # Hermes3 (after running two-block training)
    python fdm_split_source_unified.py \\
        --model /workspace/FDM_IN_WEIGHTS/two_block_hermes3 \\
        --n_steps 10000 --tag hermes3

    # LFM2.5
    python fdm_split_source_unified.py \\
        --model /workspace/FDM_IN_WEIGHTS/two_block_lfm \\
        --n_steps 10000 --tag lfm25
"""

import sys, os, re, json, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache

sys.path.insert(0, '/root/FDM_IN_WEIGHTS/scripts')
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]

# Default split: 16 parametric / 16 context
BLOCK_A_CHANNELS = list(range(8, 24))
BLOCK_B_CHANNELS = list(range(24, 40))
ALL_CHANNELS = list(range(8, 40))


# ============================================================
# Auto-detect model architecture
# ============================================================

def detect_architecture(model, config, device):
    """Probe the model to get actual KV cache shape."""
    info = {
        'model_type': getattr(config, 'model_type', 'unknown'),
        'hidden_size': config.hidden_size,
        'num_attention_heads': config.num_attention_heads,
    }

    # Get num_kv_heads (varies by architecture)
    if hasattr(config, 'num_key_value_heads') and config.num_key_value_heads:
        info['num_kv_heads'] = config.num_key_value_heads
    elif hasattr(config, 'num_kv_heads'):
        info['num_kv_heads'] = config.num_kv_heads
    else:
        info['num_kv_heads'] = config.num_attention_heads  # MHA fallback

    # Get num_layers
    if hasattr(config, 'num_hidden_layers'):
        info['num_layers'] = config.num_hidden_layers
    elif hasattr(config, 'n_layer'):
        info['num_layers'] = config.n_layer
    else:
        info['num_layers'] = 28  # fallback

    # Probe actual KV dim by running a forward pass
    try:
        dummy = torch.tensor([[1, 2, 3]], device=device)
        with torch.no_grad():
            out = model(dummy, use_cache=True)

        # Extract KV shape from past_key_values
        pkv = out.past_key_values
        if hasattr(pkv, 'key_cache'):
            # DynamicCache
            k0 = pkv.key_cache[0]
        elif hasattr(pkv, 'layers'):
            # Older DynamicCache
            k0 = pkv.layers[0].keys if hasattr(pkv.layers[0], 'keys') else pkv[0][0]
        elif isinstance(pkv, (list, tuple)):
            k0 = pkv[0][0]
        else:
            k0 = pkv[0][0]

        # k0 shape: (batch, n_kv_heads, seq_len, head_dim)
        info['kv_dim'] = k0.shape[-1]
        info['num_kv_heads'] = k0.shape[1]
        info['probed'] = True
    except Exception as e:
        print(f"  WARNING: Could not probe KV shape: {e}")
        info['kv_dim'] = config.hidden_size // config.num_attention_heads
        info['probed'] = False

    return info


def layer_kvs_to_cache(layer_kvs, n_layers):
    cache = DynamicCache()
    for l in range(n_layers):
        k, v = layer_kvs[l]
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
# Write Head (architecture-adaptive)
# ============================================================

class CrossAttentionWriteHead(nn.Module):
    """Architecture-adaptive write head. Configures to any model's KV shape."""

    def __init__(self, n_channels, n_values=64, embed_dim=384,
                 n_kv_positions=513, n_layers=28, kv_dim=128, n_kv_heads=8):
        super().__init__()
        self.n_layers = n_layers
        self.n_kv_positions = n_kv_positions
        self.kv_dim = kv_dim
        self.n_kv_heads = n_kv_heads
        self.n_channels = n_channels

        self.channel_embed = nn.Embedding(n_channels, embed_dim)
        self.value_embed = nn.Embedding(n_values + 1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=6, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.refiner = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_embed = nn.Parameter(torch.randn(n_kv_positions, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )

        total_kv_dim = n_layers * 2 * n_kv_heads * kv_dim
        self.kv_proj = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, total_kv_dim)
        )

    def forward(self, channel_ids, value_ids):
        B = channel_ids.shape[0]
        ch_emb = self.channel_embed(channel_ids)
        val_emb = self.value_embed(value_ids)
        x = ch_emb + val_emb
        x = self.refiner(x)
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.cross_attn(pos, x, x)
        kv_raw = self.kv_proj(attended)
        kv_raw = kv_raw.view(B, self.n_kv_positions, self.n_layers, 2,
                             self.n_kv_heads, self.kv_dim)
        layer_kvs = {}
        for l in range(self.n_layers):
            k = kv_raw[:, :, l, 0, :, :]
            v = kv_raw[:, :, l, 1, :, :]
            layer_kvs[l] = (k, v)
        return layer_kvs


# ============================================================
# Generation
# ============================================================

def generate_with_hybrid(model, tokenizer, write_head, memory, encoder, device,
                          block_a_channels, block_b_channels, max_new_tokens=350):
    """Generate answer using split-source hybrid with arbitrary channel split."""

    if len(block_a_channels) > 0 and write_head is not None:
        ch_ids = torch.tensor([[i for i in range(len(block_a_channels))]], device=device)
        val_ids = []
        for c in block_a_channels:
            val = memory[c]
            val_list = MEMORY_SCHEMAS[c][1]
            val_idx = val_list.index(val) if val in val_list else 0
            val_ids.append(val_idx)
        val_ids = torch.tensor([val_ids], device=device)

        layer_kvs = write_head(ch_ids, val_ids)
        past_kv = layer_kvs_to_cache(layer_kvs, write_head.n_layers)
        n_kv_pos = write_head.n_kv_positions
    else:
        past_kv = None
        n_kv_pos = 0

    # Build prompt
    fdm_text, _ = encoder.encode_memory(memory)
    ch_names = [CHANNEL_NAMES[k] for k in ALL_CHANNELS]
    question = f"Report values for: {', '.join(ch_names)}."

    if len(block_b_channels) > 0:
        prompt = f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"\nQuestion: {question}\nAnswer:"

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]

    if n_kv_pos > 0:
        position_ids = torch.arange(n_kv_pos, n_kv_pos + seq_len, device=device).unsqueeze(0)
    else:
        position_ids = None  # Let model handle default positions

    generated = input_ids
    cur_past = past_kv
    cur_pos = position_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if cur_past is not None and generated.shape[1] > input_ids.shape[1]:
                last_tok = generated[:, -1:]
                last_pos = cur_pos[:, -1:] + 1 if cur_pos is not None else None
                kwargs = dict(input_ids=last_tok, past_key_values=cur_past, use_cache=True)
                if last_pos is not None:
                    kwargs['position_ids'] = last_pos
                outputs = model(**kwargs)
                cur_pos = last_pos
            else:
                kwargs = dict(input_ids=generated, use_cache=True)
                if cur_past is not None:
                    kwargs['past_key_values'] = cur_past
                if cur_pos is not None:
                    kwargs['position_ids'] = cur_pos
                outputs = model(**kwargs)

            cur_past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# Training
# ============================================================

def train_split_source(model, tokenizer, write_head, encoder, device,
                        block_a_channels, block_b_channels,
                        n_steps=10000, lr=3e-4, eval_every=2000, n_eval=20):
    """Train write head for Block A with Block B context present."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    write_head.train()
    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr / 20)

    t0 = time.time()
    losses = []

    for step in range(1, n_steps + 1):
        mem = random_memory()

        # Write head input
        n_param = len(block_a_channels)
        ch_ids = torch.tensor([[i for i in range(n_param)]], device=device)
        val_ids = []
        for c in block_a_channels:
            val = mem[c]
            val_list = MEMORY_SCHEMAS[c][1]
            val_idx = val_list.index(val) if val in val_list else 0
            val_ids.append(val_idx)
        val_ids = torch.tensor([val_ids], device=device)

        layer_kvs = write_head(ch_ids, val_ids)

        # Block B context
        fdm_text, _ = encoder.encode_memory(mem)

        ch_names = [CHANNEL_NAMES[k] for k in ALL_CHANNELS]
        question = f"Report values for: {', '.join(ch_names)}."
        parts = [f"{CHANNEL_NAMES[k]}={mem[k]}" for k in ALL_CHANNELS]
        answer = "Context: " + ", ".join(parts) + "."

        q_text = f"\nQuestion: {question}\nAnswer:"
        a_text = f" {answer}"

        if len(block_b_channels) > 0:
            prompt = f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]{q_text}{a_text}"
        else:
            prompt = f"{q_text}{a_text}"

        full_ids = tokenizer.encode(prompt, add_special_tokens=False)
        a_ids = tokenizer.encode(a_text, add_special_tokens=False)
        prefix_len = len(full_ids) - len(a_ids)
        labels = [-100] * prefix_len + full_ids[prefix_len:]

        max_len = 1536
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
            labels = labels[:max_len]

        input_tensor = torch.tensor([full_ids], device=device)
        label_tensor = torch.tensor([labels], device=device)
        seq_len = input_tensor.shape[1]
        n_kv_pos = write_head.n_kv_positions

        position_ids = torch.arange(n_kv_pos, n_kv_pos + seq_len,
                                     device=device).unsqueeze(0)
        past_kv = layer_kvs_to_cache(layer_kvs, write_head.n_layers)

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
            evaluate_split_source(model, tokenizer, write_head, encoder, device,
                                   block_a_channels, block_b_channels, n_eval)
            write_head.train()


def evaluate_split_source(model, tokenizer, write_head, encoder, device,
                           block_a_channels, block_b_channels, n_eval=20):
    """Evaluate split-source hybrid.

    Reports BOTH slot-level (per-channel mean, inflated n) and
    sample-level joint accuracy (all channels correct on same sample, honest n).
    """
    if write_head is not None:
        write_head.eval()

    block_a_correct = 0; block_a_total = 0
    block_b_correct = 0; block_b_total = 0

    sample_a_all = 0
    sample_b_all = 0
    sample_all_all = 0
    n_samples = 0

    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        answer = generate_with_hybrid(model, tokenizer, write_head, mem, encoder, device,
                                       block_a_channels, block_b_channels)

        a_all = True
        b_all = True

        for k in block_a_channels:
            pat = r"" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r""
            hit = re.search(pat, answer) is not None
            if hit:
                block_a_correct += 1
            else:
                a_all = False
            block_a_total += 1

        for k in block_b_channels:
            pat = r"" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r""
            hit = re.search(pat, answer) is not None
            if hit:
                block_b_correct += 1
            else:
                b_all = False
            block_b_total += 1

        if a_all:
            sample_a_all += 1
        if b_all:
            sample_b_all += 1
        if a_all and b_all:
            sample_all_all += 1
        n_samples += 1

    a_slot = block_a_correct / block_a_total if block_a_total else 0.0
    b_slot = block_b_correct / block_b_total if block_b_total else 0.0
    overall_slot = (block_a_correct + block_b_correct) / (block_a_total + block_b_total) if (block_a_total + block_b_total) else 0.0

    a_joint = sample_a_all / n_samples if n_samples else 0.0
    b_joint = sample_b_all / n_samples if n_samples else 0.0
    all_joint = sample_all_all / n_samples if n_samples else 0.0

    n_a = len(block_a_channels)
    n_b = len(block_b_channels)
    slot_total = block_a_total + block_b_total

    print("")
    print("    --- SLOT-LEVEL (per-channel mean, inflated n) ---")
    print("    Block A slot-level:  {:.1f}%  (n={} = {} ch x {} samples)".format(a_slot*100, block_a_total, n_a, n_samples))
    print("    Block B slot-level:  {:.1f}%  (n={} = {} ch x {} samples)".format(b_slot*100, block_b_total, n_b, n_samples))
    print("    Overall slot-level:  {:.1f}%  (n={})".format(overall_slot*100, slot_total))
    print("    --- SAMPLE-LEVEL JOINT (honest n) ---")
    print("    Block A joint (all {} correct/sample): {:.1f}%  (n={})".format(n_a, a_joint*100, n_samples))
    print("    Block B joint (all {} correct/sample): {:.1f}%  (n={})".format(n_b, b_joint*100, n_samples))
    print("    ALL joint (all 32 correct/sample): {:.1f}%  (n={})".format(all_joint*100, n_samples))

    return {
        "a_slot": a_slot,
        "b_slot": b_slot,
        "overall_slot": overall_slot,
        "block_a_correct": block_a_correct,
        "block_a_total": block_a_total,
        "block_b_correct": block_b_correct,
        "block_b_total": block_b_total,
        "a_joint": a_joint,
        "b_joint": b_joint,
        "all_joint": all_joint,
        "sample_a_all": sample_a_all,
        "sample_b_all": sample_b_all,
        "sample_all_all": sample_all_all,
        "n_samples": n_samples,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified split-source hybrid for any HF causal LM")
    parser.add_argument("--model", required=True,
                        help="Path to two-block trained model or HF model ID")
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training and only run evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--n_param_channels", type=int, default=16,
                        help="Number of channels in parametric write head (default: 16)")
    parser.add_argument("--tag", default="model",
                        help="Tag for output directory and log (e.g. qwen3, hermes3, lfm25)")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory")
    parser.add_argument("--embed_dim", type=int, default=384,
                        help="Write head embedding dimension")
    parser.add_argument("--n_kv_positions", type=int, default=513,
                        help="Number of KV positions for write head")
    args = parser.parse_args()


    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print("[seed] Set random seed to {}".format(args.seed))

    if args.skip_train:
        args.n_steps = 0
        print("[skip_train] Training disabled, eval-only mode.")

    if args.output_dir is None:
        args.output_dir = f"/workspace/FDM_IN_WEIGHTS/split_source_{args.tag}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Channel split
    n_param = args.n_param_channels
    n_ctx = 32 - n_param
    block_a = list(range(8, 8 + n_param))
    block_b = list(range(8 + n_param, 40))

    print("=" * 60)
    print(f"  SPLIT-SOURCE HYBRID [{args.tag.upper()}]")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Block A (parametric): channels {block_a[0]}-{block_a[-1]} ({n_param} ch)")
    print(f"  Block B (context):    channels {block_b[0]}-{block_b[-1]} ({n_ctx} ch)")
    print(f"  Steps: {args.n_steps}")
    print(f"  LR: {args.lr}")
    print(f"  Tag: {args.tag}", flush=True)

    # Load model
    print(f"\n  Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Detect architecture
    arch = detect_architecture(model, config, device)
    print(f"\n  Detected architecture:")
    print(f"    Model type:     {arch['model_type']}")
    print(f"    Hidden size:    {arch['hidden_size']}")
    print(f"    Num layers:     {arch['num_layers']}")
    print(f"    Num KV heads:   {arch['num_kv_heads']}")
    print(f"    KV head dim:    {arch['kv_dim']}")
    print(f"    Probed:         {arch['probed']}", flush=True)

    encoder = make_encoder(tokenizer)

    # Create write head matched to model architecture
    write_head = CrossAttentionWriteHead(
        n_channels=n_param,
        n_values=64,
        embed_dim=args.embed_dim,
        n_kv_positions=args.n_kv_positions,
        n_layers=arch['num_layers'],
        kv_dim=arch['kv_dim'],
        n_kv_heads=arch['num_kv_heads'],
    ).to(device)

    wh_params = sum(p.numel() for p in write_head.parameters())
    print(f"\n  Write head: {wh_params:,} params ({wh_params/1e6:.1f}M)")
    print(f"    n_channels:     {n_param}")
    print(f"    n_kv_positions: {args.n_kv_positions}")
    print(f"    n_layers:       {arch['num_layers']}")
    print(f"    kv_dim:         {arch['kv_dim']}")
    print(f"    n_kv_heads:     {arch['num_kv_heads']}")

    # Initial eval
    print(f"\n  Initial evaluation (random write head):", flush=True)
    evaluate_split_source(model, tokenizer, write_head, encoder, device,
                           block_a, block_b, n_eval=args.n_eval)

    # Train
    print(f"\n{'='*60}")
    print(f"  TRAINING [{args.tag.upper()}]")
    print(f"{'='*60}", flush=True)
    train_split_source(model, tokenizer, write_head, encoder, device,
                        block_a, block_b,
                        n_steps=args.n_steps, lr=args.lr,
                        eval_every=args.eval_every, n_eval=args.n_eval)

    # Final eval
    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION [{args.tag.upper()}]")
    print(f"{'='*60}", flush=True)
    a_acc, b_acc, all_acc = evaluate_split_source(
        model, tokenizer, write_head, encoder, device,
        block_a, block_b, n_eval=args.n_eval)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(write_head.state_dict(),
               os.path.join(args.output_dir, f"split_source_write_head_{args.tag}.pt"))

    results = {
        "tag": args.tag,
        "block_a_acc": float(a_acc),
        "block_b_acc": float(b_acc),
        "all_acc": float(all_acc),
        "architecture": arch,
        "config": {
            "n_steps": args.n_steps, "lr": args.lr,
            "base_model": args.model,
            "n_param_channels": n_param,
            "block_a_channels": block_a,
            "block_b_channels": block_b,
            "embed_dim": args.embed_dim,
            "n_kv_positions": args.n_kv_positions,
        }
    }
    results_path = os.path.join(args.output_dir, f"results_{args.tag}.json")
    json.dump(results, open(results_path, "w"), indent=2)

    print(f"\n  Saved to {args.output_dir}")
    print(f"  Block A (parametric): {a_acc*100:.1f}%")
    print(f"  Block B (context):    {b_acc*100:.1f}%")
    print(f"  ALL channels:         {all_acc*100:.1f}%")


if __name__ == "__main__":
    main()
