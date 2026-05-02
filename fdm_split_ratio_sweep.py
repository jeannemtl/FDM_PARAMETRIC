"""
Split-Ratio Sweep: How does accuracy change as more channels go parametric?
============================================================================

Uses the trained split-source write head and two-block base model.
Tests different partitions of the 32 channels (8-39) between:
  - Block A (parametric write head KV at positions 0-512)
  - Block B (FDM context tokens at positions 513+)

Sweep configurations:
  - 0/32:  All context, no write head (baseline)
  - 4/28:  4 parametric, 28 context
  - 8/24:  8 parametric, 24 context
  - 12/20: 12 parametric, 20 context
  - 16/16: 16 parametric, 16 context (original split)
  - 20/12: 20 parametric, 12 context
  - 24/8:  24 parametric, 8 context
  - 28/4:  28 parametric, 4 context
  - 32/0:  All parametric, no context (standalone write head)

For each config, trains a fresh write head for 5K steps with context present,
then evaluates Block A accuracy, Block B accuracy, and joint accuracy.

This tests Claims 1 (split-source), 7 (capacity scaling), and 12 (tiered memory).
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
ALL_QUERY_CHANNELS = list(range(8, 40))  # 32 channels total


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


class CrossAttentionWriteHead(nn.Module):
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


def generate_with_split(model, tokenizer, write_head, memory, encoder, device,
                         block_a_channels, block_b_channels, max_new_tokens=350):
    """Generate answer with arbitrary split between parametric and context."""

    if len(block_a_channels) > 0:
        # Write head input
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
    if len(block_b_channels) > 0:
        fdm_text, _ = encoder.encode_memory(memory)
        memory_part = f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]"
    else:
        memory_part = ""

    ch_names = [CHANNEL_NAMES[k] for k in ALL_QUERY_CHANNELS]
    question = f"Report values for: {', '.join(ch_names)}."
    prompt = f"{memory_part}\nQuestion: {question}\nAnswer:"

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]

    if n_kv_pos > 0:
        position_ids = torch.arange(n_kv_pos, n_kv_pos + seq_len, device=device).unsqueeze(0)
    else:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Generate
    generated = input_ids
    cur_past = past_kv
    cur_pos = position_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if cur_past is not None and generated.shape[1] > input_ids.shape[1]:
                last_tok = generated[:, -1:]
                last_pos = cur_pos[:, -1:] + 1
                outputs = model(input_ids=last_tok, position_ids=last_pos,
                                past_key_values=cur_past, use_cache=True)
                cur_pos = last_pos
            else:
                kwargs = dict(input_ids=generated, position_ids=cur_pos, use_cache=True)
                if cur_past is not None:
                    kwargs['past_key_values'] = cur_past
                outputs = model(**kwargs)

            cur_past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)


def train_and_eval_split(model, tokenizer, encoder, device,
                          block_a_channels, block_b_channels,
                          n_steps=5000, lr=3e-4, n_eval=20):
    """Train a write head for a specific split, then evaluate."""

    n_param_ch = len(block_a_channels)
    n_ctx_ch = len(block_b_channels)
    print(f"\n{'='*60}")
    print(f"  SPLIT: {n_param_ch} parametric / {n_ctx_ch} context")
    print(f"  Block A (parametric): channels {block_a_channels}")
    print(f"  Block B (context):    channels {block_b_channels}")
    print(f"{'='*60}")

    # Handle edge case: all context, no write head
    if n_param_ch == 0:
        print("  No write head needed - evaluating context-only...")
        slot_correct = 0
        slot_total = 0
        sample_all_correct = 0
        n_samples = 0
        per_sample_correct = []  # list of bools

        for _ in tqdm(range(n_eval), desc="Eval", leave=False):
            mem = random_memory()
            answer = generate_with_split(model, tokenizer, None, mem, encoder, device,
                                          [], block_b_channels)
            sample_all = True
            sample_hits = []
            for k in ALL_QUERY_CHANNELS:
                pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
                hit = re.search(pat, answer) is not None
                sample_hits.append(bool(hit))
                if hit:
                    slot_correct += 1
                else:
                    sample_all = False
                slot_total += 1
            if sample_all:
                sample_all_correct += 1
            n_samples += 1
            per_sample_correct.append(sample_hits)

        slot_acc = slot_correct / slot_total if slot_total > 0 else 0.0
        joint_acc = sample_all_correct / n_samples if n_samples > 0 else 0.0
        print("  Slot-level (per-channel mean): {:.1f}%  (n={})".format(slot_acc*100, slot_total))
        print("  Sample-level joint (all 32 correct): {:.1f}%  (n={})".format(joint_acc*100, n_samples))
        return {
            "block_a_acc": 0.0,
            "block_b_acc": slot_acc,
            "all_acc": slot_acc,           # legacy slot-level field, kept for compatibility
            "all_acc_slot": slot_acc,
            "all_acc_joint": joint_acc,
            "n_param": 0,
            "n_ctx": n_ctx_ch,
            "n_samples": n_samples,
            "slot_correct": slot_correct,
            "slot_total": slot_total,
            "sample_all_correct": sample_all_correct,
            "per_sample_correct": per_sample_correct,
        }


    # Create write head
    write_head = CrossAttentionWriteHead(
        n_channels=n_param_ch, n_values=64, embed_dim=384,
        n_kv_positions=513, n_layers=28, kv_dim=128, n_kv_heads=8
    ).to(device)

    wh_params = sum(p.numel() for p in write_head.parameters())
    print(f"  Write head: {wh_params/1e6:.1f}M params for {n_param_ch} channels")

    # Train
    write_head.train()
    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr/20)

    t0 = time.time()
    losses = []

    for step in range(1, n_steps + 1):
        mem = random_memory()

        # Write head input
        ch_ids = torch.tensor([[i for i in range(n_param_ch)]], device=device)
        val_ids = []
        for c in block_a_channels:
            val = mem[c]
            val_list = MEMORY_SCHEMAS[c][1]
            val_idx = val_list.index(val) if val in val_list else 0
            val_ids.append(val_idx)
        val_ids = torch.tensor([val_ids], device=device)

        layer_kvs = write_head(ch_ids, val_ids)

        # Build training prompt
        if n_ctx_ch > 0:
            fdm_text, _ = encoder.encode_memory(mem)
            memory_part = f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]"
        else:
            memory_part = ""

        ch_names = [CHANNEL_NAMES[k] for k in ALL_QUERY_CHANNELS]
        question = f"Report values for: {', '.join(ch_names)}."
        parts = [f"{CHANNEL_NAMES[k]}={mem[k]}" for k in ALL_QUERY_CHANNELS]
        answer_text = "Context: " + ", ".join(parts) + "."

        q_text = f"\nQuestion: {question}\nAnswer:"
        a_text = f" {answer_text}"
        prompt = f"{memory_part}{q_text}{a_text}"

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

        position_ids = torch.arange(n_kv_pos, n_kv_pos + seq_len, device=device).unsqueeze(0)
        past_kv = layer_kvs_to_cache(layer_kvs, write_head.n_layers)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            outputs = model(input_ids=input_tensor, position_ids=position_ids,
                            past_key_values=past_kv, labels=label_tensor, use_cache=False)
            loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 1000 == 0:
            avg = np.mean(losses[-1000:])
            elapsed = time.time() - t0
            print(f"    Step {step:5d}/{n_steps} | loss {avg:.4f} | {elapsed:.0f}s")

    # Evaluate
    write_head.eval()
    block_a_slot_correct = 0; block_a_slot_total = 0
    block_b_slot_correct = 0; block_b_slot_total = 0
    all_slot_correct = 0;     all_slot_total = 0

    sample_a_all = 0
    sample_b_all = 0
    sample_all_all = 0
    n_samples = 0
    per_sample_a = []
    per_sample_b = []
    per_sample_all = []

    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        answer = generate_with_split(model, tokenizer, write_head, mem, encoder, device,
                                      block_a_channels, block_b_channels)

        a_all = True
        b_all = True

        for k in block_a_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                block_a_slot_correct += 1
            else:
                a_all = False
            block_a_slot_total += 1

        for k in block_b_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                block_b_slot_correct += 1
            else:
                b_all = False
            block_b_slot_total += 1

        for k in ALL_QUERY_CHANNELS:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                all_slot_correct += 1
            all_slot_total += 1

        if a_all:
            sample_a_all += 1
        if b_all:
            sample_b_all += 1
        if a_all and b_all:
            sample_all_all += 1
        n_samples += 1
        per_sample_a.append(bool(a_all))
        per_sample_b.append(bool(b_all))
        per_sample_all.append(bool(a_all and b_all))

    a_slot = block_a_slot_correct / block_a_slot_total if block_a_slot_total > 0 else 0.0
    b_slot = block_b_slot_correct / block_b_slot_total if block_b_slot_total > 0 else 0.0
    all_slot = all_slot_correct / all_slot_total if all_slot_total > 0 else 0.0

    a_joint = sample_a_all / n_samples if n_samples > 0 else 0.0
    b_joint = sample_b_all / n_samples if n_samples > 0 else 0.0
    all_joint = sample_all_all / n_samples if n_samples > 0 else 0.0

    print("  --- SLOT-LEVEL (per-channel mean) ---")
    print("  Block A: {:.1f}%  (n={})".format(a_slot*100, block_a_slot_total))
    print("  Block B: {:.1f}%  (n={})".format(b_slot*100, block_b_slot_total))
    print("  All:     {:.1f}%  (n={})".format(all_slot*100, all_slot_total))
    print("  --- SAMPLE-LEVEL JOINT (honest n) ---")
    print("  Block A all-correct: {:.1f}%  (n={})".format(a_joint*100, n_samples))
    print("  Block B all-correct: {:.1f}%  (n={})".format(b_joint*100, n_samples))
    print("  All-32 all-correct:  {:.1f}%  (n={})".format(all_joint*100, n_samples))

    return {
        # Legacy fields (slot-level) - kept so existing summary print works
        "block_a_acc": float(a_slot),
        "block_b_acc": float(b_slot),
        "all_acc": float(all_slot),
        # Explicit slot-level
        "block_a_acc_slot": float(a_slot),
        "block_b_acc_slot": float(b_slot),
        "all_acc_slot": float(all_slot),
        # Sample-level joint
        "block_a_acc_joint": float(a_joint),
        "block_b_acc_joint": float(b_joint),
        "all_acc_joint": float(all_joint),
        # Raw counts
        "block_a_slot_correct": block_a_slot_correct,
        "block_a_slot_total": block_a_slot_total,
        "block_b_slot_correct": block_b_slot_correct,
        "block_b_slot_total": block_b_slot_total,
        "sample_a_all_correct": sample_a_all,
        "sample_b_all_correct": sample_b_all,
        "sample_all_all_correct": sample_all_all,
        "n_samples": n_samples,
        # Per-sample arrays for exact Wilson CI computation
        "per_sample_block_a_all": per_sample_a,
        "per_sample_block_b_all": per_sample_b,
        "per_sample_all32_all":   per_sample_all,
        # Other metadata
        "n_param": n_param_ch,
        "n_ctx": n_ctx_ch,
        "final_loss": float(np.mean(losses[-500:])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/FDM_IN_WEIGHTS/two_block_model")
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", default="/workspace/FDM_IN_WEIGHTS/split_ratio_sweep")
    # Which splits to run (comma-separated parametric channel counts)
    parser.add_argument("--splits", default="0,4,8,12,16,20,24,28,32",
                        help="Comma-separated parametric channel counts to test")
    args = parser.parse_args()


    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print("[seed] Random seed set to {}".format(args.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  SPLIT-RATIO SWEEP")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    encoder = make_encoder(tokenizer)

    splits = [int(x) for x in args.splits.split(',')]
    results = []

    for n_param in splits:
        n_ctx = 32 - n_param
        # Assign first n_param channels (8, 9, ..., 8+n_param-1) to write head
        block_a = list(range(8, 8 + n_param))
        block_b = list(range(8 + n_param, 40))

        result = train_and_eval_split(
            model, tokenizer, encoder, device,
            block_a, block_b,
            n_steps=args.n_steps, lr=args.lr, n_eval=args.n_eval
        )
        result['block_a_channels'] = block_a
        result['block_b_channels'] = block_b
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("  SWEEP SUMMARY")
    print("=" * 60)
    print(f"  {'Param':>5} {'Ctx':>5} {'A acc':>7} {'B acc':>7} {'All':>7}")
    print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        print(f"  {r['n_param']:5d} {r['n_ctx']:5d} "
              f"{r['block_a_acc']*100:6.1f}% {r['block_b_acc']*100:6.1f}% "
              f"{r['all_acc']*100:6.1f}%")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(results, open(os.path.join(args.output_dir, "sweep_results.json"), "w"), indent=2)
    print(f"\n  Saved to {args.output_dir}/sweep_results.json")


if __name__ == "__main__":
    main()
