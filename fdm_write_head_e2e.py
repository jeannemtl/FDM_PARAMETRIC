"""
FDM Write Head: End-to-End Curriculum Training
===============================================

MSE on KV values doesn't work (0.97 cosine sim → 0% accuracy).
Instead: train the write head end-to-end on the GENERATION objective.

Pipeline (each training step):
  1. Write head produces KV cache from (channel_ids, value_ids)
  2. Feed question tokens into frozen model with write head's KV prepended
  3. Cross-entropy loss on answer tokens
  4. Gradient flows: loss → frozen attention → write head parameters
  5. Only write head parameters update

This is exactly how the original FDM reader was trained — not by
matching FFT coefficients, but by training on whether the model
gets the right answer. The write head learns the same way.

Curriculum (matching the paper's progression):
  Stage 0: 5 channels (ch8-12), easy — learn basic KV structure
  Stage 1: 10 channels (ch8-17)
  Stage 2: 20 channels (ch8-27)
  Stage 3: All 32 extra channels (ch8-39)

Usage:
    python fdm_write_head_e2e.py --n_steps 5000
    python fdm_write_head_e2e.py --n_steps 10000 --eval_every 1000
"""

import sys, os, re, json, random, time, argparse, copy
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
# Write Head (same architecture, but will be trained E2E)
# ============================================================

class FDMWriteHead(nn.Module):
    """
    Produces KV cache from channel values.
    Trained end-to-end: loss on answer tokens flows back through
    frozen model attention into these parameters.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, seq_len,
                 n_channels=40, max_values=5, embed_dim=256, n_attn_layers=2):
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

        # Learned position embeddings for KV sequence
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.channel_to_pos = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Deeper projection to KV space
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
        """Produce a DynamicCache ready for model.forward()."""
        kv = self.forward(ch_ids, val_ids, device)
        cache = DynamicCache()
        for layer_idx in range(self.n_layers):
            # Cast to model dtype (float16)
            cache.update(kv[layer_idx, 0].half(), kv[layer_idx, 1].half(), layer_idx)
        return cache, kv.shape[4]  # return seq_len too


# ============================================================
# Build answer targets for a given memory config
# ============================================================

def build_answer_for_channels(memory, active_channels):
    """
    Build the expected answer text for a subset of extra channels.
    active_channels: list of channel indices (e.g., [8,9,10,11,12])
    """
    parts = [f"{CHANNEL_NAMES[k]}={memory[k]}" for k in active_channels]
    return "Context: " + ", ".join(parts) + "."


def build_question_for_channels(active_channels):
    """Question that asks for specific channels."""
    ch_names = [CHANNEL_NAMES[k] for k in active_channels]
    return f"Report values for: {', '.join(ch_names)}."


# ============================================================
# End-to-end training step
# ============================================================

def e2e_train_step(model, tokenizer, write_head, memory,
                   active_channels, device):
    """
    One end-to-end training step:
    1. Write head → KV cache
    2. Model forward with KV cache + question → logits
    3. Cross-entropy on answer tokens → gradient → write head
    
    Returns loss value.
    """
    # Build question and answer for active channels
    question = build_question_for_channels(active_channels)
    answer = build_answer_for_channels(memory, active_channels)

    # Tokenize question and answer
    q_text = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
    a_text = f" {answer}"
    full_text = q_text + a_text

    q_ids = tokenizer.encode(q_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    a_len = len(full_ids) - len(q_ids)

    full_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)

    # Write head produces KV cache (DIFFERENTIABLE)
    ch_ids, val_ids = memory_to_indices(memory)

    # We need the KV in float32 for gradient flow, then cast for model
    kv_raw = write_head(ch_ids, val_ids, device)  # float32
    fdm_seq_len = kv_raw.shape[4]

    # Build DynamicCache — but we need gradients to flow through!
    # So we create the cache with float32 KV and cast the model input
    cache = DynamicCache()
    for li in range(write_head.n_layers):
        # Keep as float32 for gradient, model will handle mixed precision
        cache.update(kv_raw[li, 0], kv_raw[li, 1], li)

    # Position IDs and attention mask
    seq_len = len(full_ids)
    position_ids = torch.arange(fdm_seq_len, fdm_seq_len + seq_len,
                                 device=device).unsqueeze(0)
    attn_mask = torch.ones(1, fdm_seq_len + seq_len, device=device, dtype=torch.long)

    # Forward pass through frozen model
    # Note: model is in float16 but KV cache is float32.
    # We need to handle this — cast model to float32 for this pass,
    # or cast KV to float16 but keep gradient.
    # Solution: use autocast
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(
            full_tensor,
            past_key_values=cache,
            position_ids=position_ids,
            attention_mask=attn_mask,
            use_cache=False,  # don't need cache for generation here
        )

    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Build labels: -100 for question tokens, real labels for answer tokens
    labels = torch.full((1, seq_len), -100, dtype=torch.long, device=device)
    labels[0, len(q_ids):] = full_tensor[0, len(q_ids):]

    # Cross-entropy loss on answer tokens only
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    return loss


# ============================================================
# Generation for evaluation
# ============================================================

def generate_with_write_head(model, tokenizer, write_head, memory,
                             active_channels, device):
    """Generate using write head's KV cache."""
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


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, tokenizer, write_head, active_channels, device, n_eval=50):
    """Evaluate write head on random memories."""
    write_head.eval()
    channel_correct = {CHANNEL_NAMES[k]: 0 for k in active_channels}
    total = 0

    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        text = generate_with_write_head(
            model, tokenizer, write_head, mem, active_channels, device
        )
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
# Curriculum Training
# ============================================================

CURRICULUM = [
    {"name": "Stage 0: 5 channels",  "channels": list(range(8, 13)),  "steps": None, "lr_mult": 1.0},
    {"name": "Stage 1: 10 channels", "channels": list(range(8, 18)),  "steps": None, "lr_mult": 0.8},
    {"name": "Stage 2: 20 channels", "channels": list(range(8, 28)),  "steps": None, "lr_mult": 0.6},
    {"name": "Stage 3: 32 channels", "channels": list(range(8, 40)),  "steps": None, "lr_mult": 0.4},
]


def train_curriculum(model, tokenizer, write_head, device,
                     total_steps, lr, eval_every, n_eval):
    """
    Curriculum training: progressively add channels.
    Each stage gets total_steps / len(CURRICULUM) steps.
    """
    steps_per_stage = total_steps // len(CURRICULUM)
    for stage in CURRICULUM:
        stage["steps"] = steps_per_stage

    print("\n" + "=" * 60)
    print(f"E2E CURRICULUM TRAINING ({total_steps} total steps)")
    print("=" * 60)

    # Single optimizer across all stages
    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0
    t0 = time.time()

    for stage_idx, stage in enumerate(CURRICULUM):
        active = stage["channels"]
        n_steps = stage["steps"]
        stage_lr = lr * stage["lr_mult"]

        # Update LR for this stage
        for pg in optimizer.param_groups:
            pg['lr'] = stage_lr

        print(f"\n{'─' * 60}")
        print(f"STAGE {stage_idx}: {stage['name']}")
        print(f"  Channels: {[CHANNEL_NAMES[k] for k in active[:5]]}{'...' if len(active)>5 else ''}")
        print(f"  Steps: {n_steps}, LR: {stage_lr:.1e}")
        print(f"{'─' * 60}")

        losses = []
        write_head.train()

        for step in range(1, n_steps + 1):
            global_step += 1
            mem = random_memory()

            loss = e2e_train_step(model, tokenizer, write_head, mem,
                                  active, device)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            if step % 200 == 0 or step == n_steps:
                avg = np.mean(losses[-200:])
                elapsed = time.time() - t0
                print(f"    Step {step:5d}/{n_steps} (global {global_step}) | "
                      f"loss {avg:.4f} | {elapsed:.0f}s")

            if eval_every > 0 and global_step % eval_every == 0:
                acc, _ = evaluate(model, tokenizer, write_head,
                                  active, device, n_eval=n_eval)
                print(f"    *** Eval ({len(active)} ch): {acc*100:.1f}% ***")
                write_head.train()

        # End of stage eval
        acc, per_ch = evaluate(model, tokenizer, write_head,
                               active, device, n_eval=n_eval)
        print(f"\n  Stage {stage_idx} final: {acc*100:.1f}%")
        worst = sorted(per_ch, key=per_ch.get)[:3]
        for name in worst:
            print(f"    {name}: {per_ch[name]*100:.1f}%")

    return write_head


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_steps", type=int, default=8000,
                        help="Total steps across all curriculum stages")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--checkpoint", default="fdm_write_head_e2e.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16
    ).to(device)
    model.eval()

    # Freeze base model — gradients flow THROUGH it but don't update it
    for p in model.parameters():
        p.requires_grad = False

    encoder = make_encoder(tokenizer)

    # Get seq_len
    mem = random_memory()
    _, fdm_tokens = encoder.encode_memory(mem)
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    sample_seq_len = len(memory_start) + len(fdm_tokens)

    n_kv_heads = model.config.num_key_value_heads
    head_dim = 128  # actual KV head dim
    max_values = max(len(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS))

    write_head = FDMWriteHead(
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
    print(f"  Write head: {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"  Seq len: {sample_seq_len}")
    print(f"  KV heads: {n_kv_heads}, head dim: {head_dim}")

    # Train
    write_head = train_curriculum(
        model, tokenizer, write_head, device,
        total_steps=args.n_steps,
        lr=args.lr,
        eval_every=args.eval_every,
        n_eval=args.n_eval,
    )

    torch.save(write_head.state_dict(), args.checkpoint)
    print(f"\nSaved checkpoint: {args.checkpoint}")

    # Final evaluation on all 32 channels
    print("\n" + "=" * 60)
    print("FINAL EVALUATION: All 32 extra channels")
    print("=" * 60)
    all_channels = list(range(8, 40))
    acc, per_ch = evaluate(model, tokenizer, write_head,
                           all_channels, device, n_eval=50)
    print(f"\n  Overall: {acc*100:.1f}%")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Baseline (FDM in context):    93.7%")
    print(f"  Frozen KV injection:          89.1%")
    print(f"  E2E write head:               {acc*100:.1f}%")

    results = {
        "baseline": 0.937,
        "frozen_kv": 0.891,
        "e2e_write_head": float(acc),
        "per_channel": {k: float(v) for k, v in per_ch.items()},
        "write_head_params": n_params,
        "total_steps": args.n_steps,
    }
    json.dump(results, open("fdm_e2e_results.json", "w"), indent=2)
    print(f"Results saved to fdm_e2e_results.json")


if __name__ == "__main__":
    main()
