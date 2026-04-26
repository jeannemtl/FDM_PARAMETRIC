"""
FDM Episodic Memory Accumulation
=================================

The continual learning experiment: can a write head accumulate facts
across episodes without catastrophic forgetting, using FDM carrier
orthogonality as the anti-forgetting mechanism?

Design:
  - N episodes, each with a FIXED set of 40 channel values
  - After each episode: train write head to produce correct KV cache
    for that episode's facts
  - After training: evaluate ALL previous episodes
  - If carrier orthogonality holds in KV space: old episodes stay accurate
  - If it doesn't: old episodes degrade (catastrophic forgetting)

This directly tests the Memento essay's claim: can the model compress
new experience into parameters without destroying what it already knows?

The write head's parameters are the "memory" — they must encode ALL
episodes simultaneously. The FDM structure (orthogonal carriers at
different frequencies) is what should prevent interference.

Additionally tests:
  - Fixed vs growing parameter count
  - Per-channel accuracy across episodes (frequency gradient in weights?)
  - Selective channel update (change ch5 in episode 3, verify ch0-4 intact)

Usage:
    python fdm_episodic.py --n_episodes 5 --steps_per_episode 1000
    python fdm_episodic.py --n_episodes 10 --steps_per_episode 2000
"""

import sys, os, re, json, random, time, argparse, copy
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
# Write Head (same architecture as fdm_write_head.py)
# ============================================================

class FDMMemoryLayer(nn.Module):
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
            dropout=0.0, batch_first=True, activation='gelu'
        )
        self.channel_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)

        self.kv_per_pos = n_layers * 2 * n_kv_heads * head_dim
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.channel_to_pos = nn.Linear(embed_dim, embed_dim)
        self.pos_to_kv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.kv_per_pos),
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

    def to_dynamic_cache(self, kv, device):
        from transformers import DynamicCache
        cache = DynamicCache()
        for layer_idx in range(self.n_layers):
            k = kv[layer_idx, 0].half()
            v = kv[layer_idx, 1].half()
            cache.update(k, v, layer_idx)
        return cache


# ============================================================
# Target KV from real FDM
# ============================================================

def get_target_kv(model, tokenizer, encoder, memory, device):
    fdm_text, fdm_tokens = encoder.encode_memory(memory)
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    full_ids = memory_start + fdm_tokens
    ids_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(ids_tensor, use_cache=True)
    kv_cache = out.past_key_values
    kv_list = []
    for layer_idx in range(len(kv_cache.layers)):
        layer = kv_cache.layers[layer_idx]
        kv_list.append((layer.keys.detach(), layer.values.detach()))
    return kv_list, len(full_ids)


# ============================================================
# Generate with write head KV
# ============================================================

def generate_with_write_head(model, tokenizer, write_head, memory, device):
    """Generate model output using write head's KV cache (no FDM encoder)."""
    ch_ids, val_ids = memory_to_indices(memory)

    with torch.no_grad():
        pred_kv_raw = write_head(ch_ids, val_ids, device)
        pred_cache = write_head.to_dynamic_cache(pred_kv_raw, device)

    fdm_seq_len = pred_kv_raw.shape[4]

    question = "Report all context values for channels 8-39."
    rest_prompt = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
    rest_ids = tokenizer.encode(rest_prompt, add_special_tokens=False)
    rest_tensor = torch.tensor([rest_ids], dtype=torch.long).to(device)

    position_ids = torch.arange(fdm_seq_len, fdm_seq_len + len(rest_ids),
                                 device=device).unsqueeze(0)
    attn_mask = torch.ones(1, fdm_seq_len + len(rest_ids),
                            device=device, dtype=torch.long)

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

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ============================================================
# Evaluate write head on a specific episode's facts
# ============================================================

def evaluate_episode(model, tokenizer, write_head, memory, device, n_eval=20):
    """
    Evaluate retrieval accuracy for a FIXED set of channel values.
    Runs n_eval times (same memory each time) to average out generation variance.
    """
    write_head.eval()
    channel_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0

    for _ in range(n_eval):
        generated = generate_with_write_head(model, tokenizer, write_head, memory, device)
        scores = score_extra_channels(generated, memory)
        for name, correct in scores.items():
            channel_correct[name] += int(correct)
        total += 1

    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    return overall, per_ch


# ============================================================
# Train write head on ONE episode's facts
# ============================================================

def train_episode(model, tokenizer, encoder, write_head, memory,
                  device, n_steps, lr, optimizer=None):
    """
    Train write head to produce correct KV for a specific memory configuration.
    Uses the same memory every step (memorize this specific set of facts).
    """
    model.eval()
    write_head.train()

    if optimizer is None:
        optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=1e-4)

    # Pre-compute target KV (same every step for this episode)
    target_kv, seq_len = get_target_kv(model, tokenizer, encoder, memory, device)
    ch_ids, val_ids = memory_to_indices(memory)

    losses = []
    for step in range(1, n_steps + 1):
        pred_kv_raw = write_head(ch_ids, val_ids, device)

        loss = 0.0
        n_layers = len(target_kv)
        for layer_idx in range(n_layers):
            tgt_k, tgt_v = target_kv[layer_idx]
            pred_k = pred_kv_raw[layer_idx, 0]
            pred_v = pred_kv_raw[layer_idx, 1]
            min_len = min(tgt_k.shape[2], pred_k.shape[2])
            loss += F.mse_loss(pred_k[:, :, :min_len, :], tgt_k[:, :, :min_len, :].float())
            loss += F.mse_loss(pred_v[:, :, :min_len, :], tgt_v[:, :, :min_len, :].float())
        loss = loss / (2 * n_layers)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(write_head.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 200 == 0 or step == n_steps:
            avg = np.mean(losses[-200:])
            print(f"    Step {step:5d}/{n_steps} | loss {avg:.6f}")

    return optimizer


# ============================================================
# Episodic accumulation experiment
# ============================================================

def run_episodic_experiment(model, tokenizer, encoder, write_head, device,
                            n_episodes, steps_per_episode, lr, n_eval_per_episode):
    """
    The main experiment:
    1. Generate N episodes, each with fixed random channel values
    2. For each episode: train write head, then evaluate ALL episodes
    3. Track whether old episodes degrade (catastrophic forgetting)
    """

    print("\n" + "=" * 60)
    print(f"EPISODIC ACCUMULATION: {n_episodes} episodes")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Eval samples per episode: {n_eval_per_episode}")
    print("=" * 60)

    # Generate fixed memories for each episode
    random.seed(42)
    episodes = []
    for ep in range(n_episodes):
        mem = random_memory()
        episodes.append(mem)
        vals_preview = [f"{CHANNEL_NAMES[k]}={mem[k]}" for k in [8, 9, 10]]
        print(f"  Episode {ep}: {', '.join(vals_preview)}, ...")

    # Accumulation matrix: accuracy[ep_trained][ep_evaluated]
    # After training on episodes 0..i, evaluate accuracy on episode j
    accuracy_matrix = np.zeros((n_episodes, n_episodes))

    # Single optimizer across all episodes (continual learning)
    optimizer = torch.optim.AdamW(write_head.parameters(), lr=lr, weight_decay=1e-4)

    results_log = []

    for train_ep in range(n_episodes):
        print(f"\n{'─' * 60}")
        print(f"EPISODE {train_ep}: Training on new facts")
        print(f"{'─' * 60}")

        mem = episodes[train_ep]

        # Train on this episode's facts
        optimizer = train_episode(
            model, tokenizer, encoder, write_head, mem,
            device, n_steps=steps_per_episode, lr=lr, optimizer=optimizer,
        )

        # Evaluate ALL episodes seen so far
        print(f"\n  Evaluating episodes 0..{train_ep}:")
        ep_results = {}

        for eval_ep in range(train_ep + 1):
            eval_mem = episodes[eval_ep]
            acc, per_ch = evaluate_episode(
                model, tokenizer, write_head, eval_mem, device,
                n_eval=n_eval_per_episode,
            )
            accuracy_matrix[train_ep][eval_ep] = acc

            tag = "← current" if eval_ep == train_ep else ""
            print(f"    Episode {eval_ep}: {acc*100:.1f}% {tag}")
            ep_results[f"episode_{eval_ep}"] = acc

        # Check forgetting
        if train_ep > 0:
            prev_accs = [accuracy_matrix[train_ep][j] for j in range(train_ep)]
            orig_accs = [accuracy_matrix[j][j] for j in range(train_ep)]
            avg_retention = np.mean(prev_accs)
            avg_original = np.mean(orig_accs)
            forgetting = avg_original - avg_retention
            print(f"\n    Avg retention of old episodes: {avg_retention*100:.1f}%")
            print(f"    Avg original accuracy:         {avg_original*100:.1f}%")
            print(f"    Forgetting:                    {forgetting*100:+.1f}%")
            ep_results["avg_retention"] = float(avg_retention)
            ep_results["forgetting"] = float(forgetting)

        results_log.append(ep_results)

    # ============================================================
    # Final summary
    # ============================================================

    print("\n" + "=" * 60)
    print("ACCUMULATION MATRIX")
    print("  Rows: after training on episodes 0..i")
    print("  Cols: accuracy on episode j")
    print("=" * 60)

    header = "          " + "".join(f"  Ep{j:2d}" for j in range(n_episodes))
    print(header)
    for i in range(n_episodes):
        row = f"After Ep{i}: "
        for j in range(n_episodes):
            if j <= i:
                row += f" {accuracy_matrix[i][j]*100:5.1f}%"
            else:
                row += "     - "
        print(row)

    # Diagonal = accuracy right after training on that episode
    diag = [accuracy_matrix[i][i] for i in range(n_episodes)]
    # Final row = accuracy of all episodes after all training
    final_row = [accuracy_matrix[n_episodes-1][j] for j in range(n_episodes)]

    print(f"\n  Diagonal (immediate accuracy): {[f'{d*100:.1f}%' for d in diag]}")
    print(f"  Final row (after all training): {[f'{d*100:.1f}%' for d in final_row]}")
    print(f"  Avg diagonal:  {np.mean(diag)*100:.1f}%")
    print(f"  Avg final row: {np.mean(final_row)*100:.1f}%")
    print(f"  Total forgetting: {(np.mean(diag) - np.mean(final_row))*100:.1f}%")

    # Selective channel update test
    print("\n" + "=" * 60)
    print("SELECTIVE UPDATE TEST")
    print("  Change ONE channel in episode 0, verify others intact")
    print("=" * 60)

    original_mem = copy.deepcopy(episodes[0])
    modified_mem = copy.deepcopy(episodes[0])

    # Change channel 20 (CIPHER) to a different value
    ch20_vals = MEMORY_SCHEMAS[20][1]
    current_val = modified_mem[20]
    new_val = [v for v in ch20_vals if v != current_val][0]
    modified_mem[20] = new_val
    print(f"  Changed CIPHER: {current_val} → {new_val}")

    # Train briefly on modified memory
    write_head_backup = copy.deepcopy(write_head.state_dict())
    train_episode(model, tokenizer, encoder, write_head, modified_mem,
                  device, n_steps=500, lr=lr * 0.5)

    # Evaluate: channel 20 should change, others should stay
    _, per_ch_modified = evaluate_episode(
        model, tokenizer, write_head, modified_mem, device, n_eval=20
    )
    _, per_ch_original = evaluate_episode(
        model, tokenizer, write_head, original_mem, device, n_eval=20
    )

    print(f"\n  CIPHER accuracy (modified facts): {per_ch_modified.get('CIPHER', 0)*100:.1f}%")
    print(f"  CIPHER accuracy (original facts): {per_ch_original.get('CIPHER', 0)*100:.1f}%")

    other_modified = [v for k, v in per_ch_modified.items() if k != 'CIPHER']
    other_original = [v for k, v in per_ch_original.items() if k != 'CIPHER']
    print(f"  Other channels (modified query): {np.mean(other_modified)*100:.1f}%")
    print(f"  Other channels (original query): {np.mean(other_original)*100:.1f}%")
    drift = abs(np.mean(other_modified) - np.mean(other_original))
    print(f"  Cross-channel drift: {drift*100:.2f}%")

    # Restore write head
    write_head.load_state_dict(write_head_backup)

    return accuracy_matrix, results_log


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--n_eval", type=int, default=10,
                        help="Eval samples per episode (same memory repeated)")
    parser.add_argument("--lr", type=float, default=5e-5)
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

    print(f"  Loaded. Layers: {model.config.num_hidden_layers}")

    encoder = make_encoder(tokenizer)

    # Get seq_len from sample
    mem = random_memory()
    _, fdm_tokens = encoder.encode_memory(mem)
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    sample_seq_len = len(memory_start) + len(fdm_tokens)

    n_kv_heads = model.config.num_key_value_heads
    head_dim = 128  # actual KV head dim
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
    print(f"  Write head: {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"  Seq len: {sample_seq_len}")

    acc_matrix, results = run_episodic_experiment(
        model, tokenizer, encoder, write_head, device,
        n_episodes=args.n_episodes,
        steps_per_episode=args.steps_per_episode,
        lr=args.lr,
        n_eval_per_episode=args.n_eval,
    )

    # Save
    out = {
        "accuracy_matrix": acc_matrix.tolist(),
        "results_log": results,
        "config": {
            "n_episodes": args.n_episodes,
            "steps_per_episode": args.steps_per_episode,
            "lr": args.lr,
            "write_head_params": n_params,
        },
    }
    json.dump(out, open("fdm_episodic_results.json", "w"), indent=2, default=float)
    print("\nSaved to fdm_episodic_results.json")


if __name__ == "__main__":
    main()
