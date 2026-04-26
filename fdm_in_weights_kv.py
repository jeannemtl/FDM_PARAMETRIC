"""
FDM-in-Weights via KV Cache Injection
======================================

Takes your existing FDM-trained models (which already know how to read
frequency-multiplexed signals from context tokens) and tests whether
the same signal can be read from a PARAMETER BUFFER injected as
pre-computed KV cache entries — moving facts from context into weights.

Architecture:
  Normal FDM:   [MEMORY] <512 FDM tokens> [/MEMORY] Question: ... Answer: ...
                     ↑ lives in input_ids → embedding → attention

  FDM-in-weights: [MEMORY] [/MEMORY] Question: ... Answer: ...
                     ↑ FDM tokens REMOVED from context
                     ↑ Instead: a learned FDMMemoryLayer produces KV pairs
                       that get prepended to the KV cache, so attention
                       sees them as if they were context — but they live
                       in parameters, not input tokens.

Three experiments:
  1. BASELINE: Normal FDM in context (sanity check, should match paper)
  2. FROZEN INJECTION: Take the embeddings the model would compute for
     FDM tokens, freeze them as a parameter buffer, prepend to KV cache.
     Tests: does the model read identically from cached vs live embeddings?
  3. LEARNED BUFFER: Train a small FDMMemoryLayer that maps
     (channel_id, value) → KV pairs matching what the frozen injection
     produces. This is the write head — it learns to produce the right
     KV representations so the model's existing read mechanism works.

Usage:
    # On your GPU machine with the fine-tuned models:
    python fdm_in_weights_kv.py --model prompterminal/qwen3-0.6b-fdm-v3 \
                                --arch qwen3 \
                                --n_eval 500

    # Or test all architectures:
    python fdm_in_weights_kv.py --model prompterminal/gpt2-medium-fdm-v3 --arch gpt2
    python fdm_in_weights_kv.py --model prompterminal/qwen3-0.6b-fdm-v3 --arch qwen3
    python fdm_in_weights_kv.py --model prompterminal/hermes3-3b-fdm-v3 --arch hermes3

Requirements:
    pip install torch transformers numpy tqdm
"""

import argparse
import math
import re
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
from typing import List, Dict, Tuple, Optional
from copy import deepcopy


# ============================================================
# Channel definitions (from paper Appendix A)
# ============================================================

CHANNEL_VOCAB = [
    ["RED","BLUE","GREEN","GOLD"],                              # 0  SECRET
    ["PARIS","TOKYO","LONDON","BERLIN"],                        # 1  LOCATION
    ["ALICE","BOB","CAROL","DAVE"],                             # 2  AGENT
    ["CLEAR","COMPROMISED","UNKNOWN"],                          # 3  STATUS
    ["HIGH","MEDIUM","LOW"],                                    # 4  PRIORITY
    ["AVAILABLE","UNAVAILABLE"],                                # 5  BACKUP
    ["SAFETY_FIRST","MISSION_FIRST","BALANCED","CAUTIOUS"],     # 6  RULE
    ["NONE","OVERRIDE_STATUS","OVERRIDE_PRIORITY","EMERGENCY","LOCKDOWN"], # 7 META
    ["RED_TEAM","BLUE_TEAM","GREEN_TEAM","GOLD_TEAM"],          # 8  TEAM
    ["NORTH","SOUTH","EAST","WEST"],                            # 9  REGION
    ["ALPHA","BETA","GAMMA","DELTA"],                           # 10 PHASE
    ["OPEN","CLOSED","RESTRICTED"],                             # 11 COMM
    ["VEHICLE","AIRCRAFT","DRONE","BOAT"],                      # 12 ASSET
    ["DAWN","MIDDAY","DUSK","NIGHT"],                           # 13 WINDOW
    ["DEEP","SHALLOW","NONE"],                                  # 14 COVER
    ["ACTIVE","STANDBY","OFFLINE"],                             # 15 SUPPORT
    ["LOW","MEDIUM","HIGH","CRITICAL"],                         # 16 THREAT
    ["CLEAR","STORM","FOG"],                                    # 17 WEATHER
    ["URBAN","RURAL","COASTAL","MOUNTAIN"],                     # 18 TERRAIN
    ["READY","DELAYED","UNAVAILABLE"],                          # 19 EXTRACT
    ["AES","RSA","BLOWFISH","TWOFISH"],                         # 20 CIPHER
    ["HF","VHF","UHF","SHF"],                                   # 21 FREQ
    ["LIGHT","MEDIUM","HEAVY","CRITICAL"],                      # 22 PAYLOAD
    ["ALPHA","BRAVO","CHARLIE","DELTA"],                        # 23 ROUTE
    ["SHORT","MEDIUM","LONG","EXTENDED"],                       # 24 DURATION
    ["FRIENDLY","NEUTRAL","HOSTILE","UNKNOWN"],                 # 25 CONTACT
    ["FULL","HALF","LOW","CRITICAL"],                            # 26 FUEL
    ["LOW","MEDIUM","HIGH"],                                    # 27 ALTITUDE
    ["CLEAR","REDUCED","ZERO"],                                 # 28 VISIBILITY
    ["SILENT","QUIET","MODERATE","LOUD"],                       # 29 NOISE
    ["SINGLE","PAIR","SQUAD","PLATOON"],                        # 30 FORMATION
    ["NONE","LIGHT","MEDIUM","HEAVY"],                          # 31 ARMOR
    ["STRONG","WEAK","JAMMED","LOST"],                          # 32 SIGNAL
    ["HIGH","MEDIUM","LOW"],                                    # 33 MORALE
    ["ABUNDANT","ADEQUATE","SCARCE","DEPLETED"],                # 34 SUPPLY
    ["CONFIRMED","PROBABLE","UNCERTAIN","NONE"],                # 35 INTEL
    ["STANDING","PREPPED","LAUNCHED","ABORTED"],                # 36 EVAC
    ["SUNNY","OVERCAST","RAIN","SNOW"],                         # 37 WEATHER2
    ["OFFENSIVE","DEFENSIVE","RECON","SUPPORT"],                # 38 DOCTRINE
    ["SECURE","OPEN","COMPROMISED","SILENT"],                   # 39 COMMS
]

CHANNEL_NAMES = [
    "SECRET","LOCATION","AGENT","STATUS","PRIORITY","BACKUP","RULE","META",
    "TEAM","REGION","PHASE","COMM","ASSET","WINDOW","COVER","SUPPORT",
    "THREAT","WEATHER","TERRAIN","EXTRACT","CIPHER","FREQ","PAYLOAD","ROUTE",
    "DURATION","CONTACT","FUEL","ALTITUDE","VISIBILITY","NOISE","FORMATION",
    "ARMOR","SIGNAL","MORALE","SUPPLY","INTEL","EVAC","WEATHER2","DOCTRINE","COMMS",
]

K = 40
A_LOW, A_HIGH = 0.25, 1.0
L = 64  # quantization levels
N_SAMPLES = 256  # samples per encoder pass
FS = 100.0


# ============================================================
# FDM Signal Encoder (matches paper exactly)
# ============================================================

def generate_channel_values():
    """Random channel values for all 40 channels."""
    return [random.choice(vocab) for vocab in CHANNEL_VOCAB]


def encode_fdm_signal(channel_values: List[str], vocab_size: int,
                      seed: int = 42) -> List[int]:
    """
    Encode channel values into FDM token IDs.
    Returns W=512 token IDs (256 natural + 256 S-random interleaved).
    """
    # Convert string values to indices
    val_indices = []
    for k, val in enumerate(channel_values):
        val_indices.append(CHANNEL_VOCAB[k].index(val))

    # Build composite signal
    t = np.arange(N_SAMPLES) / FS
    signal = np.zeros(N_SAMPLES)
    for k, val_idx in enumerate(val_indices):
        freq = k + 1  # carrier at (k+1) Hz
        n_bits = math.ceil(math.log2(max(len(CHANNEL_VOCAB[k]), 2)))
        for bit_idx in range(n_bits):
            bit = (val_idx >> bit_idx) & 1
            amp = A_HIGH if bit else A_LOW
            signal += amp * np.sin(2 * math.pi * freq * t) / n_bits

    # Global normalization (fixed range, matches Qwen3/Hermes3 encoder)
    amp_range = K * A_HIGH
    signal_norm = (signal + amp_range) / (2 * amp_range)  # [0, 1]
    quantized = np.floor(signal_norm * (L - 1)).clip(0, L - 1).astype(int)

    # Pseudorandom token mapping (bijection)
    rng = np.random.default_rng(seed)
    token_map = rng.permutation(vocab_size)[:L].tolist()
    tokens_natural = [token_map[q] for q in quantized]

    # S-random interleaving
    S = int(math.floor(math.sqrt(N_SAMPLES / 2)))
    s_perm = _srandom_permutation(N_SAMPLES, S, seed=seed + 1)
    tokens_interleaved = [token_map[quantized[s_perm[i]]] for i in range(N_SAMPLES)]

    return tokens_natural + tokens_interleaved


def _srandom_permutation(n: int, s: int, seed: int = 43) -> List[int]:
    """Generate S-random permutation (greedy approximation)."""
    rng = np.random.RandomState(seed)
    available = list(range(n))
    rng.shuffle(available)
    perm = []
    used = set()

    for pos in range(n):
        placed = False
        for idx, candidate in enumerate(available):
            ok = True
            for prev_pos in range(max(0, pos - s), pos):
                if abs(perm[prev_pos] - candidate) <= s:
                    ok = False
                    break
            if ok:
                perm.append(candidate)
                available.pop(idx)
                placed = True
                break
        if not placed:
            perm.append(available.pop(0))

    return perm


# ============================================================
# Build prompts
# ============================================================

def build_prompt_with_fdm(channel_values: List[str], tokenizer,
                          vocab_size: int) -> Tuple[List[int], List[int], str]:
    """
    Build full prompt with FDM tokens in context.
    Returns: (full_input_ids, fdm_token_ids, question_text)
    """
    fdm_tokens = encode_fdm_signal(channel_values, vocab_size)

    question = "Report all context values for channels 8-39."
    answer_parts = [f"{CHANNEL_NAMES[k]}={channel_values[k]}" for k in range(8, 40)]
    expected_answer = "Context: " + ", ".join(answer_parts) + "."

    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    memory_end = tokenizer.encode("[/MEMORY]", add_special_tokens=False)
    question_ids = tokenizer.encode(
        f"\nQuestion: {question}\nAnswer:",
        add_special_tokens=False
    )

    bos = tokenizer.encode(tokenizer.bos_token or "", add_special_tokens=True)
    full_ids = bos + memory_start + fdm_tokens + memory_end + question_ids

    return full_ids, fdm_tokens, expected_answer


def build_prompt_without_fdm(channel_values: List[str], tokenizer) -> List[int]:
    """
    Build prompt WITHOUT FDM tokens — just [MEMORY][/MEMORY] empty.
    The FDM signal will be injected via KV cache instead.
    """
    question = "Report all context values for channels 8-39."
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    memory_end = tokenizer.encode("[/MEMORY]", add_special_tokens=False)
    question_ids = tokenizer.encode(
        f"\nQuestion: {question}\nAnswer:",
        add_special_tokens=False
    )
    bos = tokenizer.encode(tokenizer.bos_token or "", add_special_tokens=True)
    return bos + memory_start + memory_end + question_ids


# ============================================================
# Evaluation
# ============================================================

def score_extra_channels(generated_text: str, channel_values: List[str]) -> Dict:
    """Score per-channel accuracy for channels 8-39."""
    results = {}
    for k in range(8, 40):
        name = CHANNEL_NAMES[k]
        expected = channel_values[k]
        pattern = rf"\b{re.escape(name)}={re.escape(expected)}\b"
        results[name] = bool(re.search(pattern, generated_text))
    return results


# ============================================================
# Experiment 1: Baseline (FDM in context)
# ============================================================

def run_baseline(model, tokenizer, n_eval: int, device: str):
    """Normal FDM-in-context evaluation. Should match paper results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline (FDM in context tokens)")
    print("=" * 60)

    channel_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0

    for i in tqdm(range(n_eval), desc="Baseline"):
        cv = generate_channel_values()
        input_ids, _, expected = build_prompt_with_fdm(
            cv, tokenizer, tokenizer.vocab_size
        )
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            out = model.generate(
                input_tensor,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][len(input_ids):], skip_special_tokens=True)
        scores = score_extra_channels(generated, cv)
        for name, correct in scores.items():
            channel_correct[name] += int(correct)
        total += 1

    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    joint_correct = sum(1 for i in range(n_eval)
                        if all(per_ch.values()))  # approximate
    print(f"\n  Overall extra-channel accuracy: {overall*100:.1f}%")
    print(f"  Worst 5 channels:")
    for name in sorted(per_ch, key=per_ch.get)[:5]:
        print(f"    {name}: {per_ch[name]*100:.1f}%")

    return per_ch, overall


# ============================================================
# Experiment 2: Frozen KV Cache Injection
# ============================================================

def compute_fdm_kv_cache(model, tokenizer, channel_values: List[str],
                         device: str):
    """
    Compute the KV cache entries that the model would produce for
    FDM tokens, then return them as a frozen tensor.

    This is the bridge: we run the model on just the FDM tokens,
    capture the KV pairs from each layer, and return them.
    These can then be prepended to the KV cache of a prompt
    that doesn't contain FDM tokens.
    """
    fdm_tokens = encode_fdm_signal(channel_values, tokenizer.vocab_size)
    fdm_tensor = torch.tensor([fdm_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(fdm_tensor, use_cache=True)
        kv_cache = outputs.past_key_values  # tuple of (K, V) per layer

    # Detach and clone — these are now frozen parameters
    frozen_kv = tuple(
        (k.detach().clone(), v.detach().clone())
        for k, v in kv_cache
    )
    return frozen_kv


def run_frozen_injection(model, tokenizer, n_eval: int, device: str):
    """
    Inject frozen KV cache from FDM tokens, run inference on
    prompt WITHOUT FDM tokens in context. Tests whether the model
    reads identically from cached vs live embeddings.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Frozen KV Cache Injection")
    print("  FDM tokens removed from context, injected as KV cache")
    print("=" * 60)

    channel_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0

    for i in tqdm(range(n_eval), desc="Frozen injection"):
        cv = generate_channel_values()

        # Step 1: compute KV cache from FDM tokens
        frozen_kv = compute_fdm_kv_cache(model, tokenizer, cv, device)

        # Step 2: build prompt WITHOUT FDM tokens
        prompt_ids = build_prompt_without_fdm(cv, tokenizer)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        # Step 3: run model with prepended KV cache
        # The model sees the KV cache as if FDM tokens were processed
        # earlier — attention will attend to them during generation.
        #
        # We need to adjust position IDs: the prompt tokens should have
        # positions starting AFTER the FDM sequence length
        fdm_seq_len = frozen_kv[0][0].shape[2]  # (batch, heads, seq_len, dim)
        position_ids = torch.arange(
            fdm_seq_len, fdm_seq_len + len(prompt_ids),
            dtype=torch.long, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            out = model.generate(
                prompt_tensor,
                past_key_values=frozen_kv,
                position_ids=position_ids,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
        scores = score_extra_channels(generated, cv)
        for name, correct in scores.items():
            channel_correct[name] += int(correct)
        total += 1

    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    print(f"\n  Overall extra-channel accuracy: {overall*100:.1f}%")
    print(f"  Worst 5 channels:")
    for name in sorted(per_ch, key=per_ch.get)[:5]:
        print(f"    {name}: {per_ch[name]*100:.1f}%")

    return per_ch, overall


# ============================================================
# Experiment 3: Learned Memory Layer (the write head)
# ============================================================

class FDMMemoryLayer(nn.Module):
    """
    Learned write head: maps channel values → KV cache entries
    that match what the frozen model would produce from FDM tokens.

    This is the parametric write path. Once trained, it replaces
    both the external FDM encoder AND the context tokens — facts
    live entirely in this module's parameters + the channel values
    fed to it at inference time.

    Architecture:
      For each channel k with value v:
        - Embed (k, v) into a hidden vector
        - Project to (n_layers × 2 × n_heads × head_dim) to produce
          the KV contributions for all layers simultaneously
        - Sum contributions across channels (additive, like FDM superposition)
    """

    def __init__(self, model_config, n_channels: int = 40,
                 max_values: int = 5, seq_len: int = 512):
        super().__init__()
        self.n_channels = n_channels
        self.max_values = max_values
        self.seq_len = seq_len

        # Extract model dimensions
        self.n_layers = model_config.num_hidden_layers
        self.n_heads = getattr(model_config, 'num_key_value_heads',
                               model_config.num_attention_heads)
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads
        self.hidden_size = model_config.hidden_size

        # KV size per layer: 2 (K and V) × n_heads × seq_len × head_dim
        kv_total_dim = 2 * self.n_heads * self.seq_len * self.head_dim

        # Channel + value embedding
        embed_dim = 256
        self.channel_embed = nn.Embedding(n_channels, embed_dim)
        self.value_embed = nn.Embedding(max_values, embed_dim)

        # Project to KV contributions for ALL layers at once
        self.kv_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, self.n_layers * 2 * self.n_heads * self.head_dim),
            # Output: per-position contribution to K and V across all layers
            # Shape will be reshaped to (n_layers, 2, n_heads, head_dim)
        )

        # Position-dependent expansion: broadcast channel contribution
        # across seq_len positions with learned position weights
        self.pos_weights = nn.Parameter(
            torch.randn(self.seq_len, 1) * 0.01
        )

    def forward(self, channel_ids: torch.Tensor,
                value_ids: torch.Tensor) -> tuple:
        """
        channel_ids: (n_active_channels,) int
        value_ids: (n_active_channels,) int — index into channel vocab
        Returns: past_key_values format matching the model
        """
        device = channel_ids.device

        # Embed each (channel, value) pair
        ch_emb = self.channel_embed(channel_ids)    # (n_ch, embed_dim)
        val_emb = self.value_embed(value_ids)       # (n_ch, embed_dim)
        combined = torch.cat([ch_emb, val_emb], -1) # (n_ch, 2*embed_dim)

        # Project to KV space
        kv_raw = self.kv_proj(combined)  # (n_ch, n_layers*2*n_heads*head_dim)
        kv_raw = kv_raw.view(
            -1, self.n_layers, 2, self.n_heads, self.head_dim
        )  # (n_ch, n_layers, 2, n_heads, head_dim)

        # Sum across channels (additive superposition, like FDM)
        kv_summed = kv_raw.sum(dim=0)  # (n_layers, 2, n_heads, head_dim)

        # Expand across positions with learned weights
        # kv_summed: (n_layers, 2, n_heads, head_dim)
        # pos_weights: (seq_len, 1)
        pos_w = torch.sigmoid(self.pos_weights)  # (seq_len, 1)

        # Build full KV cache: (n_layers, 2, 1, n_heads, seq_len, head_dim)
        kv_expanded = kv_summed.unsqueeze(2).unsqueeze(4)  # add batch + seq dims
        kv_expanded = kv_expanded.expand(
            -1, -1, 1, -1, self.seq_len, -1
        ).clone()

        # Apply position-dependent modulation
        # pos_w: (seq_len, 1) → broadcast across heads and head_dim
        kv_expanded = kv_expanded * pos_w.view(1, 1, 1, 1, self.seq_len, 1)

        # Format as past_key_values: tuple of (K, V) per layer
        # Each K, V shape: (batch=1, n_heads, seq_len, head_dim)
        past_kv = tuple(
            (kv_expanded[layer, 0],   # K: (1, n_heads, seq_len, head_dim)
             kv_expanded[layer, 1])   # V: (1, n_heads, seq_len, head_dim)
            for layer in range(self.n_layers)
        )
        return past_kv


def train_memory_layer(model, tokenizer, memory_layer: FDMMemoryLayer,
                       n_steps: int, device: str, lr: float = 1e-4):
    """
    Train the memory layer to produce KV caches that match
    what the frozen model computes from actual FDM tokens.

    Loss: MSE between memory_layer's KV output and the model's
    KV output when processing actual FDM tokens.
    """
    print("\n" + "=" * 60)
    print(f"TRAINING: FDMMemoryLayer ({n_steps} steps)")
    print("  Target: match frozen model's KV cache from FDM tokens")
    print("=" * 60)

    optimizer = torch.optim.AdamW(memory_layer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr / 10
    )

    model.eval()
    memory_layer.train()

    import time
    t0 = time.time()

    for step in range(1, n_steps + 1):
        # Generate random channel values
        cv = generate_channel_values()

        # Get target KV cache from model processing actual FDM tokens
        target_kv = compute_fdm_kv_cache(model, tokenizer, cv, device)

        # Get channel/value indices for memory layer
        ch_ids = torch.arange(K, device=device)
        val_ids = torch.tensor(
            [CHANNEL_VOCAB[k].index(cv[k]) for k in range(K)],
            device=device
        )

        # Forward through memory layer
        pred_kv = memory_layer(ch_ids, val_ids)

        # MSE loss across all layers
        loss = 0.0
        for layer_idx in range(len(target_kv)):
            target_k, target_v = target_kv[layer_idx]
            pred_k, pred_v = pred_kv[layer_idx]
            loss += F.mse_loss(pred_k, target_k)
            loss += F.mse_loss(pred_v, target_v)
        loss = loss / (2 * len(target_kv))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(memory_layer.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == n_steps:
            elapsed = time.time() - t0
            print(f"  Step {step:5d}/{n_steps} | loss {loss.item():.6f} | {elapsed:.0f}s")

    return memory_layer


def run_learned_injection(model, tokenizer, memory_layer: FDMMemoryLayer,
                          n_eval: int, device: str):
    """
    Use the trained memory layer to produce KV cache,
    then run inference without FDM tokens in context.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Learned Memory Layer Injection")
    print("  FDM encoder + context tokens replaced by learned parameters")
    print("=" * 60)

    memory_layer.eval()
    channel_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0

    for i in tqdm(range(n_eval), desc="Learned injection"):
        cv = generate_channel_values()

        # Get KV cache from memory layer
        ch_ids = torch.arange(K, device=device)
        val_ids = torch.tensor(
            [CHANNEL_VOCAB[k].index(cv[k]) for k in range(K)],
            device=device
        )
        with torch.no_grad():
            learned_kv = memory_layer(ch_ids, val_ids)

        # Build prompt without FDM tokens
        prompt_ids = build_prompt_without_fdm(cv, tokenizer)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        fdm_seq_len = learned_kv[0][0].shape[2]
        position_ids = torch.arange(
            fdm_seq_len, fdm_seq_len + len(prompt_ids),
            dtype=torch.long, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            out = model.generate(
                prompt_tensor,
                past_key_values=learned_kv,
                position_ids=position_ids,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
        scores = score_extra_channels(generated, cv)
        for name, correct in scores.items():
            channel_correct[name] += int(correct)
        total += 1

    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    print(f"\n  Overall extra-channel accuracy: {overall*100:.1f}%")
    print(f"  Worst 5 channels:")
    for name in sorted(per_ch, key=per_ch.get)[:5]:
        print(f"    {name}: {per_ch[name]*100:.1f}%")

    return per_ch, overall


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FDM-in-Weights via KV Cache Injection"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model path (e.g. prompterminal/qwen3-0.6b-fdm-v3)")
    parser.add_argument("--arch", type=str, required=True,
                        choices=["gpt2", "qwen3", "lfm25", "hermes3"],
                        help="Architecture tag (for encoder seed/vocab matching)")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Number of evaluation samples per experiment")
    parser.add_argument("--n_train_steps", type=int, default=2000,
                        help="Training steps for learned memory layer")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline evaluation")
    parser.add_argument("--skip_learned", action="store_true",
                        help="Skip learned memory layer training")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("FDM-in-Weights via KV Cache Injection")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Arch:   {args.arch}")
    print(f"Device: {device}")
    print(f"N eval: {args.n_eval}")

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print(f"  Loaded. Vocab size: {tokenizer.vocab_size}")
    print(f"  Config: {model.config.num_hidden_layers} layers, "
          f"{model.config.hidden_size} hidden, "
          f"{model.config.num_attention_heads} heads")

    results = {}

    # ---- Experiment 1: Baseline ----
    if not args.skip_baseline:
        per_ch, overall = run_baseline(model, tokenizer, args.n_eval, device)
        results["baseline"] = {"overall": overall, "per_channel": per_ch}

    # ---- Experiment 2: Frozen injection ----
    per_ch, overall = run_frozen_injection(model, tokenizer, args.n_eval, device)
    results["frozen_injection"] = {"overall": overall, "per_channel": per_ch}

    # ---- Experiment 3: Learned memory layer ----
    if not args.skip_learned:
        memory_layer = FDMMemoryLayer(
            model.config,
            n_channels=K,
            max_values=max(len(v) for v in CHANNEL_VOCAB),
            seq_len=512,
        ).to(device)

        memory_layer = train_memory_layer(
            model, tokenizer, memory_layer,
            n_steps=args.n_train_steps,
            device=device,
        )

        per_ch, overall = run_learned_injection(
            model, tokenizer, memory_layer, args.n_eval, device
        )
        results["learned_injection"] = {"overall": overall, "per_channel": per_ch}

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:25s}: {r['overall']*100:.1f}%")

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print("""
  Baseline:          Should match paper results (~98-100% for Qwen3/Hermes3).
                     Confirms the model reads FDM from context tokens.

  Frozen injection:  If this matches baseline, the model's read mechanism
                     operates on KV representations, not raw token IDs.
                     The FDM signal can live in cached KV pairs (= parameters)
                     instead of context tokens with zero accuracy loss.

  Learned injection: If this approaches baseline after training, we have
                     a working WRITE HEAD — a parametric module that can
                     produce the right KV entries from channel values alone,
                     replacing both the external FDM encoder AND context tokens.
                     Facts now live entirely in weights.

  If frozen matches but learned doesn't: the write head needs more capacity
  or training. The read path works from weights; the write path needs work.

  If frozen degrades: the model's read mechanism depends on something
  beyond KV cache content (e.g., raw embedding values, layernorm state).
  This would mean a different injection point is needed.
    """)

    # Save results
    out_path = f"fdm_kv_results_{args.arch}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
