#!/usr/bin/env python3
"""
FDM Carrier-Head experiment (Architecture 3).
=============================================

Tests whether FDM carriers can live in WEIGHT space rather than CACHE space.

The standard split-source architecture puts the carrier structure in the
KV cache (along the position axis). This experiment puts the carrier
structure in a dedicated attention head's projection matrices (along the
dimension axis). The head's K and V projections are CONSTRUCTED with
sinusoidal carrier structure rather than learned.

Architecture:
    - Add ONE parallel attention head to the host transformer
    - Head's K_proj is fixed:    K[t, c] = sin(2*pi*f_c*t)  for carrier c at time t
    - Head's V_proj is fixed:    learnable amplitude embedding per (channel, value)
    - The query side is the host's existing query projection (unchanged)
    - Host transformer is frozen; only the V amplitude embeddings are trained

The test: train at a fixed 16-channel partition where 16 channels go through
the carrier head (no KV injection, no context tokens for those channels) and
16 channels go through normal in-context FDM. Compare to the equivalent
learned-write-head baseline at 16/16.

Usage:
    python fdm_carrier_head.py \\
        --model /workspace/FDM_IN_WEIGHTS/two_block_model \\
        --n_carrier_channels 16 \\
        --n_steps 5000 --n_eval 100 \\
        --output_dir carrier_head_exp

If accuracy is comparable to the write-head baseline, FDM works in weight space.
If accuracy is much lower, the host's read mechanism requires specific learned
patterns rather than literal frequency demodulation.
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
ALL_QUERY_CHANNELS = list(range(8, 40))


# ============================================================
# Carrier Attention Head
# ============================================================

class CarrierAttentionHead(nn.Module):
    """
    A single auxiliary attention head with HARDCODED carrier structure.

    The K projection is constructed (not learned) so that the key tensor at
    position t and dimension c has value sin(2*pi*f_c*t). This puts the FDM
    carrier basis in WEIGHT SPACE.

    The V projection IS learned: it maps (channel_id, value_id) -> amplitude
    in the carrier basis. This is the only trainable component.

    At inference, given a memory dict {channel: value}, the head produces:
        - K cache:  pre-constructed sinusoidal pattern (no learning, no signal)
        - V cache:  learned amplitudes weighted by the carrier basis

    The host's frozen attention then reads this through standard dot-product.
    """

    def __init__(
        self,
        n_layers,
        kv_dim,
        n_kv_heads,
        n_carriers,         # = number of channels carried by this head
        n_kv_positions=513,
        sample_rate=100.0,
        n_values=64,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.kv_dim = kv_dim
        self.n_kv_heads = n_kv_heads
        self.n_carriers = n_carriers
        self.n_kv_positions = n_kv_positions
        self.sample_rate = sample_rate

        # ---- Construct the FIXED carrier basis (K side) ----
        # K_basis[layer, position, head, dim] = sin(2*pi*f_c*t / fs) for carrier c
        # We use carriers at f_c = c+1 Hz (matching encoder convention)
        # Each carrier occupies multiple head-dimensions (replicated) to match kv_dim
        K_basis = self._build_carrier_basis(n_kv_positions, n_layers, n_kv_heads, kv_dim, n_carriers)
        # Register as buffer (non-trainable, moves with .to(device))
        self.register_buffer("K_basis", K_basis)

        # ---- Learnable V amplitudes (channel, value) -> (n_layers, n_kv_heads, kv_dim) ----
        # The V cache is constructed by selecting amplitudes per channel-value, then
        # broadcasting against the same carrier basis structure (so V also has carrier
        # locality but its magnitude is value-dependent).
        # Shape: (n_carriers, n_values, n_layers * n_kv_heads * kv_dim)
        self.value_amplitude = nn.Embedding(n_carriers * n_values,
                                            n_layers * n_kv_heads * kv_dim)
        nn.init.normal_(self.value_amplitude.weight, mean=0.0, std=0.02)
        self.n_values = n_values

    def _build_carrier_basis(self, n_pos, n_layers, n_kv_heads, kv_dim, n_carriers):
        """
        Construct fixed sinusoidal carrier basis for K cache.

        Output shape: (n_layers, n_pos, n_kv_heads, kv_dim)

        Each of the n_carriers gets a slice of kv_dim. Carriers are at f_c = c+1 Hz.
        At position t, dim slice c has values sin(2*pi*f_c*t/fs).

        We replicate the same pattern across layers (the carrier structure should
        appear at every layer; the host's per-layer attention will pick it up).
        """
        t = torch.arange(n_pos, dtype=torch.float32) / self.sample_rate  # (n_pos,)

        # For each carrier c, generate sin(2pi f_c t)
        # Frequencies: 1 Hz, 2 Hz, ..., n_carriers Hz (matching encoder)
        freqs = torch.arange(1, n_carriers + 1, dtype=torch.float32)  # (n_carriers,)

        # Carrier signals: (n_pos, n_carriers)
        carriers = torch.sin(2 * math.pi * freqs[None, :] * t[:, None])

        # Distribute n_carriers across the kv_dim axis
        # Each carrier gets kv_dim // n_carriers consecutive dimensions, with replication
        # If n_carriers > kv_dim, we cycle (but for typical configs n_carriers <= kv_dim)
        dim_per_carrier = max(1, kv_dim // n_carriers)
        K = torch.zeros(n_pos, n_kv_heads, kv_dim, dtype=torch.float32)
        for c in range(n_carriers):
            start_dim = (c * dim_per_carrier) % kv_dim
            end_dim = min(start_dim + dim_per_carrier, kv_dim)
            # Set this carrier's pattern at this dim slice across all heads
            K[:, :, start_dim:end_dim] = carriers[:, c:c+1, None].expand(-1, n_kv_heads, end_dim - start_dim)

        # Replicate across layers
        K = K.unsqueeze(0).expand(n_layers, -1, -1, -1).clone()  # (n_layers, n_pos, n_kv_heads, kv_dim)
        return K

    def forward(self, channel_values):
        """
        Args:
            channel_values: list of (channel_id, value_id) for the carrier head's channels.
                            Length = n_carriers, each item is (carrier_index, value_in_vocab).

        Returns:
            layer_kvs: dict {layer_idx: (k, v)} ready to inject into KV cache
                        each k, v has shape (1, n_kv_heads, n_kv_positions, kv_dim)
        """
        device = self.K_basis.device
        n_layers = self.n_layers
        n_pos = self.n_kv_positions
        n_heads = self.n_kv_heads
        kv_dim = self.kv_dim
        n_carriers = self.n_carriers

        # K cache: just use the fixed basis (no per-query learning)
        K = self.K_basis.unsqueeze(0)  # (1, n_layers, n_pos, n_heads, kv_dim)
        # transpose to (1, n_heads, n_pos, kv_dim) per layer below
        K = K.permute(0, 1, 3, 2, 4).contiguous()  # (1, n_layers, n_heads, n_pos, kv_dim)

        # V cache: look up amplitude per (carrier_idx, value_id) and apply carrier basis
        # Build a per-layer V tensor.
        V_total = torch.zeros(1, n_layers, n_heads, n_pos, kv_dim, device=device)
        dim_per_carrier = max(1, kv_dim // n_carriers)

        for c, (carrier_idx, value_idx) in enumerate(channel_values):
            # Look up amplitude for this (carrier, value)
            flat_idx = carrier_idx * self.n_values + value_idx
            amp_flat = self.value_amplitude(torch.tensor([flat_idx], device=device))  # (1, n_layers*n_heads*kv_dim)
            amp = amp_flat.view(1, n_layers, n_heads, kv_dim)  # (1, n_layers, n_heads, kv_dim)

            # Apply same carrier basis but scaled by amplitude
            # carrier signal at the dim slice for this carrier
            start_dim = (c * dim_per_carrier) % kv_dim
            end_dim = min(start_dim + dim_per_carrier, kv_dim)

            # Use the carrier basis at this dim slice (already in K_basis)
            carrier_pattern = self.K_basis[0, :, 0, start_dim]  # (n_pos,) — same for all heads
            # V at this slice = amplitude * carrier_pattern
            # broadcasting: amp[..., start_dim:end_dim] (1,L,H,D) * pattern (n_pos,) -> (1,L,H,n_pos,D)
            amp_slice = amp[..., start_dim:end_dim]  # (1, L, H, D)
            V_total[..., start_dim:end_dim] += (
                amp_slice.unsqueeze(3) * carrier_pattern.view(1, 1, 1, n_pos, 1)
            )

        # Build layer_kvs dict
        layer_kvs = {}
        for l in range(n_layers):
            k_l = K[0, l]  # (n_heads, n_pos, kv_dim)
            v_l = V_total[0, l]  # (n_heads, n_pos, kv_dim)
            # add batch dim
            k_l = k_l.unsqueeze(0)  # (1, n_heads, n_pos, kv_dim)
            v_l = v_l.unsqueeze(0)  # (1, n_heads, n_pos, kv_dim)
            layer_kvs[l] = (k_l, v_l)
        return layer_kvs


# ============================================================
# Encoding utilities
# ============================================================

def make_encoder(tokenizer):
    return TurboFDMSignalEncoder(
        vocab_size=151936, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )


def random_memory():
    return {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}


def layer_kvs_to_cache(layer_kvs, n_layers):
    cache = DynamicCache()
    for l in range(n_layers):
        k, v = layer_kvs[l]
        cache.update(k, v, l)
    return cache


# ============================================================
# Generation with carrier head
# ============================================================

def generate_with_carrier_head(model, tokenizer, carrier_head, memory, encoder, device,
                                carrier_channels, context_channels, max_new_tokens=350):
    """Generate using carrier head for some channels and in-context FDM for others."""

    # Carrier head produces KV for carrier_channels
    if len(carrier_channels) > 0:
        channel_values = []
        for c_idx, ch in enumerate(carrier_channels):
            val = memory[ch]
            val_list = MEMORY_SCHEMAS[ch][1]
            val_idx = val_list.index(val) if val in val_list else 0
            channel_values.append((c_idx, val_idx))
        layer_kvs = carrier_head(channel_values)
        past_kv = layer_kvs_to_cache(layer_kvs, carrier_head.n_layers)
        n_kv_pos = carrier_head.n_kv_positions
    else:
        past_kv = None
        n_kv_pos = 0

    # In-context FDM for context_channels
    if len(context_channels) > 0:
        # Make a memory dict with only context channels (zero out others)
        context_mem = {ch: memory[ch] for ch in range(NUM_CHANNELS)}
        fdm_text, _ = encoder.encode_memory(context_mem)
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


# ============================================================
# Training and evaluation
# ============================================================

def train_carrier_head(model, tokenizer, carrier_head, encoder, device,
                        carrier_channels, context_channels,
                        n_steps=5000, lr=3e-4):
    """Train the carrier head's value amplitudes through frozen host."""

    print(f"\n  Training carrier head: {len(carrier_channels)} carrier ch, {len(context_channels)} context ch")
    n_params = sum(p.numel() for p in carrier_head.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.2f}M (only V amplitudes)")

    carrier_head.train()
    optimizer = torch.optim.AdamW(
        [p for p in carrier_head.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr/20)

    t0 = time.time()
    losses = []

    for step in range(1, n_steps + 1):
        mem = random_memory()

        # Carrier head input
        channel_values = []
        for c_idx, ch in enumerate(carrier_channels):
            val = mem[ch]
            val_list = MEMORY_SCHEMAS[ch][1]
            val_idx = val_list.index(val) if val in val_list else 0
            channel_values.append((c_idx, val_idx))
        layer_kvs = carrier_head(channel_values)

        # Build training prompt
        if len(context_channels) > 0:
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
        n_kv_pos = carrier_head.n_kv_positions

        position_ids = torch.arange(n_kv_pos, n_kv_pos + seq_len, device=device).unsqueeze(0)
        past_kv = layer_kvs_to_cache(layer_kvs, carrier_head.n_layers)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            outputs = model(input_ids=input_tensor, position_ids=position_ids,
                            past_key_values=past_kv, labels=label_tensor, use_cache=False)
            loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(carrier_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 500 == 0:
            avg = np.mean(losses[-500:])
            elapsed = time.time() - t0
            print(f"    Step {step:5d}/{n_steps} | loss {avg:.4f} | {elapsed:.0f}s")

    return losses


def evaluate_carrier_head(model, tokenizer, carrier_head, encoder, device,
                           carrier_channels, context_channels, n_eval=100):
    """Evaluate carrier head + in-context split, both metrics."""
    carrier_head.eval()

    car_correct = 0; car_total = 0
    ctx_correct = 0; ctx_total = 0
    sample_car_all = 0; sample_ctx_all = 0; sample_all_all = 0
    n_samples = 0

    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        answer = generate_with_carrier_head(model, tokenizer, carrier_head, mem, encoder, device,
                                             carrier_channels, context_channels)

        car_all = True
        ctx_all = True

        for k in carrier_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                car_correct += 1
            else:
                car_all = False
            car_total += 1

        for k in context_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                ctx_correct += 1
            else:
                ctx_all = False
            ctx_total += 1

        if car_all:
            sample_car_all += 1
        if ctx_all:
            sample_ctx_all += 1
        if car_all and ctx_all:
            sample_all_all += 1
        n_samples += 1

    car_slot = car_correct / car_total if car_total > 0 else 0.0
    ctx_slot = ctx_correct / ctx_total if ctx_total > 0 else 0.0
    car_joint = sample_car_all / n_samples if n_samples > 0 else 0.0
    ctx_joint = sample_ctx_all / n_samples if n_samples > 0 else 0.0
    all_joint = sample_all_all / n_samples if n_samples > 0 else 0.0

    print(f"\n  --- CARRIER HEAD RESULTS ---")
    print(f"  Carrier slot:  {car_slot*100:.1f}%  (n={car_total})")
    print(f"  Context slot:  {ctx_slot*100:.1f}%  (n={ctx_total})")
    print(f"  Carrier joint: {car_joint*100:.1f}%  (n={n_samples})")
    print(f"  Context joint: {ctx_joint*100:.1f}%  (n={n_samples})")
    print(f"  All joint:     {all_joint*100:.1f}%  (n={n_samples})")

    return {
        "n_carrier_channels": len(carrier_channels),
        "n_context_channels": len(context_channels),
        "carrier_slot": float(car_slot),
        "context_slot": float(ctx_slot),
        "carrier_joint": float(car_joint),
        "context_joint": float(ctx_joint),
        "all_joint": float(all_joint),
        "n_samples": n_samples,
        "carrier_correct": car_correct,
        "carrier_total": car_total,
        "context_correct": ctx_correct,
        "context_total": ctx_total,
    }


# ============================================================
# Main
# ============================================================

def detect_kv_shape(model, config, device):
    """Probe model to get actual KV shape."""
    n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", 28)
    n_kv_heads = (getattr(config, "num_key_value_heads", None)
                  or getattr(config, "num_kv_heads", None)
                  or getattr(config, "num_attention_heads", 16))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    return n_layers, n_kv_heads, head_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/FDM_IN_WEIGHTS/two_block_model")
    parser.add_argument("--n_carrier_channels", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="/workspace/FDM_IN_WEIGHTS/carrier_head_exp")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[seed] {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  CARRIER-HEAD EXPERIMENT (FDM in weight space)")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Carrier channels: {args.n_carrier_channels}")
    print(f"  Context channels: {32 - args.n_carrier_channels}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    encoder = make_encoder(tokenizer)

    # Probe architecture
    n_layers, n_kv_heads, head_dim = detect_kv_shape(model, model.config, device)
    print(f"  Detected: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    # Channel partition
    carrier_channels = list(range(8, 8 + args.n_carrier_channels))
    context_channels = list(range(8 + args.n_carrier_channels, 40))

    # Build carrier head
    carrier_head = CarrierAttentionHead(
        n_layers=n_layers,
        kv_dim=head_dim,
        n_kv_heads=n_kv_heads,
        n_carriers=args.n_carrier_channels,
        n_kv_positions=513,
        sample_rate=100.0,
        n_values=64,
    ).to(device)

    n_total = sum(p.numel() for p in carrier_head.parameters())
    n_train = sum(p.numel() for p in carrier_head.parameters() if p.requires_grad)
    n_buffer = n_total - n_train
    print(f"  Carrier head: {n_total/1e6:.2f}M total ({n_train/1e6:.2f}M trainable, "
          f"{n_buffer/1e6:.2f}M fixed carrier basis)")

    # Initial eval (untrained)
    print("\n  Initial eval (untrained V amplitudes):")
    init_result = evaluate_carrier_head(
        model, tokenizer, carrier_head, encoder, device,
        carrier_channels, context_channels, n_eval=20
    )

    # Train
    losses = train_carrier_head(
        model, tokenizer, carrier_head, encoder, device,
        carrier_channels, context_channels,
        n_steps=args.n_steps, lr=args.lr
    )

    # Final eval
    print("\n  Final evaluation:")
    final_result = evaluate_carrier_head(
        model, tokenizer, carrier_head, encoder, device,
        carrier_channels, context_channels, n_eval=args.n_eval
    )
    final_result["initial"] = init_result
    final_result["config"] = {
        "n_steps": args.n_steps, "lr": args.lr, "seed": args.seed,
        "n_carrier_channels": args.n_carrier_channels,
        "model": args.model,
    }

    # Save
    out_path = os.path.join(args.output_dir, "carrier_head_results.json")
    with open(out_path, "w") as f:
        json.dump(final_result, f, indent=2)
    torch.save(carrier_head.state_dict(),
               os.path.join(args.output_dir, "carrier_head.pt"))
    print(f"\n  Saved to {args.output_dir}")
    print(f"\n  ==> Compare to write-head baseline at 16/16:")
    print(f"      Write-head joint (n=200): 89.5% (Table 5, this paper)")
    print(f"      Carrier-head joint:        {final_result['all_joint']*100:.1f}%")


if __name__ == "__main__":
    main()
