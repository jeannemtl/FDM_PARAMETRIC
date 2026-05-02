#!/usr/bin/env python3
"""
Qwen3 DSB-SC variant of the FDM carrier attention head.

Mirrors fdm_carrier_head_dsb_sc_hermes3.py exactly, but imports the Qwen3
ASK module (which uses vocab_size=151936) and defaults to the Qwen3
two-block model.

Completes the 2x2 cross-architecture matrix:
    Qwen3 ASK:    98.2% slot, 57.8% joint  (working)
    Qwen3 DSB-SC: ?                         (this run)
    Hermes3 ASK:  failed (loss 0.99 plateau)
    Hermes3 DSB-SC: in flight

Usage:
    cd /workspace/FDM_IN_WEIGHTS
    python fdm_carrier_head_dsb_sc.py \\
        --n_carrier_channels 16 \\
        --n_steps 5000 \\
        --lr 3e-4 \\
        --n_eval 100 \\
        --seed 42 \\
        --output_dir carrier_head_dsb_sc_qwen3_16_16
"""

import os
import sys
import json
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

WORK_DIR = "/workspace/FDM_IN_WEIGHTS"
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, "/root/FDM_IN_WEIGHTS")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the QWEN3 ASK module - it has make_encoder using vocab_size=151936
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "fdm_ch_ask_qwen3",
    os.path.join(WORK_DIR, "fdm_carrier_head.py"),
)
fdm_ch_ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fdm_ch_ask)


class DSBSCCarrierAttentionHead(nn.Module):
    """
    DSB-SC carrier head matching Qwen3 ASK CarrierAttentionHead's I/O exactly.
    Identical to the Hermes3 DSB-SC class - the only architecture difference
    is the host model and its tokenizer/vocab.

    Forward:
        channel_values: list of (carrier_idx, value_idx) tuples
        returns: dict {layer_idx: (K_tensor, V_tensor)}
                 each tensor shape: (1, n_kv_heads, n_kv_positions, kv_dim)
    """

    def __init__(
        self,
        n_layers,
        kv_dim,
        n_kv_heads,
        n_carriers,
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
        self.n_values = n_values

        K_cos, K_sin = self._build_paired_bases(
            n_kv_positions, n_layers, n_kv_heads, kv_dim, n_carriers
        )
        self.register_buffer("K_cos", K_cos)
        self.register_buffer("K_sin", K_sin)

        amp_dim = n_layers * n_kv_heads * kv_dim
        self.value_I = nn.Embedding(n_carriers * n_values, amp_dim)
        self.value_Q = nn.Embedding(n_carriers * n_values, amp_dim)
        nn.init.normal_(self.value_I.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value_Q.weight, mean=0.0, std=0.02)

    def _build_paired_bases(
        self, n_pos, n_layers, n_kv_heads, kv_dim, n_carriers
    ):
        t = torch.arange(n_pos, dtype=torch.float32) / self.sample_rate
        freqs = torch.arange(1, n_carriers + 1, dtype=torch.float32)
        cos_carriers = torch.cos(2 * math.pi * freqs[None, :] * t[:, None])
        sin_carriers = torch.sin(2 * math.pi * freqs[None, :] * t[:, None])

        dim_per_carrier = max(1, kv_dim // n_carriers)

        K_cos = torch.zeros(n_pos, n_kv_heads, kv_dim, dtype=torch.float32)
        K_sin = torch.zeros(n_pos, n_kv_heads, kv_dim, dtype=torch.float32)
        for c in range(n_carriers):
            start_dim = (c * dim_per_carrier) % kv_dim
            end_dim = min(start_dim + dim_per_carrier, kv_dim)
            K_cos[:, :, start_dim:end_dim] = (
                cos_carriers[:, c:c+1, None]
                .expand(-1, n_kv_heads, end_dim - start_dim)
            )
            K_sin[:, :, start_dim:end_dim] = (
                sin_carriers[:, c:c+1, None]
                .expand(-1, n_kv_heads, end_dim - start_dim)
            )

        K_cos = K_cos.unsqueeze(0).expand(n_layers, -1, -1, -1).clone()
        K_sin = K_sin.unsqueeze(0).expand(n_layers, -1, -1, -1).clone()
        return K_cos, K_sin

    def forward(self, channel_values):
        device = self.K_cos.device
        n_layers = self.n_layers
        n_pos = self.n_kv_positions
        n_heads = self.n_kv_heads
        kv_dim = self.kv_dim
        n_carriers = self.n_carriers
        dim_per_carrier = max(1, kv_dim // n_carriers)

        # K cache: K_sin (matches ASK convention)
        K = self.K_sin.unsqueeze(0).permute(0, 1, 3, 2, 4).contiguous()

        V_total = torch.zeros(1, n_layers, n_heads, n_pos, kv_dim, device=device)

        for c, (carrier_idx, value_idx) in enumerate(channel_values):
            flat_idx = carrier_idx * self.n_values + value_idx
            idx_t = torch.tensor([flat_idx], device=device)
            I_flat = self.value_I(idx_t)
            Q_flat = self.value_Q(idx_t)
            I_amp = I_flat.view(1, n_layers, n_heads, kv_dim)
            Q_amp = Q_flat.view(1, n_layers, n_heads, kv_dim)

            start_dim = (c * dim_per_carrier) % kv_dim
            end_dim = min(start_dim + dim_per_carrier, kv_dim)

            cos_pattern = self.K_cos[0, :, 0, start_dim]
            sin_pattern = self.K_sin[0, :, 0, start_dim]

            I_slice = I_amp[..., start_dim:end_dim]
            Q_slice = Q_amp[..., start_dim:end_dim]

            cos_b = cos_pattern.view(1, 1, 1, n_pos, 1)
            sin_b = sin_pattern.view(1, 1, 1, n_pos, 1)
            V_total[..., start_dim:end_dim] += (
                I_slice.unsqueeze(3) * cos_b
                + Q_slice.unsqueeze(3) * sin_b
            )

        layer_kvs = {}
        for l in range(n_layers):
            k_l = K[0, l].unsqueeze(0)
            v_l = V_total[0, l].unsqueeze(0)
            layer_kvs[l] = (k_l, v_l)
        return layer_kvs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/workspace/FDM_IN_WEIGHTS/two_block_model",
    )
    parser.add_argument("--n_carrier_channels", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/workspace/FDM_IN_WEIGHTS/carrier_head_dsb_sc_qwen3_16_16",
    )
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
    print("  QWEN3 DSB-SC CARRIER HEAD EXPERIMENT")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Carrier channels: {args.n_carrier_channels}")
    print(f"  Context channels: {32 - args.n_carrier_channels}")
    print(f"  Modulation: DSB-SC (paired cos/sin basis with I/Q amplitudes)")
    print(f"  Compare: Qwen3 ASK lr=3e-4 reached loss 0.016 by step 1000")
    print(f"  Compare: Qwen3 ASK final 98.2% slot, 57.8% joint at n=400")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    encoder = fdm_ch_ask.make_encoder(tokenizer)
    print(f"  Encoder ready (Qwen3 vocab_size=151936)")

    n_layers, n_kv_heads, head_dim = fdm_ch_ask.detect_kv_shape(
        model, model.config, device
    )
    print(f"  Detected: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    carrier_channels = list(range(8, 8 + args.n_carrier_channels))
    context_channels = list(range(8 + args.n_carrier_channels, 40))

    carrier_head = DSBSCCarrierAttentionHead(
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
    n_buffer = sum(b.numel() for b in carrier_head.buffers())
    print(f"  DSB-SC carrier head: {(n_total + n_buffer)/1e6:.2f}M total "
          f"({n_train/1e6:.2f}M trainable, "
          f"{n_buffer/1e6:.2f}M fixed cos+sin basis)")

    print("\n  Initial eval (untrained V amplitudes):")
    init_result = fdm_ch_ask.evaluate_carrier_head(
        model, tokenizer, carrier_head, encoder, device,
        carrier_channels, context_channels, n_eval=20
    )

    losses = fdm_ch_ask.train_carrier_head(
        model, tokenizer, carrier_head, encoder, device,
        carrier_channels, context_channels,
        n_steps=args.n_steps, lr=args.lr
    )

    print(f"\n  Final evaluation (n={args.n_eval}):")
    final_result = fdm_ch_ask.evaluate_carrier_head(
        model, tokenizer, carrier_head, encoder, device,
        carrier_channels, context_channels, n_eval=args.n_eval
    )
    final_result["initial"] = init_result
    final_result["modulation"] = "DSB-SC"
    final_result["host"] = "Qwen3-0.6B (two_block_model)"
    final_result["config"] = {
        "n_steps": args.n_steps, "lr": args.lr, "seed": args.seed,
        "n_carrier_channels": args.n_carrier_channels,
        "model": args.model,
        "trainable_params": n_train,
        "buffer_params": n_buffer,
    }

    out_path = os.path.join(args.output_dir, "carrier_head_dsb_sc_qwen3_results.json")
    with open(out_path, "w") as f:
        json.dump(final_result, f, indent=2)
    torch.save(
        carrier_head.state_dict(),
        os.path.join(args.output_dir, "carrier_head_dsb_sc_qwen3.pt")
    )
    print(f"\n  Saved to {args.output_dir}")
    print(f"\n  ==> 2x2 cross-architecture x modulation matrix:")
    print(f"      Qwen3 ASK:    car_slot=98.2%, joint=57.8%")
    print(f"      Qwen3 DSB-SC: car_slot={final_result['carrier_slot']*100:.1f}%, "
          f"joint={final_result['all_joint']*100:.1f}%")
    print(f"      Hermes3 ASK:  failed (loss 0.99 plateau)")
    print(f"      Hermes3 DSB-SC: see other run")


if __name__ == "__main__":
    main()
