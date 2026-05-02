#!/usr/bin/env python3
"""
Hermes3 DSB-SC carrier head WITH CURRICULUM TRAINING.

Tests whether channel-count curriculum (4 -> 8 -> 16) lifts the Hermes3
DSB-SC carrier head out of the local minimum it found at single-stage
5K steps (loss 0.47, slot 32%, joint 0%).

Curriculum design (mirrors Provisional B's write head curriculum,
adapted for 16/16 carrier head partition):

    Stage 0: 4 carriers (channels 8-11), 28 context (12-39)
        1500 steps at peak lr=3e-4, cosine annealing
    Stage 1: 8 carriers (channels 8-15), 24 context (16-39)
        1500 steps at peak lr=2e-4
    Stage 2: 16 carriers (channels 8-23), 16 context (24-39)
        2000 steps at peak lr=1e-4

Total: 5000 steps. Matches single-stage compute budget.

The carrier head module has 16 carriers structurally throughout;
unused carrier slots receive zero gradient when not included in
channel_values lists. K basis stays unchanged (fixed Fourier basis
at frequencies 1-16 Hz mapped to dim slices 0-7, 8-15, ..., 120-127).

Comparisons after training:
    Hermes3 ASK 5K single-stage:       failed (loss 0.99 plateau)
    Hermes3 DSB-SC 5K single-stage:    32.1% slot, 0% joint
    Hermes3 DSB-SC 5K curriculum:      this run

Usage:
    cd /workspace/FDM_IN_WEIGHTS
    python fdm_carrier_head_dsb_sc_hermes3_curriculum.py \\
        --n_eval 100 \\
        --seed 42 \\
        --output_dir carrier_head_dsb_sc_hermes3_curriculum_16_16
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

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "fdm_ch_ask_hermes",
    os.path.join(WORK_DIR, "fdm_carrier_head_hermes3.py"),
)
fdm_ch_ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fdm_ch_ask)


class DSBSCCarrierAttentionHead(nn.Module):
    """
    Identical to fdm_carrier_head_dsb_sc_hermes3.DSBSCCarrierAttentionHead.

    DSB-SC carrier head with paired cos/sin K bases and I/Q amplitude
    embeddings. Forward signature matches Hermes3 ASK head exactly.
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


# ====================================================================
# Curriculum schedule
# ====================================================================

# Each stage specifies (n_carrier_channels, n_steps, peak_lr, stage_name)
CURRICULUM_STAGES = [
    {
        "name": "Stage 0: 4 carriers (warmup)",
        "n_carrier_channels": 4,
        "n_steps": 1500,
        "lr": 3e-4,
    },
    {
        "name": "Stage 1: 8 carriers (expand)",
        "n_carrier_channels": 8,
        "n_steps": 1500,
        "lr": 2e-4,
    },
    {
        "name": "Stage 2: 16 carriers (full)",
        "n_carrier_channels": 16,
        "n_steps": 2000,
        "lr": 1e-4,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/workspace/FDM_IN_WEIGHTS/two_block_model_hermes3",
    )
    parser.add_argument("--n_max_carriers", type=int, default=16,
                        help="Final number of carriers in stage 2")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default="/workspace/FDM_IN_WEIGHTS/carrier_head_dsb_sc_hermes3_curriculum_16_16",
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
    print("  HERMES3 DSB-SC CARRIER HEAD WITH CURRICULUM TRAINING")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Max carriers: {args.n_max_carriers}")
    print(f"  Modulation: DSB-SC (paired cos/sin basis with I/Q amplitudes)")
    print(f"  Curriculum: {[s['n_carrier_channels'] for s in CURRICULUM_STAGES]} channels")
    print(f"  Steps per stage: {[s['n_steps'] for s in CURRICULUM_STAGES]}")
    print(f"  Peak LRs:        {[s['lr'] for s in CURRICULUM_STAGES]}")
    print(f"  Total steps: {sum(s['n_steps'] for s in CURRICULUM_STAGES)}")
    print()
    print(f"  Comparison baselines:")
    print(f"    Hermes3 ASK 5K single-stage:    failed (loss 0.99)")
    print(f"    Hermes3 DSB-SC 5K single-stage: 32.1% slot, 0% joint")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    encoder = fdm_ch_ask.make_encoder(tokenizer)
    print(f"  Encoder ready (Hermes3 vocab_size=128258)")

    n_layers, n_kv_heads, head_dim = fdm_ch_ask.detect_kv_shape(
        model, model.config, device
    )
    print(f"  Detected: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    # Build carrier head sized for the maximum carrier count.
    # All 16 carrier slots exist throughout training; only some receive
    # gradient signal in early stages.
    carrier_head = DSBSCCarrierAttentionHead(
        n_layers=n_layers,
        kv_dim=head_dim,
        n_kv_heads=n_kv_heads,
        n_carriers=args.n_max_carriers,
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

    # Initial eval before any training
    print("\n  Initial eval (untrained V amplitudes, 16/16 partition):")
    init_carrier_channels = list(range(8, 8 + args.n_max_carriers))
    init_context_channels = list(range(8 + args.n_max_carriers, 40))
    init_result = fdm_ch_ask.evaluate_carrier_head(
        model, tokenizer, carrier_head, encoder, device,
        init_carrier_channels, init_context_channels, n_eval=20
    )

    # Run each curriculum stage
    stage_results = []
    all_losses = []

    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        n_ch = stage["n_carrier_channels"]
        n_steps = stage["n_steps"]
        lr = stage["lr"]

        print(f"\n{'=' * 60}")
        print(f"  CURRICULUM STAGE {stage_idx}: {stage['name']}")
        print(f"{'=' * 60}")
        print(f"    Carrier channels: {n_ch} (channels 8 through {8+n_ch-1})")
        print(f"    Context channels: {32-n_ch} (channels {8+n_ch} through 39)")
        print(f"    Steps: {n_steps}")
        print(f"    Peak LR: {lr}")

        carrier_channels = list(range(8, 8 + n_ch))
        context_channels = list(range(8 + n_ch, 40))

        # Train for this stage
        # Note: train_carrier_head creates its own optimizer per call.
        # That's actually what we want for curriculum - each stage gets
        # a fresh AdamW with its own cosine annealing schedule.
        stage_losses = fdm_ch_ask.train_carrier_head(
            model, tokenizer, carrier_head, encoder, device,
            carrier_channels, context_channels,
            n_steps=n_steps, lr=lr
        )
        all_losses.extend(stage_losses)

        # Mid-stage eval at full 16/16 partition (regardless of current
        # stage carrier count) to track full-task progress
        print(f"\n  Mid-curriculum eval at 16/16 partition (n=20):")
        mid_result = fdm_ch_ask.evaluate_carrier_head(
            model, tokenizer, carrier_head, encoder, device,
            init_carrier_channels, init_context_channels, n_eval=20
        )

        stage_results.append({
            "stage": stage_idx,
            "n_carrier_channels_active": n_ch,
            "n_steps": n_steps,
            "lr": lr,
            "final_loss_avg_last_100": (
                sum(stage_losses[-100:]) / max(len(stage_losses[-100:]), 1)
                if stage_losses else None
            ),
            "mid_eval_at_full_16_16": mid_result,
        })

    # Final evaluation at full 16/16 partition with full n_eval
    print(f"\n{'=' * 60}")
    print(f"  FINAL EVALUATION at 16/16 partition (n={args.n_eval})")
    print(f"{'=' * 60}")
    final_result = fdm_ch_ask.evaluate_carrier_head(
        model, tokenizer, carrier_head, encoder, device,
        init_carrier_channels, init_context_channels, n_eval=args.n_eval
    )
    final_result["initial"] = init_result
    final_result["modulation"] = "DSB-SC"
    final_result["host"] = "Hermes3-3B (two_block_model_hermes3)"
    final_result["training"] = "curriculum"
    final_result["curriculum"] = {
        "stages": CURRICULUM_STAGES,
        "total_steps": sum(s["n_steps"] for s in CURRICULUM_STAGES),
        "stage_results": stage_results,
    }
    final_result["config"] = {
        "n_max_carriers": args.n_max_carriers,
        "seed": args.seed,
        "model": args.model,
        "trainable_params": n_train,
        "buffer_params": n_buffer,
    }

    out_path = os.path.join(args.output_dir, "carrier_head_dsb_sc_hermes3_curriculum_results.json")
    with open(out_path, "w") as f:
        json.dump(final_result, f, indent=2)
    torch.save(
        carrier_head.state_dict(),
        os.path.join(args.output_dir, "carrier_head_dsb_sc_hermes3_curriculum.pt")
    )
    print(f"\n  Saved to {args.output_dir}")

    print(f"\n  ==> Curriculum vs single-stage comparison (Hermes3 DSB-SC):")
    print(f"      Single-stage 5K:  car_slot=32.1%, ctx_slot=25.5%, joint=0.0%")
    print(f"      Curriculum 5K:    car_slot={final_result['carrier_slot']*100:.1f}%, "
          f"ctx_slot={final_result['context_slot']*100:.1f}%, "
          f"joint={final_result['all_joint']*100:.1f}%")
    print(f"\n  ==> Stage-by-stage 16/16 mid-eval progress:")
    for sr in stage_results:
        mid = sr["mid_eval_at_full_16_16"]
        print(f"      Stage {sr['stage']} (n_ch={sr['n_carrier_channels_active']}): "
              f"car_slot={mid['carrier_slot']*100:.1f}%, "
              f"joint={mid['all_joint']*100:.1f}%, "
              f"final_loss={sr['final_loss_avg_last_100']:.4f}")


if __name__ == "__main__":
    main()
