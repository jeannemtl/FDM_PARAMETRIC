"""
FDM-in-Weights: Proof of Concept
=================================
Tests whether a neural network can:
1. WRITE facts into a frequency-multiplexed parameter buffer
   (a learned write head modulates carrier amplitudes)
2. READ facts back by decomposing the parameter buffer
   (implicit frequency decomposition, no FFT)

The key idea: instead of FDM signal living in context tokens,
it lives in a dedicated weight buffer. The model learns both
the write path (fact → amplitude update at correct carrier freq)
and the read path (buffer → frequency decomposition → fact).

We test:
- Single-channel write/read accuracy
- Multi-channel write/read (does orthogonality hold?)
- Sequential writes (does updating ch3 corrupt ch2?)
- Catastrophic forgetting resistance from carrier orthogonality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json
import math


# ============================================================
# Configuration
# ============================================================

@dataclass
class FDMWeightConfig:
    """Configuration for FDM-in-weights experiment."""
    n_channels: int = 40          # Number of FDM channels
    buffer_size: int = 256        # Samples in the parameter buffer
    fs: float = 100.0             # Sampling rate (Hz)
    n_values: int = 4             # Values per channel (2 bits)
    a_low: float = 0.25           # ASK low amplitude
    a_high: float = 1.0           # ASK high amplitude
    hidden_dim: int = 128         # Hidden dimension for write/read heads
    n_layers: int = 3             # Layers in write/read networks
    lr: float = 1e-3              # Learning rate
    train_steps: int = 10000      # Training steps
    batch_size: int = 64          # Batch size
    eval_samples: int = 1000      # Evaluation samples
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# FDM Signal Utilities (reference encoder for ground truth)
# ============================================================

def generate_fdm_signal(channel_values: torch.Tensor, config: FDMWeightConfig) -> torch.Tensor:
    """
    Generate ground-truth FDM signal from channel values.
    channel_values: (batch, n_channels) integer values in [0, n_values)
    Returns: (batch, buffer_size) continuous signal
    """
    batch_size = channel_values.shape[0]
    t = torch.linspace(0, config.buffer_size / config.fs, config.buffer_size,
                       device=channel_values.device)  # (buffer_size,)

    signal = torch.zeros(batch_size, config.buffer_size, device=channel_values.device)

    for k in range(config.n_channels):
        freq = k + 1  # Carrier at (k+1) Hz
        # Map discrete value to amplitude using binary ASK
        # For simplicity: value 0 → a_low, value > 0 → proportional
        # More precisely: encode each bit of the value
        n_bits = int(np.ceil(np.log2(max(config.n_values, 2))))
        for bit_idx in range(n_bits):
            bit_val = (channel_values[:, k] >> bit_idx) & 1  # (batch,)
            amp = torch.where(bit_val == 1,
                              torch.tensor(config.a_high, device=channel_values.device),
                              torch.tensor(config.a_low, device=channel_values.device))
            # Each bit could use a sub-carrier, but for simplicity
            # we sum bits into the carrier amplitude
            carrier = torch.sin(2 * math.pi * freq * t).unsqueeze(0)  # (1, buffer_size)
            signal += (amp.unsqueeze(1) * carrier) / n_bits

    return signal


# ============================================================
# FDM Memory Buffer (the "weights" that store facts)
# ============================================================

class FDMMemoryBuffer(nn.Module):
    """
    A learnable parameter buffer that stores frequency-multiplexed information.
    This replaces context tokens — facts live in these parameters.
    """

    def __init__(self, config: FDMWeightConfig):
        super().__init__()
        self.config = config
        # The core innovation: a parameter vector that holds the FDM signal
        # Shape: (buffer_size,) — analogous to the 256 token positions
        self.buffer = nn.Parameter(torch.zeros(config.buffer_size))
        self.reset_buffer()

    def reset_buffer(self):
        """Reset buffer to zero (no facts stored)."""
        with torch.no_grad():
            self.buffer.zero_()

    def get_signal(self) -> torch.Tensor:
        """Return the current buffer contents as a signal."""
        return self.buffer


# ============================================================
# Write Head: learns to encode a (channel_id, value) pair
# into an amplitude modulation at the correct carrier frequency
# ============================================================

class FDMWriteHead(nn.Module):
    """
    Given a channel index and a value, produces an update to the
    memory buffer that modulates the correct carrier frequency.

    Two approaches tested:
    1. Structured: directly compute sin(2π·f·t) · amplitude (analytical)
    2. Learned: neural network predicts the buffer update (learned)
    """

    def __init__(self, config: FDMWeightConfig, mode: str = "structured"):
        super().__init__()
        self.config = config
        self.mode = mode

        # Time axis (fixed)
        self.register_buffer(
            "t", torch.linspace(0, config.buffer_size / config.fs,
                                config.buffer_size)
        )

        if mode == "learned":
            # Learned write head: maps (channel_onehot, value_onehot) → buffer update
            input_dim = config.n_channels + config.n_values
            self.write_net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.buffer_size),
            )
        elif mode == "structured":
            # Structured write: learn only the amplitude mapping
            # (channel, value) → amplitude scalar
            self.amp_net = nn.Sequential(
                nn.Linear(config.n_channels + config.n_values, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
            )

    def forward(self, channel_idx: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Compute buffer update for writing (channel_idx, value).
        channel_idx: (batch,) integer channel indices
        value: (batch,) integer values
        Returns: (batch, buffer_size) update to add to the buffer
        """
        batch_size = channel_idx.shape[0]
        device = channel_idx.device

        # One-hot encodings
        ch_onehot = F.one_hot(channel_idx, self.config.n_channels).float()
        val_onehot = F.one_hot(value, self.config.n_values).float()
        combined = torch.cat([ch_onehot, val_onehot], dim=-1)

        if self.mode == "learned":
            return self.write_net(combined)

        elif self.mode == "structured":
            # Compute amplitude from learned mapping
            amp = self.amp_net(combined).squeeze(-1)  # (batch,)

            # Generate carrier at the correct frequency
            freqs = (channel_idx + 1).float()  # (batch,)
            # (batch, buffer_size)
            carriers = torch.sin(
                2 * math.pi * freqs.unsqueeze(1) * self.t.unsqueeze(0)
            )

            return amp.unsqueeze(1) * carriers


# ============================================================
# Read Head: learns to decompose the buffer signal and
# extract a specific channel's value
# ============================================================

class FDMReadHead(nn.Module):
    """
    Given a buffer signal and a channel index, extracts the value
    stored at that channel's carrier frequency.

    Two approaches:
    1. Structured: correlate with carrier, then classify
    2. Learned: full neural network decomposition
    """

    def __init__(self, config: FDMWeightConfig, mode: str = "structured"):
        super().__init__()
        self.config = config
        self.mode = mode

        self.register_buffer(
            "t", torch.linspace(0, config.buffer_size / config.fs,
                                config.buffer_size)
        )

        if mode == "learned":
            input_dim = config.buffer_size + config.n_channels
            self.read_net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.n_values),
            )
        elif mode == "structured":
            # Correlate buffer with carrier, then classify the correlation
            self.classifier = nn.Sequential(
                nn.Linear(2, config.hidden_dim),  # sin and cos correlations
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.n_values),
            )

    def forward(self, buffer_signal: torch.Tensor,
                channel_idx: torch.Tensor) -> torch.Tensor:
        """
        Read a channel value from the buffer.
        buffer_signal: (batch, buffer_size) or (buffer_size,) broadcast
        channel_idx: (batch,) channel to read
        Returns: (batch, n_values) logits
        """
        batch_size = channel_idx.shape[0]
        device = channel_idx.device

        # Ensure buffer is batched
        if buffer_signal.dim() == 1:
            buffer_signal = buffer_signal.unsqueeze(0).expand(batch_size, -1)

        if self.mode == "learned":
            ch_onehot = F.one_hot(channel_idx, self.config.n_channels).float()
            combined = torch.cat([buffer_signal, ch_onehot], dim=-1)
            return self.read_net(combined)

        elif self.mode == "structured":
            freqs = (channel_idx + 1).float()
            # Compute sin and cos correlations (matched filter)
            sin_carrier = torch.sin(
                2 * math.pi * freqs.unsqueeze(1) * self.t.unsqueeze(0)
            )
            cos_carrier = torch.cos(
                2 * math.pi * freqs.unsqueeze(1) * self.t.unsqueeze(0)
            )

            sin_corr = (buffer_signal * sin_carrier).mean(dim=-1, keepdim=True)
            cos_corr = (buffer_signal * cos_carrier).mean(dim=-1, keepdim=True)

            features = torch.cat([sin_corr, cos_corr], dim=-1)
            return self.classifier(features)


# ============================================================
# Full FDM-in-Weights System
# ============================================================

class FDMWeightMemory(nn.Module):
    """
    Complete FDM-in-weights system:
    - Memory buffer (parameter vector storing FDM signal)
    - Write head (encodes facts into buffer)
    - Read head (decodes facts from buffer)
    """

    def __init__(self, config: FDMWeightConfig,
                 write_mode: str = "structured",
                 read_mode: str = "structured"):
        super().__init__()
        self.config = config
        self.buffer = FDMMemoryBuffer(config)
        self.write_head = FDMWriteHead(config, mode=write_mode)
        self.read_head = FDMReadHead(config, mode=read_mode)

    def write(self, channel_idx: torch.Tensor, value: torch.Tensor):
        """Write a fact into the memory buffer."""
        update = self.write_head(channel_idx, value)
        # Average the batch updates and add to buffer
        with torch.no_grad():
            self.buffer.buffer.add_(update.mean(dim=0))

    def read(self, channel_idx: torch.Tensor) -> torch.Tensor:
        """Read a fact from the memory buffer."""
        return self.read_head(self.buffer.get_signal(), channel_idx)

    def write_all(self, channel_values: torch.Tensor):
        """
        Write all channel values at once.
        channel_values: (n_channels,) integer values
        """
        self.buffer.reset_buffer()
        for k in range(self.config.n_channels):
            ch = torch.tensor([k], device=channel_values.device)
            val = channel_values[k:k+1]
            update = self.write_head(ch, val)
            with torch.no_grad():
                self.buffer.buffer.add_(update.squeeze(0))


# ============================================================
# Training: teach write head to produce correct FDM signals
# and read head to decompose them
# ============================================================

def generate_training_batch(config: FDMWeightConfig, batch_size: int):
    """Generate a batch of (channel_values, query_channel, target_value)."""
    device = config.device
    # Random channel values for all channels
    channel_values = torch.randint(0, config.n_values,
                                   (batch_size, config.n_channels),
                                   device=device)
    # Random query channel
    query_channel = torch.randint(0, config.n_channels, (batch_size,),
                                  device=device)
    # Target value for the queried channel
    target_value = channel_values[torch.arange(batch_size), query_channel]

    return channel_values, query_channel, target_value


def train_phase1_read(config: FDMWeightConfig, read_head: FDMReadHead,
                      n_steps: int = 5000):
    """
    Phase 1: Train read head on ground-truth FDM signals.
    This establishes that the read mechanism works before training writes.
    """
    print("\n" + "="*60)
    print("PHASE 1: Training Read Head on Ground-Truth FDM Signals")
    print("="*60)

    optimizer = torch.optim.Adam(read_head.parameters(), lr=config.lr)
    read_head.to(config.device)

    for step in range(n_steps):
        channel_values, query_channel, target_value = \
            generate_training_batch(config, config.batch_size)

        # Generate ground-truth FDM signal
        signal = generate_fdm_signal(channel_values, config)

        # Read from signal
        logits = read_head(signal, query_channel)
        loss = F.cross_entropy(logits, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            acc = (logits.argmax(dim=-1) == target_value).float().mean()
            print(f"  Step {step+1}/{n_steps} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc.item()*100:.1f}%")

    # Final eval
    with torch.no_grad():
        channel_values, query_channel, target_value = \
            generate_training_batch(config, config.eval_samples)
        signal = generate_fdm_signal(channel_values, config)
        logits = read_head(signal, query_channel)
        acc = (logits.argmax(dim=-1) == target_value).float().mean()
        print(f"\n  Phase 1 Final Accuracy: {acc.item()*100:.1f}%")

    return acc.item()


def train_phase2_write_read(config: FDMWeightConfig,
                            write_head: FDMWriteHead,
                            read_head: FDMReadHead,
                            n_steps: int = 10000):
    """
    Phase 2: Train write head so that write→read recovers the correct value.
    Read head is frozen (already trained in Phase 1).
    """
    print("\n" + "="*60)
    print("PHASE 2: Training Write Head (Read Head Frozen)")
    print("="*60)

    # Freeze read head
    for p in read_head.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(write_head.parameters(), lr=config.lr)
    write_head.to(config.device)

    t = torch.linspace(0, config.buffer_size / config.fs,
                       config.buffer_size, device=config.device)

    for step in range(n_steps):
        channel_values, query_channel, target_value = \
            generate_training_batch(config, config.batch_size)

        # Build buffer by writing all channels
        # (differentiable: write head outputs are summed)
        buffer_signal = torch.zeros(config.batch_size, config.buffer_size,
                                    device=config.device)
        for k in range(config.n_channels):
            ch_idx = torch.full((config.batch_size,), k,
                                dtype=torch.long, device=config.device)
            vals = channel_values[:, k]
            update = write_head(ch_idx, vals)
            buffer_signal = buffer_signal + update

        # Read the queried channel
        logits = read_head(buffer_signal, query_channel)
        loss = F.cross_entropy(logits, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            acc = (logits.argmax(dim=-1) == target_value).float().mean()
            print(f"  Step {step+1}/{n_steps} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc.item()*100:.1f}%")

    # Unfreeze read head
    for p in read_head.parameters():
        p.requires_grad = True

    return evaluate_write_read(config, write_head, read_head)


def train_joint(config: FDMWeightConfig,
                write_head: FDMWriteHead,
                read_head: FDMReadHead,
                n_steps: int = 15000):
    """
    Joint training: train write and read heads together end-to-end.
    """
    print("\n" + "="*60)
    print("JOINT TRAINING: Write + Read End-to-End")
    print("="*60)

    params = list(write_head.parameters()) + list(read_head.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)
    write_head.to(config.device)
    read_head.to(config.device)

    for step in range(n_steps):
        channel_values, query_channel, target_value = \
            generate_training_batch(config, config.batch_size)

        # Build buffer by writing all channels (differentiable)
        buffer_signal = torch.zeros(config.batch_size, config.buffer_size,
                                    device=config.device)
        for k in range(config.n_channels):
            ch_idx = torch.full((config.batch_size,), k,
                                dtype=torch.long, device=config.device)
            vals = channel_values[:, k]
            update = write_head(ch_idx, vals)
            buffer_signal = buffer_signal + update

        # Read the queried channel
        logits = read_head(buffer_signal, query_channel)
        loss = F.cross_entropy(logits, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            acc = (logits.argmax(dim=-1) == target_value).float().mean()
            print(f"  Step {step+1}/{n_steps} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc.item()*100:.1f}%")

    return evaluate_write_read(config, write_head, read_head)


# ============================================================
# Evaluation
# ============================================================

def evaluate_write_read(config: FDMWeightConfig,
                        write_head: FDMWriteHead,
                        read_head: FDMReadHead):
    """Evaluate write→read accuracy across all channels."""
    print("\n" + "-"*60)
    print("EVALUATION: Write → Read Accuracy")
    print("-"*60)

    write_head.eval()
    read_head.eval()
    n_eval = config.eval_samples

    with torch.no_grad():
        channel_values = torch.randint(0, config.n_values,
                                       (n_eval, config.n_channels),
                                       device=config.device)

        # Build buffers
        buffer_signal = torch.zeros(n_eval, config.buffer_size,
                                    device=config.device)
        for k in range(config.n_channels):
            ch_idx = torch.full((n_eval,), k, dtype=torch.long,
                                device=config.device)
            vals = channel_values[:, k]
            update = write_head(ch_idx, vals)
            buffer_signal = buffer_signal + update

        # Per-channel accuracy
        per_channel_acc = []
        for k in range(config.n_channels):
            ch_idx = torch.full((n_eval,), k, dtype=torch.long,
                                device=config.device)
            logits = read_head(buffer_signal, ch_idx)
            preds = logits.argmax(dim=-1)
            targets = channel_values[:, k]
            acc = (preds == targets).float().mean().item()
            per_channel_acc.append(acc)

        # Print by frequency band (matching paper's Table 5)
        bands = [
            ("Low (ch0-9)", range(0, 10)),
            ("Mid-low (ch10-19)", range(10, 20)),
            ("Mid-high (ch20-29)", range(20, 30)),
            ("High (ch30-39)", range(30, 40)),
        ]
        for band_name, band_range in bands:
            band_accs = [per_channel_acc[k] for k in band_range
                         if k < config.n_channels]
            if band_accs:
                avg = np.mean(band_accs)
                lo = np.min(band_accs)
                hi = np.max(band_accs)
                print(f"  {band_name}: avg={avg*100:.1f}% "
                      f"(min={lo*100:.1f}%, max={hi*100:.1f}%)")

        overall = np.mean(per_channel_acc)
        print(f"\n  Overall per-channel accuracy: {overall*100:.1f}%")

        # Joint accuracy (all channels correct simultaneously)
        all_correct = torch.ones(n_eval, dtype=torch.bool,
                                 device=config.device)
        for k in range(config.n_channels):
            ch_idx = torch.full((n_eval,), k, dtype=torch.long,
                                device=config.device)
            logits = read_head(buffer_signal, ch_idx)
            preds = logits.argmax(dim=-1)
            targets = channel_values[:, k]
            all_correct &= (preds == targets)

        joint_acc = all_correct.float().mean().item()
        print(f"  Joint accuracy (all {config.n_channels} correct): "
              f"{joint_acc*100:.1f}%")

    write_head.train()
    read_head.train()

    return overall, per_channel_acc


def evaluate_sequential_writes(config: FDMWeightConfig,
                               write_head: FDMWriteHead,
                               read_head: FDMReadHead,
                               n_eval: int = 500):
    """
    Test catastrophic forgetting: write channels sequentially,
    check if earlier channels are corrupted by later writes.
    """
    print("\n" + "-"*60)
    print("EVALUATION: Sequential Write Forgetting Test")
    print("-"*60)

    write_head.eval()
    read_head.eval()

    with torch.no_grad():
        # Generate random values
        channel_values = torch.randint(0, config.n_values,
                                       (n_eval, config.n_channels),
                                       device=config.device)

        # Write channels one at a time, checking previous channels after each
        buffer_signal = torch.zeros(n_eval, config.buffer_size,
                                    device=config.device)

        forgetting_matrix = []

        for write_k in range(config.n_channels):
            # Write channel write_k
            ch_idx = torch.full((n_eval,), write_k, dtype=torch.long,
                                device=config.device)
            vals = channel_values[:, write_k]
            update = write_head(ch_idx, vals)
            buffer_signal = buffer_signal + update

            # Check all channels written so far
            row = []
            for read_k in range(write_k + 1):
                ch_read = torch.full((n_eval,), read_k, dtype=torch.long,
                                     device=config.device)
                logits = read_head(buffer_signal, ch_read)
                preds = logits.argmax(dim=-1)
                targets = channel_values[:, read_k]
                acc = (preds == targets).float().mean().item()
                row.append(acc)
            forgetting_matrix.append(row)

        # Report: accuracy of channel 0 after writing 1, 5, 10, 20, 40 channels
        checkpoints = [1, 5, 10, 20, min(40, config.n_channels)]
        print("  Channel 0 accuracy after N channels written:")
        for cp in checkpoints:
            if cp <= config.n_channels:
                acc = forgetting_matrix[cp - 1][0]
                print(f"    After {cp:2d} channels: {acc*100:.1f}%")

        # Also report accuracy of channel written vs channels at different points
        print("\n  Diagonal check (accuracy of ch K immediately after writing ch K):")
        for k in [0, 9, 19, 29, min(39, config.n_channels - 1)]:
            if k < config.n_channels:
                acc = forgetting_matrix[k][k]
                print(f"    Ch {k:2d}: {acc*100:.1f}%")

    write_head.train()
    read_head.train()
    return forgetting_matrix


def evaluate_channel_swap(config: FDMWeightConfig,
                          write_head: FDMWriteHead,
                          read_head: FDMReadHead,
                          n_eval: int = 500):
    """
    Channel swap test (analogous to paper's mechanistic validation):
    Change one channel's value, check that the read changes accordingly
    while other channels are unaffected.
    """
    print("\n" + "-"*60)
    print("EVALUATION: Channel Swap (Mechanistic Validation)")
    print("-"*60)

    write_head.eval()
    read_head.eval()

    with torch.no_grad():
        # Generate base values
        base_values = torch.randint(0, config.n_values,
                                    (n_eval, config.n_channels),
                                    device=config.device)

        # Build base buffer
        base_buffer = torch.zeros(n_eval, config.buffer_size,
                                  device=config.device)
        for k in range(config.n_channels):
            ch_idx = torch.full((n_eval,), k, dtype=torch.long,
                                device=config.device)
            vals = base_values[:, k]
            update = write_head(ch_idx, vals)
            base_buffer = base_buffer + update

        # Pick a channel to swap (channel 5)
        swap_ch = 5
        # Generate new values different from original
        new_values = (base_values[:, swap_ch] + 1) % config.n_values

        # Build swapped buffer
        swap_values = base_values.clone()
        swap_values[:, swap_ch] = new_values

        swap_buffer = torch.zeros(n_eval, config.buffer_size,
                                  device=config.device)
        for k in range(config.n_channels):
            ch_idx = torch.full((n_eval,), k, dtype=torch.long,
                                device=config.device)
            vals = swap_values[:, k]
            update = write_head(ch_idx, vals)
            swap_buffer = swap_buffer + update

        # Check: swapped channel should read new value
        ch_idx = torch.full((n_eval,), swap_ch, dtype=torch.long,
                            device=config.device)
        logits_base = read_head(base_buffer, ch_idx)
        logits_swap = read_head(swap_buffer, ch_idx)

        base_correct = (logits_base.argmax(-1) == base_values[:, swap_ch]).float().mean()
        swap_correct = (logits_swap.argmax(-1) == new_values).float().mean()

        print(f"  Swapped channel {swap_ch}:")
        print(f"    Base read accuracy:    {base_correct.item()*100:.1f}%")
        print(f"    Swapped read accuracy: {swap_correct.item()*100:.1f}%")

        # Check: other channels should be unaffected
        other_accs = []
        for k in range(config.n_channels):
            if k == swap_ch:
                continue
            ch_idx = torch.full((n_eval,), k, dtype=torch.long,
                                device=config.device)
            logits_b = read_head(base_buffer, ch_idx)
            logits_s = read_head(swap_buffer, ch_idx)
            # Both should match their respective targets
            acc_b = (logits_b.argmax(-1) == base_values[:, k]).float().mean()
            acc_s = (logits_s.argmax(-1) == swap_values[:, k]).float().mean()
            other_accs.append((acc_b.item(), acc_s.item()))

        avg_other_base = np.mean([a[0] for a in other_accs])
        avg_other_swap = np.mean([a[1] for a in other_accs])
        print(f"\n  Other channels (avg over {config.n_channels - 1} channels):")
        print(f"    Base accuracy:   {avg_other_base*100:.1f}%")
        print(f"    After swap:      {avg_other_swap*100:.1f}%")
        print(f"    Drift:           {abs(avg_other_base - avg_other_swap)*100:.2f}%")

    write_head.train()
    read_head.train()


# ============================================================
# Main Experiment Runner
# ============================================================

def run_experiment(n_channels: int = 10, write_mode: str = "structured",
                   read_mode: str = "structured", train_steps: int = 10000):
    """Run a single configuration experiment."""
    config = FDMWeightConfig(
        n_channels=n_channels,
        train_steps=train_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {n_channels} channels, "
          f"write={write_mode}, read={read_mode}")
    print(f"Device: {config.device}")
    print(f"{'='*60}")

    write_head = FDMWriteHead(config, mode=write_mode).to(config.device)
    read_head = FDMReadHead(config, mode=read_mode).to(config.device)

    # Phase 1: train read head on ground-truth signals
    read_acc = train_phase1_read(config, read_head, n_steps=5000)

    # Phase 2: train write head with frozen read head
    if read_acc > 0.5:
        overall, per_ch = train_phase2_write_read(
            config, write_head, read_head, n_steps=train_steps
        )

        # Forgetting test
        if overall > 0.5:
            evaluate_sequential_writes(config, write_head, read_head)
            evaluate_channel_swap(config, write_head, read_head)
    else:
        print("\n  Read head didn't converge, skipping write training.")
        print("  Trying joint training instead...")
        # Reset
        write_head = FDMWriteHead(config, mode=write_mode).to(config.device)
        read_head = FDMReadHead(config, mode=read_mode).to(config.device)
        overall, per_ch = train_joint(config, write_head, read_head,
                                       n_steps=train_steps * 2)

        if overall > 0.5:
            evaluate_sequential_writes(config, write_head, read_head)
            evaluate_channel_swap(config, write_head, read_head)

    return overall


def main():
    print("FDM-in-Weights: Proof of Concept")
    print("Can a neural network learn to WRITE and READ")
    print("frequency-multiplexed facts in a parameter buffer?")
    print()

    results = {}

    # Experiment 1: Small scale (10 channels), structured write+read
    results["10ch_structured"] = run_experiment(
        n_channels=10, write_mode="structured", read_mode="structured",
        train_steps=10000
    )

    # Experiment 2: Small scale, fully learned
    results["10ch_learned"] = run_experiment(
        n_channels=10, write_mode="learned", read_mode="learned",
        train_steps=15000
    )

    # Experiment 3: Scale to 20 channels, structured
    results["20ch_structured"] = run_experiment(
        n_channels=20, write_mode="structured", read_mode="structured",
        train_steps=15000
    )

    # Experiment 4: Full 40 channels, structured
    results["40ch_structured"] = run_experiment(
        n_channels=40, write_mode="structured", read_mode="structured",
        train_steps=20000
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, acc in results.items():
        print(f"  {name:25s}: {acc*100:.1f}%")

    print("\n" + "="*60)
    print("KEY QUESTIONS ANSWERED:")
    print("="*60)
    print("1. Can a write head learn to produce correct FDM signals?")
    print("2. Does carrier orthogonality prevent catastrophic forgetting?")
    print("3. Does the frequency-resolution gradient appear in weights")
    print("   the same way it does in context tokens?")
    print("4. Can a channel swap in the write path change the read output")
    print("   without affecting other channels?")


if __name__ == "__main__":
    main()
