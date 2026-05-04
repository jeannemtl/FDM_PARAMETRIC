"""
Throughput probe for FDM training runs.

Measures actual tokens/sec on the target H200 for each (model, seq_len, batch_size)
combination you plan to run. Uses synthetic data so we don't need the real FDM
encoder built yet — token-id distribution doesn't matter for throughput.

Usage:
    # Single config probe
    python probe_throughput.py --model Qwen/Qwen3-0.6B --seq_len 6144 --batch_size 2

    # Full sweep matching the K=256 experiment plan
    python probe_throughput.py --sweep

Output: per-config tok/s, step time, peak mem, and projected wall-clock for 115K
samples. Writes results to throughput_results.json for later reference.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ProbeResult:
    model_name: str
    seq_len: int
    batch_size: int
    grad_accum: int
    effective_batch_tokens: int
    n_warmup: int
    n_measure: int
    step_time_ms: float            # mean over measured steps
    step_time_std_ms: float        # std over measured steps
    tokens_per_sec: float          # batch * seq_len / step_time
    peak_memory_gb: float
    flash_attn_enabled: bool
    grad_checkpointing: bool
    # Projections for 115K samples
    projected_steps: int
    projected_hours: float
    notes: str = ""


def make_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device):
    """Random token IDs — distribution doesn't affect compute, only learning."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Loss masking: only compute loss on last 32 tokens (the "answer" — matches FDM training)
    labels = input_ids.clone()
    labels[:, :-32] = -100
    return input_ids, labels


def probe_config(
    model_name: str,
    seq_len: int,
    batch_size: int,
    grad_accum: int = 1,
    n_warmup: int = 5,
    n_measure: int = 20,
    use_flash_attn: bool = True,
    use_grad_checkpointing: bool = False,
    n_train_samples: int = 115_000,
) -> ProbeResult:
    device = torch.device("cuda")
    print(f"\n{'=' * 70}")
    print(f"Probing: {model_name} | seq_len={seq_len} | batch_size={batch_size} "
          f"| grad_accum={grad_accum}")
    print(f"{'=' * 70}")

    # Load model
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    notes_list = []
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(device)
        flash_active = (attn_impl == "flash_attention_2")
    except Exception as e:
        print(f"  flash_attn_2 unavailable ({e}); falling back to sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        flash_active = False
        notes_list.append("flash_attn_2 unavailable, used sdpa")

    if use_grad_checkpointing:
        model.gradient_checkpointing_enable()
        notes_list.append("grad_checkpointing=True")

    model.train()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = model.config.vocab_size

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Warmup — important: compile + first-iter overhead distorts measurements
    print(f"  warmup ({n_warmup} steps)...")
    for _ in range(n_warmup):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            input_ids, labels = make_synthetic_batch(batch_size, seq_len, vocab_size, device)
            out = model(input_ids=input_ids, labels=labels)
            (out.loss / grad_accum).backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Reset peak memory tracker so we measure peak during the timed loop
    torch.cuda.reset_peak_memory_stats()

    # Measure
    print(f"  measuring ({n_measure} steps)...")
    step_times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            input_ids, labels = make_synthetic_batch(batch_size, seq_len, vocab_size, device)
            out = model(input_ids=input_ids, labels=labels)
            (out.loss / grad_accum).backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    # Stats
    step_times_t = torch.tensor(step_times)
    mean_ms = step_times_t.mean().item()
    std_ms = step_times_t.std().item()
    eff_tokens_per_step = batch_size * grad_accum * seq_len
    toks_per_sec = eff_tokens_per_step / (mean_ms / 1000)

    # Project wall-clock for full training run
    projected_steps = n_train_samples // (batch_size * grad_accum)
    projected_hours = (projected_steps * mean_ms / 1000) / 3600

    result = ProbeResult(
        model_name=model_name,
        seq_len=seq_len,
        batch_size=batch_size,
        grad_accum=grad_accum,
        effective_batch_tokens=eff_tokens_per_step,
        n_warmup=n_warmup,
        n_measure=n_measure,
        step_time_ms=mean_ms,
        step_time_std_ms=std_ms,
        tokens_per_sec=toks_per_sec,
        peak_memory_gb=peak_mem_gb,
        flash_attn_enabled=flash_active,
        grad_checkpointing=use_grad_checkpointing,
        projected_steps=projected_steps,
        projected_hours=projected_hours,
        notes="; ".join(notes_list),
    )

    # Print summary
    print(f"\n  RESULTS:")
    print(f"    step_time:    {mean_ms:7.1f} ± {std_ms:5.1f} ms")
    print(f"    tokens/sec:   {toks_per_sec:8,.0f}")
    print(f"    peak mem:     {peak_mem_gb:6.1f} GB")
    print(f"    flash_attn:   {flash_active}")
    print(f"    proj. steps:  {projected_steps:,} (115K samples / {batch_size * grad_accum} eff. batch)")
    print(f"    PROJ HOURS:   {projected_hours:6.2f} h")

    # Cleanup before next config
    del model, optimizer
    torch.cuda.empty_cache()

    return result


def run_sweep():
    """Full sweep matching the K=256 experiment plan."""
    configs = [
        # (model, seq_len, batch_size, grad_accum, grad_ckpt, label)
        # Qwen3-0.6B
        ("Qwen/Qwen3-0.6B", 1024, 8,  1,  False, "Qwen3-0.6B  | FDM-256 fixed (1024 tok)"),
        ("Qwen/Qwen3-0.6B", 6144, 2,  4,  False, "Qwen3-0.6B  | FDM-256 matched / Plaintext-256"),
        # Hermes3-3B — start conservative on batch, may need grad_ckpt
        ("NousResearch/Hermes-3-Llama-3.2-3B", 1024, 4, 2, False,
         "Hermes3-3B  | FDM-256 fixed (1024 tok)"),
        ("NousResearch/Hermes-3-Llama-3.2-3B", 6144, 1, 8, True,
         "Hermes3-3B  | FDM-256 matched / Plaintext-256"),
    ]

    results = []
    for model_name, seq_len, bs, ga, gc, label in configs:
        print(f"\n\n>>> {label}")
        try:
            r = probe_config(
                model_name=model_name,
                seq_len=seq_len,
                batch_size=bs,
                grad_accum=ga,
                use_grad_checkpointing=gc,
            )
            results.append(r)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            print(f"  → Retrying with batch_size=1 + grad_checkpointing=True")
            torch.cuda.empty_cache()
            try:
                r = probe_config(
                    model_name=model_name,
                    seq_len=seq_len,
                    batch_size=1,
                    grad_accum=bs * ga,
                    use_grad_checkpointing=True,
                )
                r.notes = (r.notes + "; OOM retry with bs=1+gc=True").strip("; ")
                results.append(r)
            except Exception as e2:
                print(f"  Retry also failed: {e2}")

    # Summary table
    print("\n\n" + "=" * 90)
    print("SWEEP SUMMARY")
    print("=" * 90)
    print(f"{'Config':<55} {'tok/s':>10} {'mem GB':>8} {'hours':>7}")
    print("-" * 90)
    for r in results:
        label = f"{r.model_name.split('/')[-1]} | seq={r.seq_len} bs={r.batch_size}x{r.grad_accum}"
        print(f"{label:<55} {r.tokens_per_sec:>10,.0f} {r.peak_memory_gb:>8.1f} {r.projected_hours:>7.2f}")

    total_h = sum(r.projected_hours for r in results)
    # Each long-context config covers TWO conditions (matched FDM + plaintext-256)
    # so we double-count those
    long_ctx_h = sum(r.projected_hours for r in results if r.seq_len >= 4096)
    total_with_doubling = total_h + long_ctx_h
    print("-" * 90)
    print(f"  Sum of probed configs:                {total_h:.1f} h")
    print(f"  + Long-context configs run twice (matched FDM + plaintext):")
    print(f"     full 6-run experiment estimate:    {total_with_doubling:.1f} h")
    print(f"     at $3.50/hr H200:                  ${total_with_doubling * 3.50:.0f}")
    print(f"     at $4.00/hr H200:                  ${total_with_doubling * 4.00:.0f}")

    # Save
    with open("throughput_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Wrote throughput_results.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--no_flash", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="Run the full K=256 experiment sweep")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — this probe requires the H200")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total HBM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.sweep:
        run_sweep()
    else:
        if args.model is None:
            raise ValueError("--model required when not using --sweep")
        result = probe_config(
            model_name=args.model,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            use_flash_attn=not args.no_flash,
            use_grad_checkpointing=args.grad_ckpt,
        )
        with open("throughput_results.json", "w") as f:
            json.dump([asdict(result)], f, indent=2)


if __name__ == "__main__":
    main()
