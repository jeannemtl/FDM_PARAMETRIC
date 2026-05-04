"""
Generate Plaintext-K training and evaluation data for any K >= 8.

Channels 0-39 use hand-named vocabularies (Table 12 of paper draft 18);
channels 40+ use synthetic CH{idx:03d}_V{0..7} vocabularies.

Output format (JSONL, one per line):
    {
      "memory_block": "SECRET=GOLD\nLOCATION=PARIS\n...",
      "channel_order": [3, 17, 0, ...],
      "channel_values": {"0": "GOLD", ...},
      "question": "Report current status.",
      "answer": "CLEAR",
      "extra_channels": [8, 9, ..., K-1],
      "template": "1h_status",
      "hops": 1,
      "K": 40
    }

Usage:
    python make_plaintext_data.py --K 40  --out_dir data_K40/  --n_train 115000 --n_eval 1500
    python make_plaintext_data.py --K 160 --out_dir data_K160/ --n_train 115000 --n_eval 1500
    python make_plaintext_data.py --K 320 --out_dir data_K320/ --n_train 115000 --n_eval 1500
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from channels import (
    build_channel_vocab,
    sample_channel_values,
    get_template,
)


def build_memory_block(channel_values: Dict[int, str],
                       order: List[int],
                       channel_names: Dict[int, str]) -> str:
    """Build the [MEMORY] block: NAME=VALUE lines in the given channel order."""
    lines = [f"{channel_names[idx]}={channel_values[idx]}" for idx in order]
    return "\n".join(lines)


def make_sample(rng: random.Random, K: int, vocab, channel_names,
                randomize_order: bool, hop_count: int = None):
    values = sample_channel_values(rng, vocab)

    if hop_count is None:
        hop_count = rng.randint(1, 5)

    template_name, channels_used, question, reasoner = get_template(hop_count, rng)
    answer = reasoner(values)

    if randomize_order:
        order = list(range(K))
        rng.shuffle(order)
    else:
        order = list(range(K))

    extra_channels = list(range(8, K))   # all non-reasoning channels

    return {
        "memory_block": build_memory_block(values, order, channel_names),
        "channel_order": order,
        "channel_values": {str(k): v for k, v in values.items()},
        "question": question,
        "answer": str(answer),
        "extra_channels": extra_channels,
        "template": template_name,
        "hops": hop_count,
        "K": K,
    }


def build_prompt(sample: dict, channel_names: Dict[int, str],
                 include_answer: bool = True) -> str:
    prefix = (
        f"[MEMORY]\n{sample['memory_block']}\n[/MEMORY]\n"
        f"Question: {sample['question']}\nAnswer:"
    )
    if include_answer:
        extras = " ".join(
            f"{channel_names[i]}={sample['channel_values'][str(i)]}"
            for i in sample["extra_channels"]
        )
        return f"{prefix} {sample['answer']} [EXTRA] {extras}"
    return prefix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=40, help="Number of channels (default 40)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_train", type=int, default=115_000)
    p.add_argument("--n_eval", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating Plaintext-{args.K} dataset...")
    print(f"  out_dir: {args.out_dir}")
    print(f"  K:       {args.K}")
    print(f"  n_train: {args.n_train:,}")
    print(f"  n_eval:  {args.n_eval} per split")

    vocab = build_channel_vocab(args.K)
    channel_names = {idx: name for idx, (name, _) in vocab.items()}
    rng = random.Random(args.seed)

    print(f"\nGenerating training samples...")
    with open(args.out_dir / "train.jsonl", "w") as f:
        for i in range(args.n_train):
            sample = make_sample(rng, args.K, vocab, channel_names, randomize_order=True)
            f.write(json.dumps(sample) + "\n")
            if (i + 1) % 20_000 == 0:
                print(f"  {i+1:,}")

    print(f"Generating eval (fixed ordering)...")
    eval_rng = random.Random(args.seed + 1)
    with open(args.out_dir / "eval_fixed.jsonl", "w") as f:
        for i in range(args.n_eval):
            sample = make_sample(eval_rng, args.K, vocab, channel_names,
                                  randomize_order=False)
            f.write(json.dumps(sample) + "\n")

    print(f"Generating eval (random ordering)...")
    eval_rng = random.Random(args.seed + 2)
    with open(args.out_dir / "eval_random.jsonl", "w") as f:
        for i in range(args.n_eval):
            sample = make_sample(eval_rng, args.K, vocab, channel_names,
                                  randomize_order=True)
            f.write(json.dumps(sample) + "\n")

    # Print sample for sanity check
    print("\n" + "=" * 70)
    print(f"SAMPLE PROMPT (K={args.K}):")
    print("=" * 70)
    sample = make_sample(random.Random(0), args.K, vocab, channel_names,
                          randomize_order=True)
    full = build_prompt(sample, channel_names, include_answer=True)
    print(full[:800] + ("\n  ...[truncated]\n" if len(full) > 800 else ""))
    print(f"  total chars: {len(full)}")
    print(f"  est tokens (chars/4): ~{len(full)//4}")
    print(f"  template: {sample['template']}, hops: {sample['hops']}")
    print(f"\nFiles written to {args.out_dir}/")


if __name__ == "__main__":
    main()
