"""
Generate Plaintext-40 training and evaluation data.

Training: random channel ordering per sample (you train on the harder version).
Eval: two splits — fixed channel ordering (ch0..ch39) AND random ordering, so we
can measure whether the model learned name-based retrieval (transfers to fixed)
vs position-based retrieval (would fail on the held-out condition).

Output format (JSONL, one per line):
    {
      "memory_block": "SECRET=GOLD\nLOCATION=PARIS\n...",
      "channel_order": [3, 17, 0, ...],         # which ch_idx at each position
      "channel_values": {"0": "GOLD", ...},     # ground-truth values
      "question": "Report current status.",
      "answer": "CLEAR",                         # action/decision token
      "extra_channels": [8, 9, ..., 39],         # which channels go into "Extra" eval
      "template": "1h_status",
      "hops": 1
    }

Usage:
    python make_plaintext_data.py --out_dir data/ --n_train 115000 --n_eval 1500
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from channels import (
    CHANNEL_NAMES,
    CHANNEL_VOCAB,
    sample_channel_values,
    get_template,
    TEMPLATES,
)


def build_memory_block(channel_values: Dict[int, str], order: List[int]) -> str:
    """Build the [MEMORY] block: NAME=VALUE lines in the given channel order."""
    lines = [f"{CHANNEL_NAMES[idx]}={channel_values[idx]}" for idx in order]
    return "\n".join(lines)


def make_sample(rng: random.Random, randomize_order: bool, hop_count: int = None):
    """Generate one (memory, question, answer) sample."""
    values = sample_channel_values(rng)

    # Pick a hop count (1..5, uniform if not specified — matches the n-hop training stage)
    if hop_count is None:
        hop_count = rng.randint(1, 5)

    template_name, channels_used, question, reasoner = get_template(hop_count, rng)
    answer = reasoner(values)

    # Channel ordering in the [MEMORY] block
    if randomize_order:
        order = list(range(40))
        rng.shuffle(order)
    else:
        order = list(range(40))

    # Extra channels (8-39) are scored for parallel-retrieval ("Extra") accuracy
    extra_channels = list(range(8, 40))

    return {
        "memory_block": build_memory_block(values, order),
        "channel_order": order,
        "channel_values": {str(k): v for k, v in values.items()},
        "question": question,
        "answer": str(answer),
        "extra_channels": extra_channels,
        "template": template_name,
        "hops": hop_count,
    }


def build_prompt(sample: dict, include_answer: bool = True) -> str:
    """Build the full prompt string used at training & inference."""
    prefix = (
        f"[MEMORY]\n{sample['memory_block']}\n[/MEMORY]\n"
        f"Question: {sample['question']}\nAnswer:"
    )
    if include_answer:
        # Extra-channel content also appears in the answer per Section 3 spec:
        # "channels 8-39 ... must appear verbatim in the answer"
        extras = " ".join(
            f"{CHANNEL_NAMES[i]}={sample['channel_values'][str(i)]}"
            for i in sample["extra_channels"]
        )
        return f"{prefix} {sample['answer']} [EXTRA] {extras}"
    return prefix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_train", type=int, default=115_000)
    p.add_argument("--n_eval", type=int, default=1500,
                   help="Per eval split (fixed-order and random-order each get this many)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per_hop_eval", action="store_true",
                   help="Stratify eval by hop count (1500 per hop = 7500 total per split)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Training: random ordering, mixed hop counts
    print(f"Generating {args.n_train} training samples (random ordering)...")
    with open(args.out_dir / "train.jsonl", "w") as f:
        for i in range(args.n_train):
            sample = make_sample(rng, randomize_order=True)
            f.write(json.dumps(sample) + "\n")
            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,}")

    # Eval: fixed ordering
    print(f"\nGenerating eval (fixed ordering)...")
    eval_rng = random.Random(args.seed + 1)
    n_eval_total = args.n_eval * 5 if args.per_hop_eval else args.n_eval
    with open(args.out_dir / "eval_fixed.jsonl", "w") as f:
        for i in range(n_eval_total):
            hop = (i % 5) + 1 if args.per_hop_eval else None
            sample = make_sample(eval_rng, randomize_order=False, hop_count=hop)
            f.write(json.dumps(sample) + "\n")

    # Eval: random ordering
    print(f"Generating eval (random ordering)...")
    eval_rng = random.Random(args.seed + 2)
    with open(args.out_dir / "eval_random.jsonl", "w") as f:
        for i in range(n_eval_total):
            hop = (i % 5) + 1 if args.per_hop_eval else None
            sample = make_sample(eval_rng, randomize_order=True, hop_count=hop)
            f.write(json.dumps(sample) + "\n")

    # Print one sample for sanity check
    print("\n" + "=" * 70)
    print("SAMPLE (for sanity check):")
    print("=" * 70)
    sample = make_sample(random.Random(0), randomize_order=True)
    print(build_prompt(sample, include_answer=True))
    print("=" * 70)
    print(f"  template: {sample['template']}, hops: {sample['hops']}")
    print(f"  channel_order[:10]: {sample['channel_order'][:10]}")
    print(f"\nFiles written to {args.out_dir}/")


if __name__ == "__main__":
    main()
