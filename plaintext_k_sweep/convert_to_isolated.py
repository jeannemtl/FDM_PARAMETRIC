"""
Convert K=960 plaintext data to isolated single-channel-query format.

Input: data_K960/{train,eval_random,eval_fixed}.jsonl with answer = "<action> [EXTRA] all 952 extras"
Output: data_K960_isolated/{train,eval_random,eval_fixed}.jsonl with answer = single channel value

Each input sample is converted to ONE output sample querying a random channel.
For eval splits, we query specific channels deterministically so per-channel results
are reproducible.

Output format matches fdm_K960_source.py for direct comparison:
{
    "memory_block": "CH001=V3\nCH456=V7\n...",
    "channel_order": [...],            # position of each channel in [MEMORY] block
    "channel_values": {"0": "VAL", ...},
    "query_channel": 456,
    "query_name": "CH456",
    "answer": "V7",
    "K": 960
}

Usage:
    python convert_to_isolated.py
"""

import json
import random
from pathlib import Path

import sys
sys.path.insert(0, '.')

# Map channel index -> name (rebuild from same logic as make_plaintext_data.py)
HAND_NAMED = {
    0: "SECRET", 1: "LOCATION", 2: "AGENT", 3: "STATUS", 4: "PRIORITY",
    5: "BACKUP", 6: "RULE", 7: "META", 8: "TEAM", 9: "REGION",
    10: "PHASE", 11: "COMM", 12: "ASSET", 13: "WINDOW", 14: "COVER",
    15: "SUPPORT", 16: "THREAT", 17: "WEATHER", 18: "TERRAIN", 19: "EXTRACT",
    20: "CIPHER", 21: "FREQ", 22: "PAYLOAD", 23: "ROUTE", 24: "DURATION",
    25: "CONTACT", 26: "FUEL", 27: "ALTITUDE", 28: "VISIBILITY", 29: "NOISE",
    30: "FORMATION", 31: "ARMOR", 32: "SIGNAL", 33: "MORALE", 34: "SUPPLY",
    35: "INTEL", 36: "EVAC", 37: "WEATHER2", 38: "DOCTRINE", 39: "COMMS",
}


def channel_name(idx, K=960):
    if idx < 40:
        return HAND_NAMED[idx]
    return f"CH{idx:03d}"


def convert_sample(sample, query_channel, K=960):
    """Build one isolated-query sample from one input sample."""
    cv = sample["channel_values"]
    answer = cv[str(query_channel)]
    return {
        "memory_block": sample["memory_block"],
        "channel_order": sample.get("channel_order", list(range(K))),
        "channel_values": cv,
        "query_channel": query_channel,
        "query_name": channel_name(query_channel, K),
        "answer": answer,
        "K": K,
    }


def convert_split(in_path, out_path, mode="random", K=960, seed=42):
    """
    mode='random': each sample queries a random channel (training)
    mode='cycle':  channel index cycles through 0..K-1 deterministically (eval)
    """
    rng = random.Random(seed)
    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for i, line in enumerate(fin):
            sample = json.loads(line)
            if mode == "random":
                qch = rng.randrange(K)
            else:  # cycle through channels deterministically across the split
                qch = i % K
            new_sample = convert_sample(sample, qch, K)
            fout.write(json.dumps(new_sample) + "\n")
            n += 1
    return n


def main():
    in_dir = Path("data_K960")
    out_dir = Path("data_K960_isolated")
    out_dir.mkdir(exist_ok=True)

    splits = [
        ("train.jsonl",       "random", 42),
        ("eval_random.jsonl", "cycle",  43),
        ("eval_fixed.jsonl",  "cycle",  44),
    ]

    print(f"Converting K=960 plaintext data: {in_dir} -> {out_dir}")
    for filename, mode, seed in splits:
        in_path = in_dir / filename
        out_path = out_dir / filename
        if not in_path.exists():
            print(f"  SKIP: {in_path} (not found)")
            continue
        n = convert_split(in_path, out_path, mode=mode, seed=seed)
        print(f"  {filename:<20} {mode:<8} -> {n:>6,} samples")

    # Sanity check: print one converted sample
    with open(out_dir / "train.jsonl") as f:
        sample = json.loads(f.readline())
    print(f"\nSAMPLE:")
    print(f"  query_channel: {sample['query_channel']} ({sample['query_name']})")
    print(f"  answer:        {sample['answer']}")
    print(f"  memory_block (first 200 chars): {sample['memory_block'][:200]}...")
    print(f"  memory_block total chars: {len(sample['memory_block'])}")
    # Check that the answer matches what's in the memory block
    assert f"{sample['query_name']}={sample['answer']}" in sample['memory_block'], \
        "MISMATCH: query/answer not found in memory block"
    print(f"  consistency check: PASSED")
    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
