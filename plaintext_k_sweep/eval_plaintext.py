"""
Evaluate a Plaintext-K trained model and produce Table-1-style metrics.

Reports the same five metrics as Table 1 of paper draft 18:
  - Action: final decision keyword from the reasoning chain
  - Rule:   action accuracy on samples where META=NONE (proxy)
  - Meta:   action accuracy on samples where META in {EMERGENCY, LOCKDOWN}
  - Fact:   participating-channel retrieval (action correct => facts read correctly)
  - Extra:  parallel retrieval of channels 8..K-1, slot-level

K is auto-detected from the eval dataset.

Usage:
    python eval_plaintext.py \
        --model_dir models/qwen3_plaintext160/final \
        --eval_path data_K160/eval_random.jsonl \
        --out_json results/qwen3_plaintext160_random.json
"""

import argparse
import json
import re
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from channels import build_channel_vocab


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    delta = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    upper = centre + delta
    lower = centre - delta
    return max(upper - p, p - lower)


def detect_K(jsonl_path: str) -> int:
    with open(jsonl_path) as f:
        first = json.loads(f.readline())
    if "K" in first:
        return first["K"]
    return max(first["extra_channels"]) + 1


def build_eval_prompt(sample):
    return (
        f"[MEMORY]\n{sample['memory_block']}\n[/MEMORY]\n"
        f"Question: {sample['question']}\nAnswer:"
    )


def parse_response(response_text):
    """Extract action token + extras dict from model output."""
    response_text = response_text.strip()
    parts = response_text.split("[EXTRA]", 1)
    action = parts[0].strip()
    extras = {}
    if len(parts) > 1:
        extras_text = parts[1]
        # Match NAME=VALUE pairs with values that may contain underscores/digits
        for match in re.finditer(r"([A-Z][A-Z0-9_]*)=([A-Z0-9_]+)", extras_text):
            extras[match.group(1)] = match.group(2)
    return action, extras


def estimate_max_new_tokens(K: int) -> int:
    """How many tokens we need to generate to capture the answer + all extras."""
    # ~5 tokens per "NAME=VALUE " pair, plus the action keyword (~3 tokens),
    # plus "[EXTRA] " marker (~2 tokens). Add slack.
    return 5 * (K - 8) + 50


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    responses = []
    for i in range(out.shape[0]):
        full_text = tokenizer.decode(out[i], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True).rstrip()
        if full_text.startswith(prompt_text):
            response = full_text[len(prompt_text):]
        else:
            response = full_text.rsplit("Answer:", 1)[-1]
        responses.append(response)
    return responses


def evaluate(model, tokenizer, eval_path, K, channel_names, batch_size=8):
    samples = []
    with open(eval_path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  loaded {len(samples)} eval samples from {eval_path}")
    max_new = estimate_max_new_tokens(K)
    print(f"  max_new_tokens: {max_new} (for K={K})")

    action_correct = 0
    rule_correct = 0
    meta_correct = 0
    fact_correct = 0
    fact_attempted = 0
    extra_slot_correct = 0
    extra_slot_attempted = 0

    by_template_action = defaultdict(lambda: [0, 0])
    by_hop_action = defaultdict(lambda: [0, 0])

    # Per-channel extra accuracy — useful for plotting "by channel index" curves
    per_channel_extra = defaultdict(lambda: [0, 0])  # [correct, attempted]

    n_meta_relevant = 0
    n_meta_none = 0

    t_eval = time.perf_counter()
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start: batch_start + batch_size]
        prompts = [build_eval_prompt(s) for s in batch]
        responses = generate_batch(model, tokenizer, prompts, max_new)

        for sample, response in zip(batch, responses):
            action, extras = parse_response(response)
            true_action = sample["answer"]
            true_values = sample["channel_values"]

            ok_action = (action == true_action)
            action_correct += int(ok_action)
            by_template_action[sample["template"]][1] += 1
            by_template_action[sample["template"]][0] += int(ok_action)
            by_hop_action[sample["hops"]][1] += 1
            by_hop_action[sample["hops"]][0] += int(ok_action)

            if true_values["7"] == "NONE":
                rule_correct += int(ok_action)
                n_meta_none += 1
            if true_values["7"] in ("EMERGENCY", "LOCKDOWN"):
                meta_correct += int(ok_action)
                n_meta_relevant += 1

            fact_attempted += 1
            fact_correct += int(ok_action)

            for ch_idx in sample["extra_channels"]:
                ch_name = channel_names[int(ch_idx)]
                true_val = true_values[str(ch_idx)]
                got_val = extras.get(ch_name)
                extra_slot_attempted += 1
                per_channel_extra[ch_idx][1] += 1
                if got_val == true_val:
                    extra_slot_correct += 1
                    per_channel_extra[ch_idx][0] += 1

        if (batch_start // batch_size) % 10 == 0:
            elapsed = time.perf_counter() - t_eval
            done = batch_start + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(samples) - done) / rate if rate > 0 else 0
            print(f"    [{done}/{len(samples)}] {rate:.1f} samp/s, ETA {eta/60:.1f}min")

    n_samples = len(samples)
    results = {
        "K": K,
        "n_samples": n_samples,
        "action": {
            "p": action_correct / n_samples,
            "n_correct": action_correct, "n": n_samples,
            "ci": wilson_halfwidth(action_correct / n_samples, n_samples),
        },
        "rule": {
            "p": rule_correct / max(n_meta_none, 1),
            "n_correct": rule_correct, "n": n_meta_none,
            "ci": wilson_halfwidth(rule_correct / max(n_meta_none, 1), max(n_meta_none, 1)),
        },
        "meta": {
            "p": meta_correct / max(n_meta_relevant, 1),
            "n_correct": meta_correct, "n": n_meta_relevant,
            "ci": wilson_halfwidth(meta_correct / max(n_meta_relevant, 1), max(n_meta_relevant, 1)),
        },
        "fact": {
            "p": fact_correct / fact_attempted,
            "n_correct": fact_correct, "n": fact_attempted,
            "ci": wilson_halfwidth(fact_correct / fact_attempted, fact_attempted),
        },
        "extra_slot": {
            "p": extra_slot_correct / extra_slot_attempted,
            "n_correct": extra_slot_correct, "n": extra_slot_attempted,
            "ci": wilson_halfwidth(extra_slot_correct / extra_slot_attempted, extra_slot_attempted),
        },
        "per_channel_extra": {
            str(ch): {"p": v[0] / v[1], "n_correct": v[0], "n": v[1]}
            for ch, v in sorted(per_channel_extra.items())
        },
        "by_hop": {
            str(h): {"p": v[0] / v[1], "n_correct": v[0], "n": v[1],
                     "ci": wilson_halfwidth(v[0] / v[1], v[1])}
            for h, v in sorted(by_hop_action.items())
        },
        "by_template": {
            t: {"p": v[0] / v[1], "n_correct": v[0], "n": v[1]}
            for t, v in sorted(by_template_action.items())
        },
    }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--eval_path", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    K = detect_K(args.eval_path)
    vocab = build_channel_vocab(K)
    channel_names = {idx: name for idx, (name, _) in vocab.items()}
    print(f"  K = {K}")

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        ).cuda()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).cuda()
    model.eval()

    t0 = time.perf_counter()
    results = evaluate(model, tokenizer, args.eval_path, K, channel_names,
                        batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0
    results["eval_seconds"] = elapsed
    results["model_dir"] = args.model_dir
    results["eval_path"] = args.eval_path

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Eval done in {elapsed/60:.1f} min")
    print(f"\n  Results ({Path(args.eval_path).name}, K={K}):")
    print(f"    Action: {results['action']['p']*100:6.2f}% ± {results['action']['ci']*100:.1f}  "
          f"(n={results['action']['n']})")
    print(f"    Rule:   {results['rule']['p']*100:6.2f}% ± {results['rule']['ci']*100:.1f}  "
          f"(n={results['rule']['n']})")
    print(f"    Meta:   {results['meta']['p']*100:6.2f}% ± {results['meta']['ci']*100:.1f}  "
          f"(n={results['meta']['n']})")
    print(f"    Fact:   {results['fact']['p']*100:6.2f}% ± {results['fact']['ci']*100:.1f}  "
          f"(n={results['fact']['n']})")
    print(f"    Extra:  {results['extra_slot']['p']*100:6.2f}% ± {results['extra_slot']['ci']*100:.1f}  "
          f"(slot n={results['extra_slot']['n']})")
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()
