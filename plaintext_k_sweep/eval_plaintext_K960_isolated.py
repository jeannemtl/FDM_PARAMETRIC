"""
Evaluate Plaintext-K=960 isolated model on isolated single-channel queries.

Same eval format as fdm_K960_source.py's eval — one query per sample, single
value answer. Reports retrieval accuracy with per-channel-position binning
(U-shape detection) and per-channel-index.

Usage:
    python -u eval_plaintext_isolated.py 2>&1 | tee logs/plaintext_K960_isolated_eval.log
"""

import sys, os, json, re, time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = "data_K960_isolated"
MODEL_DIR = "models/qwen3_0p6b_plaintext960_isolated/final"
RESULTS_DIR = "results"
K = 960


def wilson_halfwidth(p, n, z=1.96):
    if n == 0:
        return float("nan")
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    delta = (z/denom) * ((p*(1-p)/n + z*z/(4*n*n))**0.5)
    return max(centre + delta - p, p - (centre - delta))


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens=12):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=False).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    responses = []
    for i in range(out.shape[0]):
        prompt_len = inputs["input_ids"][i].shape[0]
        new_tokens = out[i, prompt_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return responses


def parse_value(response):
    """Extract single value token from a short answer."""
    s = response.strip()
    if not s:
        return ""
    # Remove leading "=" if model echoed CHXXX=
    if s.startswith("="):
        s = s[1:].strip()
    return s.split()[0].rstrip(".,;:[]\\n")


def evaluate(eval_path, label):
    print(f"\n{'='*60}")
    print(f"  EVAL: {label}")
    print(f"  Data: {eval_path}")
    print(f"{'='*60}\n")

    print(f"Loading {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", trust_remote_code=True,
        ).cuda()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).cuda()
    model.eval()

    samples = [json.loads(l) for l in open(eval_path)]
    print(f"  loaded {len(samples)} eval samples")

    correct = 0
    by_channel = defaultdict(lambda: [0, 0])         # ch_idx -> [correct, total]
    by_position_decile = defaultdict(lambda: [0, 0]) # position decile -> [correct, total]

    BATCH_SIZE = 4

    t0 = time.perf_counter()
    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch = samples[batch_start:batch_start + BATCH_SIZE]
        prompts = [
            f"[MEMORY]\n{s['memory_block']}\n[/MEMORY]\n"
            f"Question: What is the value of {s['query_name']}?\nAnswer:"
            for s in batch
        ]
        responses = generate_batch(model, tokenizer, prompts)
        for s, resp in zip(batch, responses):
            predicted = parse_value(resp)
            ok = (predicted == s["answer"])
            correct += int(ok)

            ch = s["query_channel"]
            by_channel[ch][0] += int(ok)
            by_channel[ch][1] += 1

            # Position of query channel in the [MEMORY] block of this sample
            try:
                position = s["channel_order"].index(ch)
                decile = min(9, position * 10 // K)
                by_position_decile[decile][0] += int(ok)
                by_position_decile[decile][1] += 1
            except (ValueError, KeyError):
                pass

        if (batch_start // BATCH_SIZE) % 20 == 0:
            done = batch_start + len(batch)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(samples) - done) / rate if rate > 0 else 0
            print(f"    [{done}/{len(samples)}] {rate:.1f} samp/s, ETA {eta/60:.1f}min")

    p = correct / len(samples)
    ci = wilson_halfwidth(p, len(samples))

    print(f"\n  Eval done in {(time.perf_counter()-t0)/60:.1f} min")
    print(f"  Single-channel retrieval (Plaintext-K=960): "
          f"{p*100:.2f}% ± {ci*100:.2f}  (n={len(samples)})")

    print(f"\n  Position decile breakdown (U-shape detection):")
    for d in sorted(by_position_decile.keys()):
        c, t = by_position_decile[d]
        bp = c/t if t else 0
        bar = "#" * int(bp * 40)
        print(f"    pos decile {d}  (positions ~{d*K//10:>4}-{(d+1)*K//10-1:<4})  "
              f"{bp*100:5.1f}%  (n={t})  {bar}")

    out = {
        "label": label,
        "K": K,
        "model_dir": MODEL_DIR,
        "eval_path": eval_path,
        "n_samples": len(samples),
        "single_channel_retrieval": {
            "p": p, "n_correct": correct, "n": len(samples), "ci": ci,
        },
        "by_position_decile": {
            str(d): {"p": v[0]/v[1] if v[1] else 0,
                     "n_correct": v[0], "n": v[1]}
            for d, v in sorted(by_position_decile.items())
        },
        "by_channel": {
            str(ch): {"p": v[0]/v[1], "n_correct": v[0], "n": v[1]}
            for ch, v in sorted(by_channel.items())
        },
    }
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"plaintext_K960_isolated_{label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


def main():
    # Run on both eval splits
    evaluate(os.path.join(DATA_DIR, "eval_random.jsonl"), "random")
    evaluate(os.path.join(DATA_DIR, "eval_fixed.jsonl"), "fixed")


if __name__ == "__main__":
    main()
