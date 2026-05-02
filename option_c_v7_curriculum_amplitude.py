#!/usr/bin/env python3
"""
Option C v7: Three-stage curriculum with PROGRESSIVE AMPLITUDE CONTRAST.

Insight from your original FDM training methodology: the two-block model
was originally trained with a stage curriculum that progressed from
easy contrast (a_low=0.0) to hard contrast (a_low=0.25). When we tried
v4 and v6 at fixed a_low=0.25, we asked the model to maintain its
hardest-trained FDM regime while simultaneously learning weight-storage.

This v7 instead reproduces the original easy-to-hard schedule for
Type B (FDM-signal) batches while phasing in Type A (no-signal weight
retrieval) batches across stages.

Stage 1 (steps 0-30%): a_low=0.0 (easy contrast, OOD for the model
                       but only 6pp slot drop). 20% Type A, 80% Type B.
                       Model establishes weight-storage in a regime
                       where FDM gradient is gentle.

Stage 2 (steps 30-60%): a_low=0.10 (medium). 40% Type A, 60% Type B.
                        Ramp Type A while increasing FDM difficulty.

Stage 3 (steps 60-100%): a_low=0.25 (training/eval regime, hardest).
                         50% Type A, 50% Type B. Final consolidation.

Eval is always at a_low=0.25 (the canonical regime).

Pre-training sanity (Phase 0b) confirms slot accuracy at each a_low so we
know the curriculum stages are coherent before training begins.

Usage:
    python option_c_v7_curriculum_amplitude.py \\
        --base_model prompterminal/fdm-40ch-two-block-qwen3 \\
        --output_model_dir /workspace/FDM_IN_WEIGHTS/two_block_qwen3_v7 \\
        --n_steps 1500 --lr 1e-5
"""

import os, sys, json, random, time, argparse, shutil, re
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/root/FDM_IN_WEIGHTS/scripts')
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]
QUERY_CHANNELS = list(range(8, 40))


TRAIN_TEMPLATES = [
    "Question: What is {ch}? Answer:",
    "Question: Report {ch}. Answer:",
    "Question: Tell me {ch}. Answer:",
    "Question: Current {ch}? Answer:",
    "Question: Status of {ch}? Answer:",
]
EVAL_HELDOUT_TEMPLATES = [
    "Question: What's the {ch}? Answer:",
    "Question: Provide {ch} value. Answer:",
]


# Stage definitions: (a_low, type_a_prob)
# Stage 1: 0-30% steps, easy contrast, FDM-heavy
# Stage 2: 30-60% steps, medium contrast, balanced ramp
# Stage 3: 60-100% steps, training contrast, balanced
STAGE_DEFS = [
    (0.30, 0.00, 0.20),  # (end_frac, a_low, type_a_prob)
    (0.60, 0.10, 0.40),
    (1.00, 0.25, 0.50),
]


def make_memory(seed):
    rng = random.Random(seed)
    return {ch: rng.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}


def random_memory():
    return {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}


def make_pairs(M, templates):
    pairs = []
    for ch in QUERY_CHANNELS:
        for tpl in templates:
            prompt = tpl.format(ch=CHANNEL_NAMES[ch])
            answer = f" {CHANNEL_NAMES[ch]}={M[ch]}."
            pairs.append((prompt, answer, CHANNEL_NAMES[ch], M[ch]))
    return pairs


def build_encoders_per_stage(tokenizer):
    """Build one encoder per stage at the right a_low."""
    encoders = {}
    for end_frac, a_low, type_a_prob in STAGE_DEFS:
        encoders[a_low] = TurboFDMSignalEncoder(
            vocab_size=151936, tokenizer=tokenizer,
            num_tokens_per_encoder=256, sample_rate=100.0,
            a_high=1.0, a_low=a_low, num_levels=64, seed=42,
        )
    return encoders


def make_no_signal_batch(M_finetune):
    ch = random.choice(QUERY_CHANNELS)
    template = random.choice(TRAIN_TEMPLATES)
    prompt = template.format(ch=CHANNEL_NAMES[ch])
    answer = f" {CHANNEL_NAMES[ch]}={M_finetune[ch]}."
    return prompt, answer


def make_fdm_signal_batch(encoder):
    M_random = random_memory()
    ch = random.choice(QUERY_CHANNELS)
    template = random.choice(TRAIN_TEMPLATES)
    fdm_text, _ = encoder.encode_memory(M_random)
    prompt = (f"[MEMORY]{fdm_text}[/MEMORY]\n"
              f"{template.format(ch=CHANNEL_NAMES[ch])}")
    answer = f" {CHANNEL_NAMES[ch]}={M_random[ch]}."
    return prompt, answer


def get_stage(step, n_steps):
    """Return (stage_idx, a_low, type_a_prob) for the given step."""
    frac = step / n_steps
    cumul = 0.0
    for i, (end_frac, a_low, type_a_prob) in enumerate(STAGE_DEFS):
        if frac <= end_frac:
            return i + 1, a_low, type_a_prob
    # Last stage
    return len(STAGE_DEFS), STAGE_DEFS[-1][1], STAGE_DEFS[-1][2]


def train_step(model, tokenizer, prompt, answer, optimizer, device, max_len=2048):
    full_text = prompt + answer
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    prefix_len = len(full_ids) - len(answer_ids)

    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
        prefix_len = min(prefix_len, max_len - 1)

    input_tensor = torch.tensor([full_ids], device=device)
    labels = [-100] * prefix_len + full_ids[prefix_len:]
    label_tensor = torch.tensor([labels], device=device)

    outputs = model(input_ids=input_tensor, labels=label_tensor)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train_curriculum(model, tokenizer, encoders_by_alow, M_finetune,
                      n_steps, lr, device, log_every=50,
                      eval_callback=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr/20
    )
    model.train()
    losses_a = []
    losses_b = []
    t0 = time.time()
    intermediate_evals = []
    last_stage = 0

    for step in range(1, n_steps + 1):
        stage_idx, a_low, type_a_prob = get_stage(step, n_steps)

        # Stage transition logging + eval
        if stage_idx != last_stage:
            if last_stage > 0:
                # Eval at the end of the previous stage
                print(f"\n  --- Eval at end of Stage {last_stage} ---")
                if eval_callback is not None:
                    intermediate_evals.append(
                        eval_callback(f"end_of_stage_{last_stage}")
                    )
                    model.train()

            print(f"\n  Stage {stage_idx}: a_low={a_low}, "
                  f"type_a_prob={type_a_prob}")
            last_stage = stage_idx

        # Pick batch type
        if random.random() < type_a_prob:
            prompt, answer = make_no_signal_batch(M_finetune)
            loss = train_step(model, tokenizer, prompt, answer, optimizer, device)
            losses_a.append(loss)
        else:
            enc = encoders_by_alow[a_low]
            prompt, answer = make_fdm_signal_batch(enc)
            loss = train_step(model, tokenizer, prompt, answer, optimizer, device)
            losses_b.append(loss)

        scheduler.step()

        if step % log_every == 0:
            recent_a = losses_a[-log_every:] if losses_a else [0]
            recent_b = losses_b[-log_every:] if losses_b else [0]
            avg_a = np.mean(recent_a)
            avg_b = np.mean(recent_b)
            elapsed = time.time() - t0
            print(f"    Step {step:4d}/{n_steps} (stg {stage_idx}, a_low={a_low}) | "
                  f"loss_A {avg_a:.4f} (n={len(recent_a)}) | "
                  f"loss_B {avg_b:.4f} (n={len(recent_b)}) | "
                  f"{elapsed:.0f}s")

    return losses_a, losses_b, intermediate_evals


@torch.no_grad()
def eval_pairs(model, tokenizer, pairs, device, max_new_tokens=20):
    model.eval()
    correct = 0
    total = 0
    for prompt, _, ch_name, correct_value in pairs:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        if f"{ch_name}={correct_value}" in response:
            correct += 1
        total += 1
    return correct, total


@torch.no_grad()
def eval_fdm_proper(model, tokenizer, encoder, M_signal, M_label, device, n=10):
    model.eval()
    fdm_text, _ = encoder.encode_memory(M_signal)
    ch_names = [CHANNEL_NAMES[k] for k in QUERY_CHANNELS]
    question = f"Report values for: {', '.join(ch_names)}."
    prompt = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"

    slot_correct = 0
    slot_total = 0
    sample_all = 0
    sample_responses = []

    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=350, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        sample_responses.append(answer[:200])

        all_right = True
        for k in QUERY_CHANNELS:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(M_label[k]) + r"\b"
            if re.search(pat, answer):
                slot_correct += 1
            else:
                all_right = False
            slot_total += 1
        if all_right:
            sample_all += 1

    return {
        "slot_correct": slot_correct, "slot_total": slot_total,
        "sample_all": sample_all, "n_samples": n,
        "sample_responses": sample_responses[:1],
    }


def make_eval_callback(model, tokenizer, eval_encoder, M_finetune, M_eval,
                        heldout_pairs, device):
    def cb(label):
        ho_c, ho_t = eval_pairs(model, tokenizer, heldout_pairs, device)
        fdm = eval_fdm_proper(model, tokenizer, eval_encoder, M_eval, M_eval, device, 5)
        slot = fdm["slot_correct"] / fdm["slot_total"]
        joint = fdm["sample_all"] / fdm["n_samples"]
        result = {
            "label": label,
            "heldout": ho_c / ho_t,
            "fdm_slot": slot,
            "fdm_joint": joint,
        }
        print(f"    Held-out (M_finetune): {100*result['heldout']:.1f}%")
        print(f"    FDM slot (M_eval):     {100*result['fdm_slot']:.1f}%")
        print(f"    FDM joint (M_eval):    {100*result['fdm_joint']:.1f}%")
        return result
    return cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",
                        default="prompterminal/fdm-40ch-two-block-qwen3")
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--n_steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    out_real = os.path.realpath(args.output_model_dir)
    if os.path.exists(args.base_model):
        if os.path.realpath(args.base_model) == out_real:
            raise ValueError("output_model_dir must differ from base_model.")
    if os.path.exists(args.output_model_dir) and not args.allow_overwrite:
        raise ValueError(f"Output exists. Pass --allow_overwrite or pick new path.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("OPTION C v7: AMPLITUDE-CONTRAST CURRICULUM")
    print("=" * 70)
    print(f"Base:           {args.base_model}")
    print(f"Output:         {args.output_model_dir}")
    print(f"Steps / LR:     {args.n_steps} / {args.lr}")
    print(f"Stages:")
    for i, (end_frac, a_low, type_a_prob) in enumerate(STAGE_DEFS):
        prev_end = STAGE_DEFS[i-1][0] if i > 0 else 0.0
        print(f"  Stage {i+1}: steps {int(prev_end*args.n_steps):4d}-"
              f"{int(end_frac*args.n_steps):4d} | "
              f"a_low={a_low} | type_A={type_a_prob:.0%}")

    print(f"\nLoading {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, trust_remote_code=True
    ).to(device)

    encoders_by_alow = build_encoders_per_stage(tokenizer)
    eval_encoder = encoders_by_alow[0.25]  # canonical eval at a_low=0.25

    M_finetune = make_memory(seed=42)
    M_eval = make_memory(seed=999)
    differing = [ch for ch in QUERY_CHANNELS if M_finetune[ch] != M_eval[ch]]
    print(f"\nM_finetune: TEAM={M_finetune[8]}, REGION={M_finetune[9]}")
    print(f"M_eval:     TEAM={M_eval[8]}, REGION={M_eval[9]}")
    print(f"Differing channels: {len(differing)}/{len(QUERY_CHANNELS)}")

    heldout_pairs = make_pairs(M_finetune, EVAL_HELDOUT_TEMPLATES)

    # ----- Phase 0: Baseline eval (canonical a_low=0.25) -----
    print("\n" + "-" * 70)
    print("Phase 0: Baseline eval (a_low=0.25)")
    print("-" * 70)
    base_ho_c, base_ho_t = eval_pairs(model, tokenizer, heldout_pairs, device)
    base_fdm = eval_fdm_proper(model, tokenizer, eval_encoder, M_eval, M_eval, device, 10)
    base_slot = base_fdm["slot_correct"] / base_fdm["slot_total"]
    base_joint = base_fdm["sample_all"] / base_fdm["n_samples"]
    print(f"  Held-out (M_finetune): {100*base_ho_c/base_ho_t:.1f}% (expected ~0%)")
    print(f"  FDM slot (M_eval):     {100*base_slot:.1f}% (expected ~100%)")
    print(f"  FDM joint (M_eval):    {100*base_joint:.1f}%")

    # ----- Phase 0b: Confirm model handles all stage a_low values -----
    print("\n" + "-" * 70)
    print("Phase 0b: Verify base model decodes all stage a_low values")
    print("-" * 70)
    stage_baselines = {}
    for end_frac, a_low, type_a_prob in STAGE_DEFS:
        enc = encoders_by_alow[a_low]
        result = eval_fdm_proper(model, tokenizer, enc, M_eval, M_eval, device, 5)
        slot = result["slot_correct"] / result["slot_total"]
        stage_baselines[a_low] = slot
        print(f"  a_low={a_low}: slot {100*slot:.1f}%")
    if min(stage_baselines.values()) < 0.7:
        print("  WARNING: at least one stage's a_low gives slot <70%. "
              "Curriculum may not converge cleanly.")

    # ----- Phase 1: Curriculum training -----
    print("\n" + "-" * 70)
    print(f"Phase 1: Three-stage amplitude curriculum")
    print("-" * 70)
    eval_cb = make_eval_callback(model, tokenizer, eval_encoder,
                                   M_finetune, M_eval, heldout_pairs, device)
    losses_a, losses_b, intermediate_evals = train_curriculum(
        model, tokenizer, encoders_by_alow, M_finetune,
        args.n_steps, args.lr, device,
        eval_callback=eval_cb,
    )

    # ----- Phase 2: Final held-out -----
    print("\n" + "-" * 70)
    print("Phase 2: Held-out paraphrases (M_finetune)")
    print("-" * 70)
    ho_correct, ho_total = eval_pairs(model, tokenizer, heldout_pairs, device)
    print(f"  Held-out: {100*ho_correct/ho_total:.1f}%")

    # ----- Phase 3: Final FDM eval (canonical a_low=0.25) -----
    print("\n" + "-" * 70)
    print("Phase 3: FDM capability (canonical a_low=0.25)")
    print("-" * 70)
    after = eval_fdm_proper(model, tokenizer, eval_encoder, M_eval, M_eval, device, 10)
    after_slot = after["slot_correct"] / after["slot_total"]
    after_joint = after["sample_all"] / after["n_samples"]
    print(f"  Slot:  {100*after_slot:.1f}% (was {100*base_slot:.1f}%)")
    print(f"  Joint: {100*after_joint:.1f}% (was {100*base_joint:.1f}%)")
    print(f"  Sample response: {after['sample_responses'][0][:200]}")

    # ----- Phase 4: Bleed -----
    print("\n" + "-" * 70)
    print("Phase 4: Bleed (M_eval signal -> M_finetune values)")
    print("-" * 70)
    bleed = eval_fdm_proper(model, tokenizer, eval_encoder, M_eval, M_finetune, device, 10)
    bleed_rate = bleed["slot_correct"] / bleed["slot_total"]
    expected_bleed = (32 - len(differing)) / 32
    bleed_excess = bleed_rate - expected_bleed
    print(f"  Bleed: {100*bleed_rate:.1f}% (excess {100*bleed_excess:.1f}pp)")

    # ----- Phase 5: Save -----
    print("\n" + "-" * 70)
    print("Phase 5: Saving model")
    print("-" * 70)
    if os.path.exists(args.output_model_dir):
        shutil.rmtree(args.output_model_dir)
    os.makedirs(args.output_model_dir, exist_ok=True)
    model.save_pretrained(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)
    print(f"  Saved to {args.output_model_dir}")

    findings = {
        "config": {
            "base_model": args.base_model,
            "n_steps": args.n_steps, "lr": args.lr,
            "seed": args.seed,
            "stage_defs": STAGE_DEFS,
        },
        "M_finetune": {str(k): v for k, v in M_finetune.items()},
        "M_eval": {str(k): v for k, v in M_eval.items()},
        "n_differing_channels": len(differing),
        "stage_baselines": {str(k): v for k, v in stage_baselines.items()},
        "baseline_fdm_slot": base_slot,
        "baseline_fdm_joint": base_joint,
        "intermediate_evals": intermediate_evals,
        "heldout_after": ho_correct / ho_total,
        "fdm_slot_after": after_slot,
        "fdm_joint_after": after_joint,
        "bleed_rate": bleed_rate,
        "bleed_excess_pp": bleed_excess * 100,
        "training_loss_final_typeA": float(np.mean(losses_a[-30:])) if losses_a else None,
        "training_loss_final_typeB": float(np.mean(losses_b[-30:])) if losses_b else None,
    }
    with open(os.path.join(args.output_model_dir, "results.json"), "w") as f:
        json.dump(findings, f, indent=2)

    # ----- Summary -----
    ho_r = ho_correct / ho_total
    fdm_drop = base_slot - after_slot

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Held-out (M_finetune):   {100*ho_r:.1f}%")
    print(f"  FDM slot (M_eval):       {100*base_slot:.1f}% -> "
          f"{100*after_slot:.1f}% (drop {100*fdm_drop:.1f}pp)")
    print(f"  FDM joint (M_eval):      {100*base_joint:.1f}% -> {100*after_joint:.1f}%")
    print(f"  Bleed excess:            {100*bleed_excess:.1f}pp")
    print()
    print(f"  Compare to previous:")
    print(f"    v4 (no curriculum):    slot drop 15.6pp, bleed exc 21.9pp")
    print(f"    v6 (curriculum stages): slot drop 15.6pp, bleed exc 21.9pp")
    print(f"    v7 (amplitude curriculum): slot drop {100*fdm_drop:.1f}pp, "
          f"bleed exc {100*bleed_excess:.1f}pp")
    print()
    print(f"  Per-stage trajectory:")
    for ev in intermediate_evals:
        print(f"    {ev['label']:<25s} held {100*ev['heldout']:.1f}%, "
              f"FDM slot {100*ev['fdm_slot']:.1f}%, joint {100*ev['fdm_joint']:.1f}%")
    print(f"    {'final':<25s} held {100*ho_r:.1f}%, "
          f"FDM slot {100*after_slot:.1f}%, joint {100*after_joint:.1f}%")
    print()

    if ho_r > 0.9 and fdm_drop < 0.05 and bleed_excess < 0.05:
        verdict = "STRONG SUCCESS: clean separation of circuits"
    elif ho_r > 0.9 and fdm_drop < 0.10 and bleed_excess < 0.10:
        verdict = "SUCCESS: good separation with minor interference"
    elif ho_r > 0.9 and fdm_drop < 0.15 and bleed_excess < 0.15:
        verdict = "PARTIAL SUCCESS: better than v4/v6"
    elif ho_r > 0.9:
        verdict = "WEIGHT WINS: same trade as before"
    elif ho_r < 0.5:
        verdict = "UNDER-TRAINED: more steps needed"
    else:
        verdict = "MIXED"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
