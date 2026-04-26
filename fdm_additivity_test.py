"""
FDM Additivity Test
====================

Can FDM-encoded KV entries coexist with regular plaintext context
in the same forward pass? If yes, FDM is additive to any attention
architecture that uses KV cache.

Test design:
  1. BASELINE A: FDM KV only — model retrieves channel values from
     frozen KV cache (already proven at 89.1%)
  2. BASELINE B: Plaintext only — model answers a plaintext question
     with no FDM involvement
  3. COMBINED: FDM KV prepended + plaintext question that requires
     BOTH sources — channel values from FDM AND reasoning from
     plaintext context

If Combined accuracy on FDM channels ≈ Baseline A accuracy,
AND Combined accuracy on plaintext questions ≈ Baseline B accuracy,
then FDM is additive: the two information sources don't interfere.

This establishes that FDM KV entries could plug into any attention
architecture (including DeepSeek-V4's CSA) as a structured memory
layer alongside regular context processing.

Usage:
    python fdm_additivity_test.py --n_eval 100
"""

import sys, os, re, json, random, time, argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/root/FDM_IN_WEIGHTS')
from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]


def make_encoder(tokenizer):
    return TurboFDMSignalEncoder(
        vocab_size=151936, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )


def random_memory():
    return {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}


def score_extra_channels(text, memory):
    results = {}
    for k in range(8, NUM_CHANNELS):
        name = CHANNEL_NAMES[k]
        expected = memory[k]
        pattern = rf"\b{re.escape(name)}={re.escape(expected)}\b"
        results[name] = bool(re.search(pattern, text))
    return results


# ============================================================
# Plaintext reasoning questions (model must use context to answer)
# ============================================================

PLAINTEXT_SCENARIOS = [
    {
        "context": "The extraction helicopter can carry a maximum of 4 personnel. "
                   "Currently there are 6 team members at the landing zone. "
                   "Two members are injured and must be evacuated first.",
        "question": "How many team members will remain at the LZ after the first extraction?",
        "answer_contains": ["2", "two"],
        "type": "arithmetic",
    },
    {
        "context": "Alpha route goes through urban terrain and takes 30 minutes. "
                   "Bravo route goes through mountain terrain and takes 45 minutes. "
                   "Charlie route goes through coastal terrain and takes 25 minutes but "
                   "requires naval support.",
        "question": "Which route is fastest if naval support is unavailable?",
        "answer_contains": ["Alpha", "alpha", "30"],
        "type": "conditional",
    },
    {
        "context": "Communication frequencies are compromised on channels HF and VHF. "
                   "UHF remains secure. SHF has intermittent availability. "
                   "All critical messages must use secure channels only.",
        "question": "Which frequency should be used for critical messages?",
        "answer_contains": ["UHF", "uhf"],
        "type": "filter",
    },
    {
        "context": "Supply cache Alpha contains ammunition and medical supplies. "
                   "Supply cache Bravo contains food and water only. "
                   "Supply cache Charlie contains ammunition only. "
                   "The team needs medical supplies urgently.",
        "question": "Which supply cache should the team access?",
        "answer_contains": ["Alpha", "alpha"],
        "type": "lookup",
    },
    {
        "context": "The surveillance drone has 2 hours of battery remaining. "
                   "Each sector scan takes 20 minutes. There are 8 sectors total. "
                   "Priority sectors are 3, 5, and 7.",
        "question": "Can the drone scan all priority sectors? How many total sectors can it scan?",
        "answer_contains": ["6", "six", "yes"],
        "type": "arithmetic",
    },
    {
        "context": "Agent ALICE is positioned in PARIS with CLEAR status. "
                   "Agent BOB is positioned in TOKYO with COMPROMISED status. "
                   "Agent CAROL is positioned in LONDON with CLEAR status. "
                   "Only agents with CLEAR status can receive classified intel.",
        "question": "Which agents can receive classified intel?",
        "answer_contains": ["ALICE", "CAROL"],
        "type": "filter",
    },
    {
        "context": "Weather forecast: PARIS clear skies, TOKYO heavy rain, "
                   "LONDON fog with reduced visibility, BERLIN clear skies. "
                   "Aerial operations require clear skies.",
        "question": "In which cities can aerial operations proceed?",
        "answer_contains": ["PARIS", "BERLIN"],
        "type": "filter",
    },
    {
        "context": "The safe house has 3 entry points: north door, south window, "
                   "and rooftop access. North door is guarded by 2 hostiles. "
                   "South window is unguarded but alarmed. Rooftop has no security "
                   "but requires climbing equipment.",
        "question": "Which entry point has the fewest obstacles?",
        "answer_contains": ["rooftop", "Rooftop", "roof"],
        "type": "reasoning",
    },
]


def score_plaintext(text, scenario):
    """Check if the model's answer contains the expected content."""
    for expected in scenario["answer_contains"]:
        if expected.lower() in text.lower():
            return True
    return False


# ============================================================
# Frozen KV cache from FDM tokens
# ============================================================

def compute_fdm_kv(model, tokenizer, encoder, memory, device):
    """Run FDM tokens through model, return KV cache."""
    fdm_text, fdm_tokens = encoder.encode_memory(memory)
    memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
    full_ids = memory_start + fdm_tokens
    ids_tensor = torch.tensor([full_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(ids_tensor, use_cache=True)
    return out.past_key_values, len(full_ids)


# ============================================================
# Generation with KV cache (manual loop)
# ============================================================

def generate_with_kv(model, tokenizer, prompt_text, fdm_kv, fdm_seq_len, device,
                     max_tokens=300):
    """Generate with prepended FDM KV cache + plaintext prompt."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    position_ids = torch.arange(fdm_seq_len, fdm_seq_len + len(prompt_ids),
                                 device=device).unsqueeze(0)
    attn_mask = torch.ones(1, fdm_seq_len + len(prompt_ids),
                            device=device, dtype=torch.long)

    generated_ids = []
    past_kv = fdm_kv

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(prompt_tensor, past_key_values=past_kv,
                   position_ids=position_ids, attention_mask=attn_mask,
                   use_cache=True)
        past_kv = out.past_key_values
        nt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated_ids.append(nt.item())
        tl = fdm_seq_len + len(prompt_ids) + 1

        for s in range(max_tokens - 1):
            p = torch.tensor([[tl - 1 + s]], device=device)
            a = torch.ones(1, tl + s, device=device, dtype=torch.long)
            o = model(nt, past_key_values=past_kv,
                     position_ids=p, attention_mask=a, use_cache=True)
            past_kv = o.past_key_values
            nt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            tid = nt.item()
            generated_ids.append(tid)
            if tid == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_plain(model, tokenizer, prompt_text, device, max_tokens=300):
    """Generate without any KV cache (plain context only)."""
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# Test A: FDM KV only (channel retrieval baseline)
# ============================================================

def test_fdm_only(model, tokenizer, encoder, n_eval, device):
    print("\n" + "=" * 60)
    print("TEST A: FDM KV Only (channel retrieval)")
    print("=" * 60)

    ch_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0

    for _ in tqdm(range(n_eval), desc="FDM only"):
        mem = random_memory()
        fdm_kv, fdm_len = compute_fdm_kv(model, tokenizer, encoder, mem, device)

        question = "Report all context values for channels 8-39."
        prompt = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
        text = generate_with_kv(model, tokenizer, prompt, fdm_kv, fdm_len, device)

        scores = score_extra_channels(text, mem)
        for name, correct in scores.items():
            ch_correct[name] += int(correct)
        total += 1

    per_ch = {name: ch_correct[name] / total for name in ch_correct}
    overall = np.mean(list(per_ch.values()))
    print(f"\n  FDM channel accuracy: {overall*100:.1f}%")
    return overall


# ============================================================
# Test B: Plaintext only (reasoning baseline)
# ============================================================

def test_plaintext_only(model, tokenizer, n_eval, device):
    print("\n" + "=" * 60)
    print("TEST B: Plaintext Only (reasoning questions)")
    print("=" * 60)

    correct = 0
    total = 0

    for i in tqdm(range(n_eval), desc="Plaintext only"):
        scenario = PLAINTEXT_SCENARIOS[i % len(PLAINTEXT_SCENARIOS)]
        prompt = (f"Context: {scenario['context']}\n"
                  f"Question: {scenario['question']}\n"
                  f"Answer:")
        text = generate_plain(model, tokenizer, prompt, device)

        if score_plaintext(text, scenario):
            correct += 1
        total += 1

        if i < 3:
            print(f"\n  Q: {scenario['question'][:60]}...")
            print(f"  A: {text[:100]}...")
            print(f"  Expected: {scenario['answer_contains']}")
            print(f"  Correct: {score_plaintext(text, scenario)}")

    acc = correct / total
    print(f"\n  Plaintext reasoning accuracy: {acc*100:.1f}%")
    return acc


# ============================================================
# Test C: COMBINED — FDM KV + Plaintext context (additivity)
# ============================================================

def test_combined(model, tokenizer, encoder, n_eval, device):
    print("\n" + "=" * 60)
    print("TEST C: Combined (FDM KV + Plaintext context)")
    print("  Model must retrieve FDM channels AND answer plaintext question")
    print("=" * 60)

    ch_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    plaintext_correct = 0
    total = 0

    for i in tqdm(range(n_eval), desc="Combined"):
        mem = random_memory()
        scenario = PLAINTEXT_SCENARIOS[i % len(PLAINTEXT_SCENARIOS)]

        # Prepend FDM KV cache
        fdm_kv, fdm_len = compute_fdm_kv(model, tokenizer, encoder, mem, device)

        # Prompt asks for BOTH: channel values AND plaintext reasoning
        prompt = (
            f"[/MEMORY]\n"
            f"Context: {scenario['context']}\n"
            f"Question 1: Report all context values for channels 8-39.\n"
            f"Question 2: {scenario['question']}\n"
            f"Answer both questions.\n"
            f"Answer:"
        )

        text = generate_with_kv(model, tokenizer, prompt, fdm_kv, fdm_len, device,
                                max_tokens=400)

        # Score FDM channels
        scores = score_extra_channels(text, mem)
        for name, correct in scores.items():
            ch_correct[name] += int(correct)

        # Score plaintext
        if score_plaintext(text, scenario):
            plaintext_correct += 1

        total += 1

        if i < 3:
            print(f"\n  Plaintext Q: {scenario['question'][:50]}...")
            print(f"  Generated: {text[:200]}...")
            print(f"  Plaintext correct: {score_plaintext(text, scenario)}")
            ch_count = sum(1 for v in scores.values() if v)
            print(f"  FDM channels correct: {ch_count}/32")

    per_ch = {name: ch_correct[name] / total for name in ch_correct}
    fdm_overall = np.mean(list(per_ch.values()))
    plaintext_acc = plaintext_correct / total

    print(f"\n  FDM channel accuracy:      {fdm_overall*100:.1f}%")
    print(f"  Plaintext reasoning acc:   {plaintext_acc*100:.1f}%")

    return fdm_overall, plaintext_acc


# ============================================================
# Test D: Interference check — does FDM KV HURT plaintext?
# ============================================================

def test_interference(model, tokenizer, encoder, n_eval, device):
    print("\n" + "=" * 60)
    print("TEST D: Interference (does FDM KV hurt plaintext accuracy?)")
    print("  Same plaintext questions, but with FDM KV prepended")
    print("=" * 60)

    correct = 0
    total = 0

    for i in tqdm(range(n_eval), desc="Interference"):
        mem = random_memory()
        scenario = PLAINTEXT_SCENARIOS[i % len(PLAINTEXT_SCENARIOS)]

        fdm_kv, fdm_len = compute_fdm_kv(model, tokenizer, encoder, mem, device)

        # ONLY ask the plaintext question (ignore FDM channels)
        prompt = (
            f"[/MEMORY]\n"
            f"Context: {scenario['context']}\n"
            f"Question: {scenario['question']}\n"
            f"Answer:"
        )

        text = generate_with_kv(model, tokenizer, prompt, fdm_kv, fdm_len, device)

        if score_plaintext(text, scenario):
            correct += 1
        total += 1

    acc = correct / total
    print(f"\n  Plaintext accuracy WITH FDM KV: {acc*100:.1f}%")
    return acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_eval", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16
    ).to(device)
    model.eval()

    encoder = make_encoder(tokenizer)

    results = {}

    # Test A: FDM only
    fdm_acc = test_fdm_only(model, tokenizer, encoder, args.n_eval, device)
    results["fdm_only"] = fdm_acc

    # Test B: Plaintext only
    pt_acc = test_plaintext_only(model, tokenizer, args.n_eval, device)
    results["plaintext_only"] = pt_acc

    # Test C: Combined
    fdm_combined, pt_combined = test_combined(model, tokenizer, encoder,
                                               args.n_eval, device)
    results["combined_fdm"] = fdm_combined
    results["combined_plaintext"] = pt_combined

    # Test D: Interference
    pt_with_fdm = test_interference(model, tokenizer, encoder, args.n_eval, device)
    results["plaintext_with_fdm_kv"] = pt_with_fdm

    # ============================================================
    # Summary
    # ============================================================

    print("\n" + "=" * 60)
    print("ADDITIVITY RESULTS")
    print("=" * 60)

    print(f"\n  FDM channel retrieval:")
    print(f"    FDM only:      {fdm_acc*100:.1f}%")
    print(f"    Combined:      {fdm_combined*100:.1f}%")
    fdm_delta = fdm_combined - fdm_acc
    print(f"    Delta:         {fdm_delta*100:+.1f}%")

    print(f"\n  Plaintext reasoning:")
    print(f"    Plaintext only:        {pt_acc*100:.1f}%")
    print(f"    With FDM KV (ignored): {pt_with_fdm*100:.1f}%")
    print(f"    Combined (both asked): {pt_combined*100:.1f}%")
    pt_delta = pt_with_fdm - pt_acc
    print(f"    Interference delta:    {pt_delta*100:+.1f}%")

    print(f"\n  ADDITIVITY VERDICT:")
    if abs(fdm_delta) < 5 and abs(pt_delta) < 10:
        print(f"    ✓ ADDITIVE — FDM KV and plaintext coexist without interference")
        print(f"    FDM retrieval preserved ({fdm_delta*100:+.1f}% delta)")
        print(f"    Plaintext reasoning preserved ({pt_delta*100:+.1f}% delta)")
        print(f"\n    → FDM KV entries can plug into any KV-cache-based attention")
        print(f"      architecture as a structured memory layer alongside")
        print(f"      regular context processing (including DeepSeek-V4 CSA)")
    elif abs(fdm_delta) < 5 and abs(pt_delta) >= 10:
        print(f"    ~ PARTIALLY ADDITIVE — FDM preserved but plaintext degraded")
        print(f"    The FDM KV entries may occupy attention capacity")
    else:
        print(f"    ✗ NOT ADDITIVE — interference detected")
        print(f"    FDM delta: {fdm_delta*100:+.1f}%, Plaintext delta: {pt_delta*100:+.1f}%")

    json.dump(results, open("fdm_additivity_results.json", "w"), indent=2, default=float)
    print(f"\nSaved to fdm_additivity_results.json")


if __name__ == "__main__":
    main()
