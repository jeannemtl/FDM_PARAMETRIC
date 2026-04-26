"""
FDM-in-Weights v2: Uses REAL encoder + correct DynamicCache API
"""
import sys, os, re, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, '/root/FDM_IN_WEIGHTS')
from nhop_source import TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS

CHANNEL_NAMES = [MEMORY_SCHEMAS[i][0] for i in range(NUM_CHANNELS)]

# ============================================================
# Build prompts using REAL encoder
# ============================================================

def make_encoder(tokenizer):
    return TurboFDMSignalEncoder(
        vocab_size=151936, tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

def random_memory():
    return {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}

def build_prompt_with_fdm(memory, encoder, tokenizer):
    fdm_text, fdm_tokens = encoder.encode_memory(memory)
    question = "Report all context values for channels 8-39."
    extra_parts = [f"{CHANNEL_NAMES[k]}={memory[k]}" for k in range(8, NUM_CHANNELS)]
    expected = "Context: " + ", ".join(extra_parts) + "."
    prompt = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
    return prompt, fdm_tokens, expected

def build_prompt_without_fdm(tokenizer):
    question = "Report all context values for channels 8-39."
    return f"[MEMORY][/MEMORY]\nQuestion: {question}\nAnswer:"

def score_extra_channels(text, memory):
    results = {}
    for k in range(8, NUM_CHANNELS):
        name = CHANNEL_NAMES[k]
        expected = memory[k]
        pattern = rf"\b{re.escape(name)}={re.escape(expected)}\b"
        results[name] = bool(re.search(pattern, text))
    return results

# ============================================================
# Experiment 1: Baseline
# ============================================================

def run_baseline(model, tokenizer, encoder, n_eval, device):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline (FDM in context tokens)")
    print("=" * 60)
    
    channel_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0
    
    for i in tqdm(range(n_eval), desc="Baseline"):
        mem = random_memory()
        prompt, _, expected = build_prompt_with_fdm(mem, encoder, tokenizer)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=250, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        scores = score_extra_channels(generated, mem)
        for name, correct in scores.items():
            channel_correct[name] += int(correct)
        total += 1
    
    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    print(f"\n  Overall extra-channel accuracy: {overall*100:.1f}%")
    worst = sorted(per_ch, key=per_ch.get)[:5]
    for name in worst:
        print(f"    {name}: {per_ch[name]*100:.1f}%")
    return per_ch, overall

# ============================================================
# Experiment 2: Frozen KV Cache Injection
# ============================================================

def compute_fdm_kv(model, fdm_tokens, device):
    """Run model on FDM tokens only, return DynamicCache."""
    ids = torch.tensor([fdm_tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    return out.past_key_values  # DynamicCache

def run_frozen_injection(model, tokenizer, encoder, n_eval, device):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Frozen KV Cache Injection")
    print("=" * 60)
    
    channel_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0
    
    for i in tqdm(range(n_eval), desc="Frozen KV"):
        mem = random_memory()
        _, fdm_tokens, expected = build_prompt_with_fdm(mem, encoder, tokenizer)
        
        # Step 1: get KV cache from FDM tokens
        fdm_kv = compute_fdm_kv(model, fdm_tokens, device)
        fdm_seq_len = fdm_kv.get_seq_length()
        
        # Step 2: prompt WITHOUT FDM tokens
        prompt_no_fdm = build_prompt_without_fdm(tokenizer)
        prompt_ids = tokenizer.encode(prompt_no_fdm, return_tensors='pt').to(device)
        
        # Step 3: position IDs offset by FDM length
        seq_len = prompt_ids.shape[1]
        position_ids = torch.arange(fdm_seq_len, fdm_seq_len + seq_len,
                                     device=device).unsqueeze(0)
        
        # Step 4: generate with prepended KV cache
        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                past_key_values=fdm_kv,
                position_ids=position_ids,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][seq_len:], skip_special_tokens=True)
        scores = score_extra_channels(generated, mem)
        for name, correct in scores.items():
            channel_correct[name] += int(correct)
        total += 1
    
    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    print(f"\n  Overall extra-channel accuracy: {overall*100:.1f}%")
    worst = sorted(per_ch, key=per_ch.get)[:5]
    for name in worst:
        print(f"    {name}: {per_ch[name]*100:.1f}%")
    return per_ch, overall

# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16
    ).to(device)
    model.eval()
    print(f"  Loaded. Vocab: {tokenizer.vocab_size}, Layers: {model.config.num_hidden_layers}")
    
    encoder = make_encoder(tokenizer)
    results = {}
    
    if not args.skip_baseline:
        pc, ov = run_baseline(model, tokenizer, encoder, args.n_eval, device)
        results["baseline"] = ov
    
    pc, ov = run_frozen_injection(model, tokenizer, encoder, args.n_eval, device)
    results["frozen_kv"] = ov
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:20s}: {v*100:.1f}%")
    
    print("\nIf frozen_kv matches baseline → FDM reads from KV cache = reads from context")
    print("If frozen_kv degrades → model needs raw embeddings, not just KV state")
    
    json.dump(results, open("fdm_kv_v2_results.json", "w"), indent=2)

if __name__ == "__main__":
    main()
