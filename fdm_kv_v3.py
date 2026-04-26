"""
FDM-in-Weights v3: Fixed KV cache injection with proper attention mask
"""
import sys, os, re, json, random
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
        fdm_text, fdm_tokens = encoder.encode_memory(mem)
        question = "Report all context values for channels 8-39."
        prompt = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer:"
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
    for name in sorted(per_ch, key=per_ch.get)[:5]:
        print(f"    {name}: {per_ch[name]*100:.1f}%")
    return per_ch, overall

# ============================================================
# Experiment 2: Frozen KV injection (manual generation loop)
# ============================================================

def run_frozen_injection(model, tokenizer, encoder, n_eval, device):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Frozen KV Cache Injection")
    print("=" * 60)
    
    channel_correct = {name: 0 for name in CHANNEL_NAMES[8:]}
    total = 0
    
    for i in tqdm(range(n_eval), desc="Frozen KV"):
        mem = random_memory()
        fdm_text, fdm_tokens = encoder.encode_memory(mem)
        
        # Step 1: Run FDM tokens through model to get KV cache
        fdm_ids = torch.tensor([fdm_tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            fdm_out = model(fdm_ids, use_cache=True)
        fdm_kv = fdm_out.past_key_values
        fdm_seq_len = fdm_ids.shape[1]
        
        # Step 2: Encode the rest of the prompt (no FDM tokens)
        question = "Report all context values for channels 8-39."
        rest_prompt = f"[/MEMORY]\nQuestion: {question}\nAnswer:"
        # Note: we include [/MEMORY] but NOT [MEMORY] or FDM text
        # because the FDM tokens already went through the model above.
        # We need [MEMORY] to be part of the FDM prefix too:
        memory_start = tokenizer.encode("[MEMORY]", add_special_tokens=False)
        
        # Actually: rebuild it properly
        # The full prompt is: [MEMORY] <fdm> [/MEMORY] \nQuestion: ...\nAnswer:
        # We already processed <fdm> tokens. But we need [MEMORY] before them.
        # Let's process [MEMORY] + fdm_tokens together:
        memory_start_ids = tokenizer.encode("[MEMORY]", add_special_tokens=False)
        full_fdm_ids = torch.tensor([memory_start_ids + fdm_tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            fdm_out = model(full_fdm_ids, use_cache=True)
        fdm_kv = fdm_out.past_key_values
        fdm_seq_len = full_fdm_ids.shape[1]
        
        # Step 3: Now process the rest with the KV cache
        rest_ids = tokenizer.encode(rest_prompt, add_special_tokens=False)
        rest_tensor = torch.tensor([rest_ids], dtype=torch.long).to(device)
        
        # Position IDs for the rest tokens (continue from FDM length)
        position_ids = torch.arange(fdm_seq_len, fdm_seq_len + len(rest_ids),
                                     device=device).unsqueeze(0)
        
        # Attention mask: 1s for FDM cache + 1s for new tokens
        attn_mask = torch.ones(1, fdm_seq_len + len(rest_ids), device=device, dtype=torch.long)
        
        # Manual greedy generation loop
        generated_ids = []
        past_kv = fdm_kv
        current_ids = rest_tensor
        current_pos = position_ids
        current_attn = attn_mask
        
        with torch.no_grad():
            # First: process the rest prompt with FDM KV cache
            out = model(current_ids, past_key_values=past_kv,
                       position_ids=current_pos, attention_mask=current_attn,
                       use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            
            total_len = fdm_seq_len + len(rest_ids) + 1
            
            # Continue generating
            for step in range(249):
                pos = torch.tensor([[total_len - 1 + step]], device=device)
                attn = torch.ones(1, total_len + step, device=device, dtype=torch.long)
                
                out = model(next_token, past_key_values=past_kv,
                           position_ids=pos, attention_mask=attn,
                           use_cache=True)
                past_kv = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tok_id = next_token.item()
                generated_ids.append(tok_id)
                
                if tok_id == tokenizer.eos_token_id:
                    break
        
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        scores = score_extra_channels(generated, mem)
        for name, correct in scores.items():
            channel_correct[name] += int(correct)
        total += 1
    
    per_ch = {name: channel_correct[name] / total for name in channel_correct}
    overall = np.mean(list(per_ch.values()))
    print(f"\n  Overall extra-channel accuracy: {overall*100:.1f}%")
    for name in sorted(per_ch, key=per_ch.get)[:5]:
        print(f"    {name}: {per_ch[name]*100:.1f}%")
    return per_ch, overall

# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompterminal/fdm-40ch-nhop-qwen3")
    parser.add_argument("--n_eval", type=int, default=100)
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
    
    if "baseline" in results and "frozen_kv" in results:
        delta = results["frozen_kv"] - results["baseline"]
        print(f"\n  Delta: {delta*100:+.1f}%")
        if abs(delta) < 2:
            print("  → FDM reads identically from KV cache as from context tokens!")
        elif delta < -5:
            print("  → Degradation: model needs more than KV state to read FDM")
    
    json.dump(results, open("fdm_kv_v3_results.json", "w"), indent=2)

if __name__ == "__main__":
    main()
