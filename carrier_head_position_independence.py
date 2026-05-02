#!/usr/bin/env python3
"""
carrier_head_position_independence.py

Position-independence evaluation for the weight-resident FDM carrier head
on Qwen3-0.6B. Reuses fdm_carrier_head's evaluate_carrier_head pipeline
end-to-end, only varying how the carrier head's KV cache is positioned
relative to the in-context block.

Three conditions (all paired with substrate B in-context content):
    baseline : carrier KV at positions [0, P-1], in-context at [P, P+L-1]
    shifted  : carrier KV at positions [128, 128+P-1], in-context at [128+P, 128+P+L-1]
    swapped  : in-context at [0, L-1], carrier KV at [L, L+P-1]

Where P = carrier_head.n_kv_positions = 513.

baseline reproduces the existing reeval pipeline at 98.17%/97.84%/57.75%
(when using seeds 999 and 2024). The shifted and swapped conditions
test whether the host's frequency-decomposition attention reads the
weight-resident substrate position-independently.
"""

import os, sys, json, random, math, re
import torch
from tqdm import tqdm

WORK_DIR = "/workspace/FDM_IN_WEIGHTS"
CARRIER_DIR = os.path.join(WORK_DIR, "carrier_head_16_16")
HOST_MODEL = os.path.join(WORK_DIR, "two_block_model")
N_EVAL = 200
N_CARRIER_CHANNELS = 16

sys.path.insert(0, WORK_DIR)
sys.path.insert(0, '/root/FDM_IN_WEIGHTS')

# Import training script as a module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fdm_ch", os.path.join(WORK_DIR, "fdm_carrier_head.py")
)
fdm_ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdm_ch)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pull symbols we need from the training module
MEMORY_SCHEMAS    = fdm_ch.MEMORY_SCHEMAS
NUM_CHANNELS      = fdm_ch.NUM_CHANNELS
CHANNEL_NAMES     = fdm_ch.CHANNEL_NAMES
ALL_QUERY_CHANNELS = fdm_ch.ALL_QUERY_CHANNELS
random_memory     = fdm_ch.random_memory


def layer_kvs_to_cache(layer_kvs, n_layers):
    """Same as fdm_ch.layer_kvs_to_cache."""
    cache = DynamicCache()
    for l in range(n_layers):
        k, v = layer_kvs[l]
        cache.update(k, v, l)
    return cache


def layer_kvs_to_padded_cache(layer_kvs, n_layers, leading_pad):
    """Build a DynamicCache with `leading_pad` zero-padded positions before
    the carrier head's KV. Result has total length leading_pad + n_kv_pos.

    The host applies RoPE based on absolute cache position, so prepending
    `leading_pad` zero positions shifts the carrier head's effective position
    range from [0, n_kv_pos-1] to [leading_pad, leading_pad+n_kv_pos-1].
    """
    cache = DynamicCache()
    for l in range(n_layers):
        k, v = layer_kvs[l]
        if leading_pad == 0:
            cache.update(k, v, l)
        else:
            pad_shape = (k.shape[0], k.shape[1], leading_pad, k.shape[3])
            k_pad = torch.zeros(pad_shape, device=k.device, dtype=k.dtype)
            v_pad = torch.zeros_like(k_pad)
            k_full = torch.cat([k_pad, k], dim=2)
            v_full = torch.cat([v_pad, v], dim=2)
            cache.update(k_full, v_full, l)
    return cache


def generate_with_position_config(
    model, tokenizer, carrier_head, memory, encoder, device,
    carrier_channels, context_channels, position_config, max_new_tokens=350,
):
    """Drop-in replacement for fdm_ch.generate_with_carrier_head with
    explicit control of where the carrier-head KV and in-context block
    land in the position sequence.

    position_config in {"baseline", "shifted", "swapped"}:
      baseline : carrier@[0,P-1],          in-ctx@[P,P+L-1]
      shifted  : carrier@[128,128+P-1],    in-ctx@[128+P,128+P+L-1]
                 (with 128 leading zero-padded cache positions)
      swapped  : in-ctx@[0,L-1],           carrier@[L,L+P-1]
                 (in-context block processed first, carrier KV appended after)
    """
    P = carrier_head.n_kv_positions  # 513

    # -------- Build carrier head KVs --------
    channel_values = []
    for c_idx, ch in enumerate(carrier_channels):
        val = memory[ch]
        val_list = MEMORY_SCHEMAS[ch][1]
        val_idx = val_list.index(val) if val in val_list else 0
        channel_values.append((c_idx, val_idx))
    layer_kvs = carrier_head(channel_values)

    # -------- Build in-context FDM tokens --------
    context_mem = {ch: memory[ch] for ch in range(NUM_CHANNELS)}
    fdm_text, _ = encoder.encode_memory(context_mem)
    memory_part = f"[MEMORY]BLOCK_B {fdm_text}[/MEMORY]"

    ch_names = [CHANNEL_NAMES[k] for k in ALL_QUERY_CHANNELS]
    question = f"Report values for: {', '.join(ch_names)}."
    prompt = f"{memory_part}\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    L = input_ids.shape[1]

    # -------- Assemble cache + position_ids per condition --------
    if position_config == "baseline":
        # carrier @ [0, P-1], in-context @ [P, P+L-1]
        past_kv = layer_kvs_to_cache(layer_kvs, carrier_head.n_layers)
        position_ids = torch.arange(P, P + L, device=device).unsqueeze(0)

    elif position_config == "shifted":
        # carrier @ [128, 128+P-1], in-context @ [128+P, 128+P+L-1]
        # 128 zero-padded positions land before the carrier head's KV
        SHIFT = 128
        past_kv = layer_kvs_to_padded_cache(layer_kvs, carrier_head.n_layers, SHIFT)
        # in-context positions follow the (padded) cache
        position_ids = torch.arange(SHIFT + P, SHIFT + P + L, device=device).unsqueeze(0)

    elif position_config == "swapped":
        # in-context @ [0, L-1] processed first, then carrier @ [L, L+P-1]
        # We process the in-context block as plain input (no past KV), then
        # use its resulting cache as the leading cache, then prepend the
        # carrier head's output... but actually the cleanest swap is to
        # run the in-context block BEFORE injecting the carrier head.
        # However that doesn't match how RoPE applies. Simpler approach:
        # build a cache that has L zero-padded entries (the positions the
        # in-context block would occupy at processing time) followed by
        # the carrier head's KV. Then the in-context block at input time
        # will overwrite the L zero-padded positions during forward pass.
        # This is structurally equivalent to "swapped" but requires the
        # forward pass to handle it.
        #
        # Actually, the cleanest interpretation: we want carrier KV at
        # positions [L, L+P-1] in the final cache, and in-context tokens
        # at positions [0, L-1] in the final cache. Since the model
        # processes in-context tokens via input_ids (not as past_kv),
        # we can achieve this by:
        #   (a) NOT supplying past_kv before the in-context tokens
        #   (b) supplying carrier KV with leading_pad = L so its positions
        #       become [L, L+P-1]
        #   (c) BUT this would mean the in-context block sees no carrier
        #       KV during its forward pass, which is wrong.
        #
        # The mechanically correct swap: prepend L zero-padded slots in
        # the cache to "reserve" positions [0, L-1] for in-context content,
        # carrier head goes at [L, L+P-1]. The in-context tokens at
        # positions [0, L-1] then attend to:
        #   - their own positions (causal mask within input_ids)
        #   - carrier head's KV at [L, L+P-1] (via past_kv attention)
        # which is exactly what happens in baseline but with positions
        # swapped. Position_ids for input_ids becomes [0, L-1].
        past_kv = layer_kvs_to_padded_cache(layer_kvs, carrier_head.n_layers, L)
        # in-context block processed at positions [0, L-1]
        position_ids = torch.arange(0, L, device=device).unsqueeze(0)

    else:
        raise ValueError(f"Unknown position_config: {position_config}")

    # -------- Generate (same loop as original) --------
    generated = input_ids
    cur_past = past_kv
    cur_pos = position_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if cur_past is not None and generated.shape[1] > input_ids.shape[1]:
                last_tok = generated[:, -1:]
                last_pos = cur_pos[:, -1:] + 1
                outputs = model(
                    input_ids=last_tok, position_ids=last_pos,
                    past_key_values=cur_past, use_cache=True
                )
                cur_pos = last_pos
            else:
                kwargs = dict(input_ids=generated, position_ids=cur_pos, use_cache=True)
                if cur_past is not None:
                    kwargs['past_key_values'] = cur_past
                outputs = model(**kwargs)
            cur_past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)


def evaluate_position_config(
    model, tokenizer, carrier_head, encoder, device,
    carrier_channels, context_channels, position_config, n_eval=200,
):
    """Identical to fdm_ch.evaluate_carrier_head but routes through
    generate_with_position_config."""
    carrier_head.eval()
    car_correct = 0; car_total = 0
    ctx_correct = 0; ctx_total = 0
    sample_car_all = 0; sample_ctx_all = 0; sample_all_all = 0
    n_samples = 0

    for _ in tqdm(range(n_eval), desc=f"Eval [{position_config}]", leave=False):
        mem = random_memory()
        answer = generate_with_position_config(
            model, tokenizer, carrier_head, mem, encoder, device,
            carrier_channels, context_channels, position_config,
        )

        car_all = True
        ctx_all = True
        for k in carrier_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                car_correct += 1
            else:
                car_all = False
            car_total += 1
        for k in context_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                ctx_correct += 1
            else:
                ctx_all = False
            ctx_total += 1

        if car_all: sample_car_all += 1
        if ctx_all: sample_ctx_all += 1
        if car_all and ctx_all: sample_all_all += 1
        n_samples += 1

    return {
        "position_config": position_config,
        "n_carrier_channels": len(carrier_channels),
        "n_context_channels": len(context_channels),
        "carrier_slot": car_correct / car_total if car_total else 0.0,
        "context_slot": ctx_correct / ctx_total if ctx_total else 0.0,
        "carrier_joint": sample_car_all / n_samples if n_samples else 0.0,
        "context_joint": sample_ctx_all / n_samples if n_samples else 0.0,
        "all_joint": sample_all_all / n_samples if n_samples else 0.0,
        "n_samples": n_samples,
        "carrier_correct": car_correct,
        "carrier_total": car_total,
        "context_correct": ctx_correct,
        "context_total": ctx_total,
    }


def wilson_hw(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n
    return (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


def main():
    print("=" * 70)
    print("CARRIER HEAD POSITION-INDEPENDENCE EVAL (n=200 per condition)")
    print("=" * 70)

    # Load host
    print(f"\nLoading host model from {HOST_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(HOST_MODEL, trust_remote_code=True)
    host_model = AutoModelForCausalLM.from_pretrained(
        HOST_MODEL, dtype=torch.float32, trust_remote_code=True
    ).to(device)
    host_model.eval()
    for p in host_model.parameters():
        p.requires_grad = False

    n_layers, n_kv_heads, head_dim = fdm_ch.detect_kv_shape(
        host_model, host_model.config, device
    )
    print(f"  Detected: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    # Build carrier head
    print(f"\nConstructing CarrierAttentionHead (n_carriers={N_CARRIER_CHANNELS})...")
    carrier_head = fdm_ch.CarrierAttentionHead(
        n_layers=n_layers, kv_dim=head_dim, n_kv_heads=n_kv_heads,
        n_carriers=N_CARRIER_CHANNELS, n_kv_positions=513,
        sample_rate=100.0, n_values=64,
    ).to(device)

    state_path = os.path.join(CARRIER_DIR, "carrier_head.pt")
    print(f"  Loading state from {state_path}")
    state_dict = torch.load(state_path, map_location=device)
    carrier_head.load_state_dict(state_dict, strict=True)
    carrier_head.eval()

    encoder = fdm_ch.make_encoder(tokenizer)

    carrier_channels = list(range(8, 8 + N_CARRIER_CHANNELS))
    context_channels = list(range(8 + N_CARRIER_CHANNELS, 40))

    conditions = ["baseline", "shifted", "swapped"]
    seeds = [999, 2024]

    all_results = {}
    for cond in conditions:
        print(f"\n{'=' * 70}\nCONDITION: {cond}\n{'=' * 70}")
        cond_results = []
        for seed in seeds:
            print(f"\n--- {cond} seed={seed} ---")
            random.seed(seed)
            torch.manual_seed(seed)
            r = evaluate_position_config(
                host_model, tokenizer, carrier_head, encoder, device,
                carrier_channels, context_channels, cond, n_eval=N_EVAL,
            )
            print(f"  car_slot={r['carrier_slot']*100:.1f}% "
                  f"ctx_slot={r['context_slot']*100:.1f}% "
                  f"all_joint={r['all_joint']*100:.1f}%")
            cond_results.append(r)

        # Combined across both seeds
        n_total = sum(r['n_samples'] for r in cond_results)
        car_combined = sum(r['carrier_correct'] for r in cond_results) / \
                       sum(r['carrier_total'] for r in cond_results)
        ctx_combined = sum(r['context_correct'] for r in cond_results) / \
                       sum(r['context_total'] for r in cond_results)
        joint_combined = sum(r['all_joint'] * r['n_samples'] for r in cond_results) / n_total
        joint_correct = round(joint_combined * n_total)
        joint_hw = wilson_hw(joint_correct, n_total)

        print(f"\n  COMBINED ({cond}, n={n_total}): "
              f"car_slot={car_combined*100:.1f}% "
              f"ctx_slot={ctx_combined*100:.1f}% "
              f"all_joint={joint_combined*100:.1f}% (+/-{joint_hw*100:.1f}pp)")

        all_results[cond] = {
            "rounds": cond_results,
            "combined": {
                "n_samples": n_total,
                "carrier_slot": car_combined,
                "context_slot": ctx_combined,
                "all_joint": joint_combined,
                "joint_ci_pp_wilson": joint_hw * 100,
            },
        }

    # Save
    out_path = os.path.join(CARRIER_DIR, "position_independence_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Final summary table
    print("\n" + "=" * 70)
    print(f"{'POSITION INDEPENDENCE SUMMARY':^70}")
    print("=" * 70)
    print(f"{'Condition':<12} {'Car slot':>10} {'Ctx slot':>10} {'Joint':>10} {'95% CI':>10}")
    print("-" * 70)
    for cond in conditions:
        c = all_results[cond]['combined']
        ci = c['joint_ci_pp_wilson']
        print(f"{cond:<12} "
              f"{c['carrier_slot']*100:>9.1f}% "
              f"{c['context_slot']*100:>9.1f}% "
              f"{c['all_joint']*100:>9.1f}% "
              f"  +/-{ci:>4.1f}pp")


if __name__ == "__main__":
    main()
