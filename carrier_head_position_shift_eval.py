"""
carrier_head_position_shift_eval.py

Position-independence evaluation for the weight-resident FDM carrier head on Qwen3.
Tests whether retrieval works when the carrier head's KV output is prepended at
different position ranges in the runtime cache.

Conditions:
  baseline  : carrier head at [0, 512]      (training position)
  shifted   : carrier head at [128, 640]    (mid-substrate-A shift)
  sub_B     : carrier head at [513, 1025]   (substrate B's slot)
"""

import os
import json
import math
import argparse
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from fdm_carrier_head import (
    FDMCarrierHead,
    sample_memory,
    NUM_CHANNELS,
    MEMORY_SCHEMAS,
)

parser = argparse.ArgumentParser()
parser.add_argument('--carrier_dir', type=str, default='carrier_head_16_16')
parser.add_argument('--checkpoint_name', type=str, default='carrier_head.pt')
parser.add_argument('--host_dir', type=str, default='two_block_model')
parser.add_argument('--n_eval', type=int, default=200)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--n_carriers_A', type=int, default=16)
parser.add_argument('--n_kv_positions', type=int, default=513)
parser.add_argument('--out_json', type=str, default='position_shift_results.json')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading host from {args.host_dir} ...")
tokenizer = AutoTokenizer.from_pretrained(args.host_dir)
host = AutoModelForCausalLM.from_pretrained(
    args.host_dir, torch_dtype=torch.float32, device_map=device
)
host.eval()
for p in host.parameters():
    p.requires_grad = False

n_layers = host.config.num_hidden_layers
n_kv_heads = host.config.num_key_value_heads
head_dim = host.config.hidden_size // host.config.num_attention_heads
print(f"Host: layers={n_layers}, kv_heads={n_kv_heads}, head_dim={head_dim}")

print(f"Loading carrier head from {args.carrier_dir}/{args.checkpoint_name} ...")
carrier_head = FDMCarrierHead(
    n_carriers=args.n_carriers_A,
    n_layers=n_layers,
    n_kv_heads=n_kv_heads,
    kv_dim=head_dim,
    n_kv_positions=args.n_kv_positions,
).to(device)

ckpt_path = os.path.join(args.carrier_dir, args.checkpoint_name)
state = torch.load(ckpt_path, map_location=device)
if 'state_dict' in state:
    state = state['state_dict']
carrier_head.load_state_dict(state)
carrier_head.eval()
for p in carrier_head.parameters():
    p.requires_grad = False


def assemble_shifted_cache(carrier_layer_kvs, position_offset, n_layers):
    cache = DynamicCache()
    for l in range(n_layers):
        k, v = carrier_layer_kvs[l]
        if position_offset == 0:
            cache.update(k, v, l)
        else:
            pad_shape = (k.shape[0], k.shape[1], position_offset, k.shape[3])
            k_pad = torch.zeros(pad_shape, device=k.device, dtype=k.dtype)
            v_pad = torch.zeros_like(k_pad)
            k_full = torch.cat([k_pad, k], dim=2)
            v_full = torch.cat([v_pad, v], dim=2)
            cache.update(k_full, v_full, l)
    return cache


def eval_one_sample(memory, position_offset, n_kv_positions, channels_A):
    channel_values = []
    for c_idx, ch in enumerate(channels_A):
        value_string = memory[ch]
        value_idx = MEMORY_SCHEMAS[ch][1].index(value_string)
        channel_values.append((c_idx, value_idx))

    with torch.no_grad():
        layer_kvs = carrier_head(channel_values)

    cache = assemble_shifted_cache(layer_kvs, position_offset, n_layers)

    correct = {}
    for c_idx, ch in enumerate(channels_A):
        question = f"Question: What is the value of channel {ch}? Answer:"
        prompt_ids = tokenizer.encode(question, return_tensors='pt').to(device)

        cache_len = position_offset + n_kv_positions
        question_pos = torch.arange(
            cache_len, cache_len + prompt_ids.shape[1], device=device
        ).unsqueeze(0)

        with torch.no_grad():
            out = host(
                input_ids=prompt_ids,
                past_key_values=cache,
                position_ids=question_pos,
                use_cache=True,
            )

        next_token_id = out.logits[0, -1].argmax().item()
        expected_first_token = tokenizer.encode(
            ' ' + memory[ch], add_special_tokens=False
        )[0]
        correct[ch] = (next_token_id == expected_first_token)

    return correct


print(f"\nRunning evaluation: n={args.n_eval} samples per condition")
random.seed(args.seed)
torch.manual_seed(args.seed)
memories = [sample_memory() for _ in range(args.n_eval)]
channels_A = list(range(args.n_carriers_A))

conditions = [
    ('baseline', 0,   'Carrier head at substrate A slot [0, 512]'),
    ('shifted',  128, 'Carrier head at offset [128, 640]'),
    ('sub_B',    513, 'Carrier head at substrate B slot [513, 1025]'),
]

results = {}
for cond_name, offset, description in conditions:
    print(f"\n=== {cond_name}: {description} ===")
    slot_correct = 0
    slot_total = 0

    for i, mem in enumerate(memories):
        if (i + 1) % 50 == 0:
            print(f"  sample {i+1}/{args.n_eval}: slot acc so far = "
                  f"{slot_correct/max(slot_total,1)*100:.1f}%")
        try:
            per_channel = eval_one_sample(mem, offset, args.n_kv_positions, channels_A)
            slot_correct += sum(1 for v in per_channel.values() if v)
            slot_total += len(per_channel)
        except Exception as e:
            print(f"  sample {i+1}: error {e}")
            continue

    slot_acc = slot_correct / max(slot_total, 1)
    print(f"\n  {cond_name} slot accuracy: {slot_acc*100:.2f}% ({slot_correct}/{slot_total})")
    results[cond_name] = {
        'description': description,
        'offset': offset,
        'slot_correct': slot_correct,
        'slot_total': slot_total,
        'slot_accuracy': slot_acc,
    }

with open(args.out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {args.out_json}")

print("\n" + "="*60)
print("Position-independence summary (slot accuracy on substrate A)")
print("="*60)
for cond_name, _, description in conditions:
    r = results[cond_name]
    print(f"  {cond_name:10s} (offset {r['offset']:4d}): "
          f"{r['slot_accuracy']*100:5.2f}%  ({r['slot_correct']}/{r['slot_total']})")
