"""
N-Hop Compositional Reasoning — Model-Agnostic Training + Evaluation

Trains ONE additional stage on top of any FDM-trained model with question
templates that have explicit, countable hop structure (Greenblatt-style).

Each hop is a genuine compositional step where the output of one read
feeds into the next decision branch — NOT parallel reads.

Hop definitions:
  1-hop: Read 1 channel, report value
  2-hop: Read 2 channels, compose (output depends on BOTH values)
  3-hop: Read 3 channels, chained conditional
  4-hop: Read 4 channels, nested conditionals with override
  5-hop: Read 5 channels, full conditional chain

Works with: GPT-2 Medium, Qwen3-0.6B-Base, LFM2.5-1.2B-Base

Usage:
  python nhop_training_multi.py generate gpt2
  python nhop_training_multi.py generate qwen3
  python nhop_training_multi.py generate lfm2

  python nhop_training_multi.py train gpt2
  python nhop_training_multi.py train qwen3
  python nhop_training_multi.py train lfm2

  python nhop_training_multi.py eval gpt2 [model_path]
  python nhop_training_multi.py eval qwen3 [model_path]
  python nhop_training_multi.py eval lfm2 [model_path]
"""

import json, random, os, sys, re
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# MODEL CONFIGS — select by name
# ============================================================

MODEL_CONFIGS = {
    "gpt2": {
        "model_id": "openai-community/gpt2-medium",
        "vocab_size": 50257,
        "dtype": torch.float32,
        "batch_size": 4,
        "grad_accum": 1,
        "trust_remote_code": False,
        "model_class": "gpt2",  # use GPT2LMHeadModel
        "base_model_dir": "fdm_40ch_turbo_v3_model_final",
        "nhop_data_dir": "nhop_training_data_gpt2",
        "nhop_model_dir": "nhop_model_gpt2_final",
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "vocab_size": 151936,
        "dtype": torch.bfloat16,
        "batch_size": 4,
        "grad_accum": 1,
        "trust_remote_code": True,
        "model_class": "auto",
        "base_model_dir": "fdm_40ch_turbo_v3_qwen3_512tok_model",
        "nhop_data_dir": "nhop_training_data_qwen3",
        "nhop_model_dir": "nhop_model_qwen3_final",
    },
    "hermes3": {
        "model_id": "NousResearch/Hermes-3-Llama-3.2-3B",
        "vocab_size": 128256,
        "dtype": torch.bfloat16,
        "batch_size": 2,
        "grad_accum": 1,
        "trust_remote_code": True,
        "model_class": "auto",
        "base_model_dir": "fdm_40ch_turbo_v3_hermes3_model_final",
        "nhop_data_dir": "nhop_training_data_hermes3",
        "nhop_model_dir": "nhop_model_hermes3_final",
    },
    "lfm2": {
        "model_id": "LiquidAI/LFM2.5-1.2B-Base",
        "vocab_size": 65536,
        "dtype": torch.bfloat16,
        "batch_size": 2,
        "grad_accum": 2,
        "trust_remote_code": True,
        "model_class": "auto",
        "base_model_dir": "fdm_40ch_turbo_v3_lfm2_model_final",
        "nhop_data_dir": "nhop_training_data_lfm2",
        "nhop_model_dir": "nhop_model_lfm2_final",
    },
}


# ============================================================
# Channel definitions (shared across all models)
# ============================================================

MEMORY_SCHEMAS = {
    0: ("SECRET",   ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT",    ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS",   ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP",   ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE",     ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META",     ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",     ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",   ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",    ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",     ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",    ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",   ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",    ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",  ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",   ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",  ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",  ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",  ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",   ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",     ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",  ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",    ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION", ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",  ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",     ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE", ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",    ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",    ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",   ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",   ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",   ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",    ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",     ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2", ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE", ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",    ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}

NUM_CHANNELS = 40


# ============================================================
# FDM Encoder — model-agnostic via vocab_size parameter
# ============================================================

class TurboFDMSignalEncoder:
    def __init__(self, vocab_size, tokenizer, num_tokens_per_encoder=256,
                 sample_rate=100.0, a_high=1.0, a_low=0.0,
                 num_levels=64, seed=42):
        self.num_tokens_per_encoder = num_tokens_per_encoder
        self.total_tokens = num_tokens_per_encoder * 2
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        self.num_channels = NUM_CHANNELS
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(NUM_CHANNELS)]
        self.tokenizer = tokenizer

        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(num_tokens_per_encoder, S)

        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(vocab_size, size=num_levels, replace=False)

    def _generate_s_random_interleaver(self, length, S):
        import random as rnd
        rnd.seed(42)
        interleaver = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                valid = True
                for k in range(max(0, i - S + 1), i):
                    if abs(interleaver[j] - interleaver[k]) < S:
                        valid = False
                        break
                if valid:
                    interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                    break
        return interleaver

    def value_to_bits(self, channel_id, value):
        _, values = MEMORY_SCHEMAS[channel_id]
        idx = values.index(value)
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits

    def encode_memory(self, memory):
        all_bits = {}
        max_bits = 0
        for ch in range(self.num_channels):
            bits, nb = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = bits
            max_bits = max(max_bits, nb)
        for ch in range(self.num_channels):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        num_message_bits = max_bits
        t = np.arange(self.num_tokens_per_encoder) / self.sample_rate
        samples_per_bit = self.num_tokens_per_encoder // num_message_bits

        composite = np.zeros(self.num_tokens_per_encoder)
        for ch in range(self.num_channels):
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens_per_encoder)
                amp = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amp * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[start:end]
                )

        sig_min, sig_max = composite.min(), composite.max()
        sig_range = sig_max - sig_min + 1e-10

        norm1 = (composite - sig_min) / sig_range
        q1 = np.floor(norm1 * (self.num_levels - 1) + 0.5).astype(int)
        q1 = np.clip(q1, 0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]

        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int)
        q2 = np.clip(q2, 0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]

        all_tokens = tokens1 + tokens2
        fdm_text = self.tokenizer.decode(all_tokens)
        return fdm_text, all_tokens


# ============================================================
# HOP-CONTROLLED QUESTION TEMPLATES
# ============================================================

# ── 1-HOP ──

def hop1_status(m):
    return f"STATUS_REPORT: Status is {m[3]}."

def hop1_threat(m):
    return f"THREAT_REPORT: Threat level is {m[16]}."

def hop1_rule(m):
    return f"RULE_REPORT: Current rule is {m[6]}."

def hop1_location(m):
    return f"LOCATION_REPORT: Location is {m[1]}."


# ── 2-HOP ──

def hop2_rule_status(m):
    rule, status = m[6], m[3]
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
        elif status == "CLEAR":
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
        else:
            return f"RULE_STATUS: {rule} with {status} status. Decision: WAIT."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED":
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED WITH CAUTION."
        else:
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
    elif rule == "BALANCED":
        if status == "COMPROMISED":
            return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
        else:
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
    elif rule == "CAUTIOUS":
        if status == "CLEAR":
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
        else:
            return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
    return f"RULE_STATUS: {rule} with {status} status."

def hop2_threat_terrain(m):
    threat, terrain = m[16], m[18]
    if threat in ("HIGH", "CRITICAL"):
        return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend EVACUATION."
    elif threat == "MEDIUM" and terrain in ("URBAN", "COASTAL"):
        return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend FORTIFY."
    else:
        return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend CONTINUE."

def hop2_agent_location(m):
    agent, location = m[2], m[1]
    return f"AGENT_LOCATION: {agent} is deployed in {location}. Cover status depends on location."


# ── 3-HOP ──

def hop3_rule_status_backup(m):
    rule, status, backup = m[6], m[3], m[5]
    if rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"CHAIN_3HOP: {rule} requires CLEAR+AVAILABLE. Have {status}/{backup}. PROCEED."
        else:
            return f"CHAIN_3HOP: {rule} requires CLEAR+AVAILABLE. Have {status}/{backup}. ABORT."
    elif rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"CHAIN_3HOP: {rule} says {status} overrides all. ABORT. Backup {backup} irrelevant."
        elif backup == "UNAVAILABLE":
            return f"CHAIN_3HOP: {rule} with {status} status but backup {backup}. HOLD."
        else:
            return f"CHAIN_3HOP: {rule} with {status} and backup {backup}. PROCEED."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            return f"CHAIN_3HOP: {rule} but {status} with no backup. Even mission says ABORT."
        else:
            return f"CHAIN_3HOP: {rule} overrides {status}. Backup {backup}. PROCEED."
    elif rule == "BALANCED":
        score = 0
        if status == "CLEAR": score += 1
        if backup == "AVAILABLE": score += 1
        if score >= 2:
            return f"CHAIN_3HOP: {rule} scores {score}/2. {status}/{backup}. PROCEED."
        elif score == 1:
            return f"CHAIN_3HOP: {rule} scores {score}/2. {status}/{backup}. CAUTION."
        else:
            return f"CHAIN_3HOP: {rule} scores {score}/2. {status}/{backup}. ABORT."
    return f"CHAIN_3HOP: {rule}, {status}, {backup}."

def hop3_threat_cover_signal(m):
    threat, cover, signal = m[16], m[14], m[32]
    if threat in ("HIGH", "CRITICAL") and cover == "NONE":
        return f"THREAT_CHAIN: {threat} threat with {cover} cover. Signal {signal}. CRITICAL EXPOSURE."
    elif signal in ("JAMMED", "LOST"):
        return f"THREAT_CHAIN: {threat} threat, {cover} cover, but signal {signal}. COMMS COMPROMISED."
    else:
        return f"THREAT_CHAIN: {threat} threat, {cover} cover, signal {signal}. MANAGEABLE."


# ── 4-HOP ──

def hop4_meta_rule_status_priority(m):
    meta, rule, status, priority = m[7], m[6], m[3], m[4]
    if meta == "EMERGENCY":
        return f"OVERRIDE_4HOP: {meta} active. Skip {rule}/{status}/{priority}. PROCEED IMMEDIATELY."
    if meta == "LOCKDOWN":
        return f"OVERRIDE_4HOP: {meta} active. Skip {rule}/{status}/{priority}. ABORT ALL."
    if meta == "OVERRIDE_STATUS":
        if priority == "HIGH":
            return f"OVERRIDE_4HOP: {meta} ignores {status}. {rule} with {priority} priority. PROCEED."
        else:
            return f"OVERRIDE_4HOP: {meta} ignores {status}. {rule} with {priority} priority. CAUTION."
    # Standard: rule + status + priority
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        return f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. Safety overrides {priority}. ABORT."
    elif priority == "HIGH":
        return f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. High priority pushes PROCEED."
    else:
        return f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. Standard assessment. HOLD."

def hop4_threat_terrain_weather_asset(m):
    threat, terrain, weather, asset = m[16], m[18], m[17], m[12]
    if threat in ("HIGH", "CRITICAL") and weather == "STORM":
        return f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. Storm + high threat. GROUND {asset}."
    elif asset in ("AIRCRAFT", "DRONE") and weather == "FOG":
        return f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. Fog grounds air assets. DELAY."
    elif terrain == "MOUNTAIN" and asset == "BOAT":
        return f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. Terrain mismatch. REASSIGN."
    else:
        return f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. Conditions acceptable. DEPLOY."


# ── 5-HOP ──

def hop5_meta_rule_status_backup_threat(m):
    meta, rule, status, backup, threat = m[7], m[6], m[3], m[5], m[16]
    if meta == "EMERGENCY":
        return f"FULL_5HOP: {meta} overrides all. Threat {threat} noted. PROCEED."
    if meta == "LOCKDOWN":
        return f"FULL_5HOP: {meta} overrides all. ABORT regardless of {threat}."
    # Step 2: rule + status
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        if threat in ("HIGH", "CRITICAL"):
            return f"FULL_5HOP: {rule}+{status}+{threat}. Escalate to EMERGENCY EXTRACT."
        else:
            return f"FULL_5HOP: {rule}+{status}. Threat only {threat}. Standard ABORT."
    # Step 3: backup contingency
    if backup == "UNAVAILABLE" and threat in ("HIGH", "CRITICAL"):
        return f"FULL_5HOP: {rule}/{status}, no backup, {threat} threat. ABORT with EXTRACT REQUEST."
    elif backup == "AVAILABLE":
        return f"FULL_5HOP: {rule}/{status}, backup ready, {threat} threat. PROCEED with backup."
    else:
        return f"FULL_5HOP: {rule}/{status}/{backup}/{threat}. Assess and HOLD."

def hop5_weather_vis_window_terrain_extract(m):
    weather, vis, window, terrain, extract = m[17], m[28], m[13], m[18], m[19]
    if weather == "STORM" and vis == "ZERO":
        if extract == "READY":
            return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Extract {extract}. EVACUATE."
        else:
            return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Extract {extract}. SHELTER."
    elif window == "NIGHT" and vis == "REDUCED":
        return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Limited ops. Extract {extract}. HOLD."
    elif terrain == "MOUNTAIN" and weather in ("STORM", "FOG"):
        return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Mountain hazard. Extract {extract}. DESCEND."
    else:
        return f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. Conditions OK. Extract {extract}. CONTINUE."


NHOP_TEMPLATES = {
    1: [
        {"question": "Report current status.", "fn": hop1_status, "channels": [3]},
        {"question": "Report threat level.", "fn": hop1_threat, "channels": [16]},
        {"question": "Report active rule.", "fn": hop1_rule, "channels": [6]},
        {"question": "Report current location.", "fn": hop1_location, "channels": [1]},
    ],
    2: [
        {"question": "Evaluate status under current rule.", "fn": hop2_rule_status, "channels": [6, 3]},
        {"question": "Assess threat in current terrain.", "fn": hop2_threat_terrain, "channels": [16, 18]},
        {"question": "Report agent deployment location.", "fn": hop2_agent_location, "channels": [2, 1]},
    ],
    3: [
        {"question": "Should we proceed given rule, status, and backup?",
         "fn": hop3_rule_status_backup, "channels": [6, 3, 5]},
        {"question": "Evaluate threat exposure with cover and signal status.",
         "fn": hop3_threat_cover_signal, "channels": [16, 14, 32]},
    ],
    4: [
        {"question": "Full override check: meta, rule, status, priority.",
         "fn": hop4_meta_rule_status_priority, "channels": [7, 6, 3, 4]},
        {"question": "Tactical assessment: threat, terrain, weather, asset.",
         "fn": hop4_threat_terrain_weather_asset, "channels": [16, 18, 17, 12]},
    ],
    5: [
        {"question": "Complete chain: meta override, rule, status, backup, threat.",
         "fn": hop5_meta_rule_status_backup_threat, "channels": [7, 6, 3, 5, 16]},
        {"question": "Environmental chain: weather, visibility, window, terrain, extract.",
         "fn": hop5_weather_vis_window_terrain_extract, "channels": [17, 28, 13, 18, 19]},
    ],
}


# ============================================================
# Dataset
# ============================================================

class NHopDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        target_text = f" {s['answer']}"
        full_text = input_text + target_text

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()

        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================
# Data generation
# ============================================================

def generate_nhop_data(model_name, samples_per_hop=20000):
    cfg = MODEL_CONFIGS[model_name]
    output_dir = cfg["nhop_data_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], trust_remote_code=cfg["trust_remote_code"]
    )

    encoder = TurboFDMSignalEncoder(
        vocab_size=cfg["vocab_size"], tokenizer=tokenizer,
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    random.seed(77777)
    np.random.seed(77777)

    all_samples = []

    for nhops, templates in sorted(NHOP_TEMPLATES.items()):
        print(f"Generating {samples_per_hop} {model_name} samples for {nhops}-hop...")

        for _ in tqdm(range(samples_per_hop)):
            memory = {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}
            fdm_text, token_ids = encoder.encode_memory(memory)

            template = random.choice(templates)
            base_answer = template["fn"](memory)

            extra = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(8, NUM_CHANNELS)}
            extra_parts = [f"{k}={v}" for k, v in extra.items()]
            answer = f"{base_answer} Context: {', '.join(extra_parts)}."

            all_samples.append({
                "fdm_text": fdm_text,
                "question": template["question"],
                "answer": answer,
                "nhops": nhops,
                "channels": template["channels"],
                "memory": {str(ch): memory[ch] for ch in range(NUM_CHANNELS)},
            })

    random.shuffle(all_samples)
    n = len(all_samples)
    te, ve = int(0.85 * n), int(0.95 * n)
    splits = {"train": all_samples[:te], "val": all_samples[te:ve], "test": all_samples[ve:]}

    for name, data in splits.items():
        path = os.path.join(output_dir, f"nhop_{name}.jsonl")
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(data)} samples -> {path}")
    print(f"Total: {n} samples for {model_name}")


# ============================================================
# Training
# ============================================================

def train(model_name, epochs=10, lr=1e-5):
    cfg = MODEL_CONFIGS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = cfg["base_model_dir"]
    data_dir = cfg["nhop_data_dir"]
    output_path = cfg["nhop_model_dir"]

    print(f"Model: {model_name} | Base: {base_model} | Data: {data_dir}")
    print(f"Device: {device} | dtype: {cfg['dtype']}")

    if not os.path.exists(base_model):
        print(f"ERROR: Base FDM model not found at {base_model}/")
        print(f"  Train the FDM model first with the e2e script.")
        return

    train_path = os.path.join(data_dir, "nhop_train.jsonl")
    val_path = os.path.join(data_dir, "nhop_val.jsonl")
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run 'generate {model_name}' first.")
        return

    # Load from fine-tuned FDM model (not base pretrained)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["model_class"] == "gpt2":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=cfg["dtype"], trust_remote_code=True
        )

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    train_dataset = NHopDataset(train_path, tokenizer)
    val_dataset = NHopDataset(val_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    if bnb is not None and cfg["model_class"] == "auto" and cfg["dtype"] == torch.bfloat16:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    else:
        optimizer = AdamW(model.parameters(), lr=lr)
    grad_accum = cfg["grad_accum"]
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"E{epoch+1} [{model_name}]")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            train_loss += loss.item() * grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if (step + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)

        tag = " *best*" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"  Saved best model to {output_path}/")
        print(f"  Epoch {epoch+1}: Train={train_loss:.4f} Val={val_loss:.4f}{tag}")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nSaved: {output_path}/")


# ============================================================
# Evaluation
# ============================================================

def evaluate(model_name, model_path=None):
    cfg = MODEL_CONFIGS[model_name]
    if model_path is None:
        model_path = cfg["nhop_model_dir"]
    data_dir = cfg["nhop_data_dir"]
    test_path = os.path.join(data_dir, "nhop_test.jsonl")

    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} nhop model from {model_path}/...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["model_class"] == "gpt2":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=cfg["dtype"], trust_remote_code=True
        )
    model.to(device)
    model.eval()

    samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating {len(samples)} samples...\n")

    hop_results = defaultdict(lambda: {"total": 0, "action_correct": 0, "fact_correct": 0})
    per_channel_correct = defaultdict(int)
    per_channel_total = defaultdict(int)

    # Per-sample joint accuracy tracking
    per_sample_wrong = []

    for i, s in enumerate(tqdm(samples, desc=f"Eval [{model_name}]")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = s['answer']
        nhops = s['nhops']

        hop_results[nhops]["total"] += 1

        # Action keywords
        for kw in ['ABORT', 'PROCEED', 'HOLD', 'WAIT', 'CAUTION', 'EVACUATE',
                    'EMERGENCY', 'LOCKDOWN', 'CRITICAL', 'DEPLOY', 'DELAY',
                    'SHELTER', 'DESCEND', 'CONTINUE', 'FORTIFY', 'REASSIGN',
                    'GROUND', 'EXTRACT']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                hop_results[nhops]["action_correct"] += 1
                break

        # Channel-specific values
        memory = s['memory']
        for ch_id in s['channels']:
            name = MEMORY_SCHEMAS[ch_id][0]
            expected_val = memory[str(ch_id)]
            if expected_val in response:
                hop_results[nhops]["fact_correct"] += 1

        # Extra channels (8-39)
        sample_wrongs = []
        for ch in range(8, NUM_CHANNELS):
            name = MEMORY_SCHEMAS[ch][0]
            expected_val = memory[str(ch)]
            per_channel_total[ch] += 1
            if f"{name}={expected_val}" in response:
                per_channel_correct[ch] += 1
            else:
                sample_wrongs.append((ch, name, expected_val))
        per_sample_wrong.append(sample_wrongs)

        if i < 3:
            print(f"\n  [{model_name}] Sample {i+1} | {nhops}-hop")
            print(f"  Expected: {expected[:100]}...")
            print(f"  Got:      {response[:100]}...")

    # Results
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} N-HOP COMPOSITIONAL REASONING RESULTS")
    print(f"{'='*70}")
    print(f"{'Hops':<6} {'Action%':<10} {'Fact%':<10} {'N':<6}")
    print(f"{'-'*32}")
    for nhops in sorted(hop_results.keys()):
        r = hop_results[nhops]
        t = r['total']
        a_pct = 100 * r['action_correct'] / t if t else 0
        f_pct = 100 * r['fact_correct'] / (t * nhops) if t else 0
        print(f"{nhops:<6} {a_pct:<10.1f} {f_pct:<10.1f} {t:<6}")

    # Per-channel accuracy
    print(f"\nPer-Channel Extra Accuracy (ch8-39):")
    for ch in range(8, NUM_CHANNELS):
        t = per_channel_total[ch]
        c = per_channel_correct[ch]
        pct = 100 * c / t if t else 0
        flag = " <<<" if pct < 95 else ""
        print(f"  ch{ch} ({MEMORY_SCHEMAS[ch][0]:<12}): {pct:.1f}%{flag}")

    # Joint accuracy
    num_extra = NUM_CHANNELS - 8
    total = len(per_sample_wrong)
    print(f"\nJoint Accuracy (all N extra channels correct per sample):")
    print(f"{'N channels':<12} {'Correct':<14} {'Pct':<10}")
    for n in [1, 2, 4, 8, 16, 24, 32]:
        if n > num_extra:
            break
        extra_channels = list(range(8, 8 + n))
        count = sum(1 for wrongs in per_sample_wrong
                    if all(ch not in [w[0] for w in wrongs] for ch in extra_channels))
        pct = 100 * count / total if total else 0
        print(f"  {n:<10} {count}/{total:<12} {pct:.1f}%")

    all_correct = sum(1 for wrongs in per_sample_wrong if len(wrongs) == 0)
    print(f"\n  ALL {num_extra} correct: {all_correct}/{total} = {100*all_correct/total:.1f}%")

    # Error analysis
    error_samples = [(i, w) for i, w in enumerate(per_sample_wrong) if w]
    print(f"\nError Analysis: {len(error_samples)}/{total} samples have errors")
    ch_error_count = defaultdict(int)
    for _, wrongs in error_samples:
        for ch, name, exp in wrongs:
            ch_error_count[ch] += 1
    if ch_error_count:
        print(f"Most error-prone:")
        for ch, count in sorted(ch_error_count.items(), key=lambda x: -x[1])[:5]:
            print(f"  ch{ch} ({MEMORY_SCHEMAS[ch][0]}): {count} errors")

    # Save results
    results = {
        "model": model_name,
        "hop_results": {str(k): v for k, v in hop_results.items()},
        "per_channel": {str(ch): 100 * per_channel_correct[ch] / per_channel_total[ch]
                        for ch in range(8, NUM_CHANNELS) if per_channel_total[ch] > 0},
        "all_32_correct_pct": 100 * all_correct / total if total else 0,
        "error_samples": len(error_samples),
        "total_samples": total,
    }
    os.makedirs(f"nhop_results_{model_name}", exist_ok=True)
    with open(f"nhop_results_{model_name}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: nhop_results_{model_name}/results.json")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("N-Hop Compositional Reasoning — Model-Agnostic")
        print()
        print("Usage:")
        print("  python nhop_training_multi.py generate <model>")
        print("  python nhop_training_multi.py train <model>")
        print("  python nhop_training_multi.py eval <model> [model_path]")
        print()
        print("Models: gpt2, qwen3, lfm2")
        sys.exit(0)

    cmd = sys.argv[1]
    model_name = sys.argv[2]

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    if cmd == "generate":
        generate_nhop_data(model_name)
    elif cmd == "train":
        train(model_name)
    elif cmd == "eval":
        mp = sys.argv[3] if len(sys.argv) > 3 else None
        evaluate(model_name, model_path=mp)
