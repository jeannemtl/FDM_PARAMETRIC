"""
External curriculum driver for fdm_K960_source.py.

Imports the original module, overrides relevant constants per stage,
and runs generate/train as a curriculum without modifying the source file.

Usage:
  python3 run_curriculum.py generate   # ~30 min CPU
  python3 run_curriculum.py train      # ~5-7 hours GPU
"""
import sys, os, json, time, random
sys.path.insert(0, '/workspace/FDM_IN_WEIGHTS')

import importlib.util
spec = importlib.util.spec_from_file_location('m', '/workspace/FDM_IN_WEIGHTS/fdm_K960_source.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm


# ============================================================
# Curriculum stages — progressive a_low, lr, sample count
# ============================================================
STAGES = [
    {"name": "Stage 0: max contrast",   "a_low": 0.00, "samples":  8000, "epochs": 3, "lr": 5e-5},
    {"name": "Stage 1: gentle ASK",     "a_low": 0.05, "samples": 12000, "epochs": 4, "lr": 3e-5},
    {"name": "Stage 2: standard ASK",   "a_low": 0.10, "samples": 15000, "epochs": 4, "lr": 2e-5},
    {"name": "Stage 3: harder ASK",     "a_low": 0.15, "samples": 15000, "epochs": 5, "lr": 1e-5},
    {"name": "Stage 4: hardest ASK",    "a_low": 0.20, "samples": 15000, "epochs": 5, "lr": 1e-5},
]


def stage_data_dir(stage_idx):
    return f"/workspace/FDM_IN_WEIGHTS/data_fdm_K960_stage{stage_idx}"

def stage_model_dir(stage_idx):
    return f"/workspace/FDM_IN_WEIGHTS/models/fdm_K960_qwen3/stage{stage_idx}"


def generate():
    """Generate one dataset per stage with stage-specific a_low."""
    print(f"Generating curriculum-stage datasets...")
    print(f"  Total stages: {len(STAGES)}")
    print(f"  NUM_LEVELS will be bumped to 128 in encoder for finer K=960 resolution")

    tokenizer = AutoTokenizer.from_pretrained(m.QWEN3_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = random.Random(m.SEED)
    n_eval = 1500

    for stage_idx, stage in enumerate(STAGES):
        data_dir = stage_data_dir(stage_idx)
        os.makedirs(data_dir, exist_ok=True)
        print(f"\n=== {stage['name']} (a_low={stage['a_low']}, samples={stage['samples']}) ===")

        # Build encoder with stage-specific a_low and bumped num_levels
        stage_encoder = m.FDMSignalEncoder(
            vocab_size=m.VOCAB_SIZE_QWEN3, tokenizer=tokenizer,
            a_low=stage["a_low"], num_levels=256,
        )

        out_path = os.path.join(data_dir, "train.jsonl")
        print(f"Generating train ({stage['samples']:,} samples)...")
        with open(out_path, "w") as f:
            for i in tqdm(range(stage["samples"])):
                memory = {ch: rng.choice(m.MEMORY_SCHEMAS[ch][1]) for ch in range(m.K)}
                fdm_text, token_ids = stage_encoder.encode_memory(memory)
                query_ch = rng.randrange(m.K)
                sample = {
                    "fdm_text": fdm_text,
                    "query_channel": query_ch,
                    "query_name": m.CHANNEL_NAMES[query_ch],
                    "answer": memory[query_ch],
                    "memory": {str(ch): memory[ch] for ch in range(m.K)},
                }
                f.write(json.dumps(sample) + "\n")
        print(f"  -> {out_path}")

        # Last stage also generates eval data
        if stage_idx == len(STAGES) - 1:
            eval_path = os.path.join(data_dir, "eval.jsonl")
            print(f"Generating eval ({n_eval:,} samples) at hardest stage...")
            with open(eval_path, "w") as f:
                for i in tqdm(range(n_eval)):
                    memory = {ch: rng.choice(m.MEMORY_SCHEMAS[ch][1]) for ch in range(m.K)}
                    fdm_text, token_ids = stage_encoder.encode_memory(memory)
                    query_ch = rng.randrange(m.K)
                    sample = {
                        "fdm_text": fdm_text,
                        "query_channel": query_ch,
                        "query_name": m.CHANNEL_NAMES[query_ch],
                        "answer": memory[query_ch],
                        "memory": {str(ch): memory[ch] for ch in range(m.K)},
                    }
                    f.write(json.dumps(sample) + "\n")
            print(f"  -> {eval_path}")

    print("\nDone generating all stages.")


def train():
    """Curriculum training: 5 stages, each loading its own dataset, progressive a_low and lr."""
    os.makedirs("/workspace/FDM_IN_WEIGHTS/models/fdm_K960_qwen3", exist_ok=True)
    os.makedirs(m.LOGS_DIR, exist_ok=True)

    print(f"Loading {m.QWEN3_BASE}...")
    tokenizer = AutoTokenizer.from_pretrained(m.QWEN3_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine resume point
    resume_stage = 0
    for i in range(len(STAGES)):
        if os.path.exists(os.path.join(stage_model_dir(i), "config.json")):
            resume_stage = i + 1

    if resume_stage > 0:
        load_from = stage_model_dir(resume_stage - 1)
        print(f"Resuming from {load_from} (stage {resume_stage - 1} complete)")
    else:
        load_from = m.QWEN3_BASE

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            load_from, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", trust_remote_code=True,
        ).cuda()
        print("  flash_attention_2 enabled")
    except Exception as e:
        print(f"  flash_attention_2 unavailable ({e}); using sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            load_from, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).cuda()

    for stage_idx in range(resume_stage, len(STAGES)):
        stage = STAGES[stage_idx]
        data_dir = stage_data_dir(stage_idx)
        train_path = os.path.join(data_dir, "train.jsonl")

        if not os.path.exists(train_path):
            print(f"\nERROR: {train_path} not found. Run 'generate' first.")
            return

        print(f"\n{'='*60}")
        print(f"  STAGE {stage_idx}: {stage['name']}")
        print(f"  a_low={stage['a_low']}, lr={stage['lr']}, epochs={stage['epochs']}")
        print(f"{'='*60}")

        dataset = m.FDMDataset(train_path, tokenizer)
        print(f"  loaded {len(dataset):,} training samples")

        loader = DataLoader(
            dataset, batch_size=m.BATCH_SIZE, shuffle=True,
            num_workers=4,
            collate_fn=lambda b: m.collate(b, tokenizer.pad_token_id),
            pin_memory=True, drop_last=True,
        )

        stage_steps = (len(loader) * stage["epochs"]) // m.GRAD_ACCUM
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=stage["lr"], weight_decay=m.WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(m.WARMUP_STEPS, stage_steps // 10),
            num_training_steps=stage_steps,
        )

        print(f"\n  Training plan (stage {stage_idx}):")
        print(f"    samples/epoch  = {len(dataset):,}")
        print(f"    batch/grad_acc = {m.BATCH_SIZE} x {m.GRAD_ACCUM}")
        print(f"    epochs         = {stage['epochs']}")
        print(f"    optim steps    = {stage_steps:,}")
        print(f"    lr             = {stage['lr']} (cosine)")

        model.train()
        step = 0
        t_start = time.perf_counter()
        accum_loss = 0.0
        accum_count = 0

        for epoch in range(stage["epochs"]):
            for it, batch in enumerate(loader):
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / m.GRAD_ACCUM
                loss.backward()
                accum_loss += loss.item() * m.GRAD_ACCUM
                accum_count += 1

                if (it + 1) % m.GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if step % 50 == 0:
                        avg_loss = accum_loss / accum_count
                        elapsed = time.perf_counter() - t_start
                        samples_done = step * m.BATCH_SIZE * m.GRAD_ACCUM
                        samples_per_sec = samples_done / elapsed
                        eta_h = (stage_steps - step) * (elapsed / step) / 3600
                        print(f"  S{stage_idx} step {step:>6}/{stage_steps} "
                              f"| loss {avg_loss:.4f} "
                              f"| {samples_per_sec:.1f} samp/s "
                              f"| ETA {eta_h:.2f}h")
                        accum_loss = 0.0
                        accum_count = 0

        # Save stage checkpoint
        save_dir = stage_model_dir(stage_idx)
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"\n  Saved stage {stage_idx} -> {save_dir}")

    # Save final model alias
    final_dir = "/workspace/FDM_IN_WEIGHTS/models/fdm_K960_qwen3/final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final -> {final_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_curriculum.py {generate,train}")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "generate":
        generate()
    elif cmd == "train":
        train()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
