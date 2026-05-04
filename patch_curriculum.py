"""
Apply curriculum patch to fdm_K960_source.py.

Changes:
  1. Bump NUM_LEVELS from 64 to 128 (finer per-channel resolution at K=960)
  2. Add STAGES list with progressive a_low
  3. Replace generate_data with stage-aware version
  4. Replace train_model with curriculum-aware version
  5. Replace evaluate_model to use final stage's data

Original file is backed up to fdm_K960_source.py.bak.
"""
import re

PATH = "fdm_K960_source.py"
src = open(PATH).read()

# === 1. Bump NUM_LEVELS ===
src = re.sub(r"^NUM_LEVELS\s*=\s*\d+.*$",
             "NUM_LEVELS = 128                 # quantization levels (bumped from 64 for K=960)",
             src, count=1, flags=re.MULTILINE)

# === 2. Insert STAGES + helper near the top, after constants ===
stages_block = '''

# ============================================================
# Curriculum stages (added by patch_curriculum.py)
# ============================================================
STAGES = [
    {"name": "Stage 0: max contrast",   "a_low": 0.00, "samples":  8000, "epochs": 3, "lr": 5e-5},
    {"name": "Stage 1: gentle ASK",     "a_low": 0.05, "samples": 12000, "epochs": 4, "lr": 3e-5},
    {"name": "Stage 2: standard ASK",   "a_low": 0.10, "samples": 15000, "epochs": 4, "lr": 2e-5},
    {"name": "Stage 3: harder ASK",     "a_low": 0.15, "samples": 15000, "epochs": 5, "lr": 1e-5},
    {"name": "Stage 4: hardest ASK",    "a_low": 0.20, "samples": 15000, "epochs": 5, "lr": 1e-5},
]

def stage_data_dir(stage_idx):
    return f"data_fdm_K960_stage{stage_idx}"

def stage_model_dir(stage_idx):
    return f"models/fdm_K960_qwen3/stage{stage_idx}"

def stage_eval_data_dir(stage_idx):
    return f"data_fdm_K960_stage{stage_idx}"

'''

# Insert after the "DATA_DIR = ..." / "MODEL_DIR = ..." block
src = re.sub(
    r'(MODEL_DIR\s*=\s*"[^"]+")\n',
    r'\1\n' + stages_block,
    src, count=1
)

# === 3. Replace generate_data ===
new_generate = '''def generate_data(args):
    """Curriculum-aware data generation: one dataset per stage, each with its own a_low."""
    print(f"Generating curriculum-stage datasets...")
    print(f"  Total stages: {len(STAGES)}")

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = random.Random(SEED)

    # Eval data uses the hardest stage's a_low (final challenge)
    eval_a_low = STAGES[-1]["a_low"]
    n_eval = getattr(args, "n_eval", 1500)

    for stage_idx, stage in enumerate(STAGES):
        data_dir = stage_data_dir(stage_idx)
        os.makedirs(data_dir, exist_ok=True)
        print(f"\\n=== {stage['name']} (a_low={stage['a_low']}, samples={stage['samples']}) ===")

        # Build encoder for this stage's a_low
        stage_encoder = FDMSignalEncoder(
            vocab_size=VOCAB_SIZE_QWEN3, tokenizer=tokenizer, a_low=stage["a_low"],
        )

        out_path = os.path.join(data_dir, "train.jsonl")
        print(f"Generating train ({stage['samples']:,} samples)...")
        with open(out_path, "w") as f:
            for i in tqdm(range(stage["samples"])):
                memory = {ch: rng.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(K)}
                fdm_text, token_ids = stage_encoder.encode_memory(memory)
                query_ch = rng.randrange(K)
                sample = {
                    "fdm_text": fdm_text,
                    "query_channel": query_ch,
                    "query_name": CHANNEL_NAMES[query_ch],
                    "answer": memory[query_ch],
                    "memory": {str(ch): memory[ch] for ch in range(K)},
                }
                f.write(json.dumps(sample) + "\\n")
        print(f"  -> {out_path}")

        # Last stage also generates eval data
        if stage_idx == len(STAGES) - 1:
            eval_path = os.path.join(data_dir, "eval.jsonl")
            print(f"Generating eval ({n_eval:,} samples) at hardest stage...")
            with open(eval_path, "w") as f:
                for i in tqdm(range(n_eval)):
                    memory = {ch: rng.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(K)}
                    fdm_text, token_ids = stage_encoder.encode_memory(memory)
                    query_ch = rng.randrange(K)
                    sample = {
                        "fdm_text": fdm_text,
                        "query_channel": query_ch,
                        "query_name": CHANNEL_NAMES[query_ch],
                        "answer": memory[query_ch],
                        "memory": {str(ch): memory[ch] for ch in range(K)},
                    }
                    f.write(json.dumps(sample) + "\\n")
            print(f"  -> {eval_path}")
'''

src = re.sub(
    r'def generate_data\(args\):.*?(?=\n# =+\n# Training|\ndef train_model)',
    new_generate + "\n\n",
    src, count=1, flags=re.DOTALL
)

# === 4. Replace train_model ===
new_train = '''def train_model(args):
    """Curriculum training: 5 stages, each loading its own dataset, progressive a_low and lr."""
    os.makedirs("models/fdm_K960_qwen3", exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"Loading {QWEN3_BASE}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine resume point
    resume_stage = 0
    for i in range(len(STAGES)):
        if os.path.exists(os.path.join(stage_model_dir(i), "config.json")):
            resume_stage = i + 1

    if resume_stage > 0:
        # Load most recent completed stage
        load_from = stage_model_dir(resume_stage - 1)
        print(f"Resuming from {load_from} (stage {resume_stage - 1} complete)")
    else:
        load_from = QWEN3_BASE

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
            print(f"\\nERROR: {train_path} not found. Run 'generate' first.")
            return

        print(f"\\n{'='*60}")
        print(f"  STAGE {stage_idx}: {stage['name']}")
        print(f"  a_low={stage['a_low']}, lr={stage['lr']}, epochs={stage['epochs']}")
        print(f"{'='*60}")

        dataset = FDMDataset(train_path, tokenizer)
        print(f"  loaded {len(dataset):,} training samples")

        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
            pin_memory=True, drop_last=True,
        )

        stage_steps = (len(loader) * stage["epochs"]) // GRAD_ACCUM
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=stage["lr"], weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(WARMUP_STEPS, stage_steps // 10),
            num_training_steps=stage_steps,
        )

        print(f"\\n  Training plan (stage {stage_idx}):")
        print(f"    samples/epoch  = {len(dataset):,}")
        print(f"    batch/grad_acc = {BATCH_SIZE} x {GRAD_ACCUM}")
        print(f"    epochs         = {stage['epochs']}")
        print(f"    optim steps    = {stage_steps:,}")
        print(f"    lr             = {stage['lr']} (cosine)\\n")

        model.train()
        step = 0
        t_start = time.perf_counter()
        accum_loss = 0.0
        accum_count = 0

        for epoch in range(stage["epochs"]):
            for it, batch in enumerate(loader):
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / GRAD_ACCUM
                loss.backward()
                accum_loss += loss.item() * GRAD_ACCUM
                accum_count += 1

                if (it + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if step % 50 == 0:
                        avg_loss = accum_loss / accum_count
                        elapsed = time.perf_counter() - t_start
                        samples_done = step * BATCH_SIZE * GRAD_ACCUM
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
        print(f"\\n  Saved stage {stage_idx} -> {save_dir}")

    # Save final model (alias of last stage)
    final_dir = "models/fdm_K960_qwen3/final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\\nSaved final model -> {final_dir}")
'''

src = re.sub(
    r'def train_model\(args\):.*?(?=\n# =+|\ndef \w+\()',
    new_train + "\n\n",
    src, count=1, flags=re.DOTALL
)

# === 5. Patch evaluate_model to use last stage's eval data ===
src = re.sub(
    r'eval_path\s*=\s*os\.path\.join\(DATA_DIR,\s*["\']eval\.jsonl["\']\)',
    'eval_path = os.path.join(stage_data_dir(len(STAGES) - 1), "eval.jsonl")',
    src
)

# Patch the final_dir reference in evaluate_model
src = re.sub(
    r'final_dir\s*=\s*os\.path\.join\(MODEL_DIR,\s*["\']final["\']\)',
    'final_dir = "models/fdm_K960_qwen3/final"',
    src
)

# Write back
with open(PATH, "w") as f:
    f.write(src)

print(f"Patched {PATH}")
print("Backup at fdm_K960_source.py.bak")
print()
print("New constants:")
print("  NUM_LEVELS: 128 (was 64)")
print()
print("New stages:")
for i, s in enumerate([
    ("Stage 0", 0.00, 8000, 3, "5e-5"),
    ("Stage 1", 0.05, 12000, 4, "3e-5"),
    ("Stage 2", 0.10, 15000, 4, "2e-5"),
    ("Stage 3", 0.15, 15000, 5, "1e-5"),
    ("Stage 4", 0.20, 15000, 5, "1e-5"),
]):
    name, a_low, samples, epochs, lr = s
    print(f"  {name}: a_low={a_low}, {samples} samples, {epochs} ep, lr={lr}")
