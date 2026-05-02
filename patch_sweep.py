#!/usr/bin/env python3
"""
Patch fdm_split_ratio_sweep.py to:
  1. Report sample-level joint accuracy alongside slot-level mean.
  2. Add --seed flag for reproducibility.
  3. Save per-sample correctness arrays so Wilson CIs can be computed exactly.
  4. Validate via AST parse before writing.

Run from the directory containing fdm_split_ratio_sweep.py:
    python patch_sweep.py
"""

import ast
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("fdm_split_ratio_sweep.py")
BACKUP = Path("fdm_split_ratio_sweep.py.bak")


# Replacement for the no-write-head (n_param == 0) eval block
NEW_NOWH_EVAL = r'''        print("  No write head needed - evaluating context-only...")
        slot_correct = 0
        slot_total = 0
        sample_all_correct = 0
        n_samples = 0
        per_sample_correct = []  # list of bools

        for _ in tqdm(range(n_eval), desc="Eval", leave=False):
            mem = random_memory()
            answer = generate_with_split(model, tokenizer, None, mem, encoder, device,
                                          [], block_b_channels)
            sample_all = True
            sample_hits = []
            for k in ALL_QUERY_CHANNELS:
                pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
                hit = re.search(pat, answer) is not None
                sample_hits.append(bool(hit))
                if hit:
                    slot_correct += 1
                else:
                    sample_all = False
                slot_total += 1
            if sample_all:
                sample_all_correct += 1
            n_samples += 1
            per_sample_correct.append(sample_hits)

        slot_acc = slot_correct / slot_total if slot_total > 0 else 0.0
        joint_acc = sample_all_correct / n_samples if n_samples > 0 else 0.0
        print("  Slot-level (per-channel mean): {:.1f}%  (n={})".format(slot_acc*100, slot_total))
        print("  Sample-level joint (all 32 correct): {:.1f}%  (n={})".format(joint_acc*100, n_samples))
        return {
            "block_a_acc": 0.0,
            "block_b_acc": slot_acc,
            "all_acc": slot_acc,           # legacy slot-level field, kept for compatibility
            "all_acc_slot": slot_acc,
            "all_acc_joint": joint_acc,
            "n_param": 0,
            "n_ctx": n_ctx_ch,
            "n_samples": n_samples,
            "slot_correct": slot_correct,
            "slot_total": slot_total,
            "sample_all_correct": sample_all_correct,
            "per_sample_correct": per_sample_correct,
        }

'''


# Replacement for the post-training eval block
NEW_POSTTRAIN_EVAL = r'''    write_head.eval()
    block_a_slot_correct = 0; block_a_slot_total = 0
    block_b_slot_correct = 0; block_b_slot_total = 0
    all_slot_correct = 0;     all_slot_total = 0

    sample_a_all = 0
    sample_b_all = 0
    sample_all_all = 0
    n_samples = 0
    per_sample_a = []
    per_sample_b = []
    per_sample_all = []

    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        answer = generate_with_split(model, tokenizer, write_head, mem, encoder, device,
                                      block_a_channels, block_b_channels)

        a_all = True
        b_all = True

        for k in block_a_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                block_a_slot_correct += 1
            else:
                a_all = False
            block_a_slot_total += 1

        for k in block_b_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                block_b_slot_correct += 1
            else:
                b_all = False
            block_b_slot_total += 1

        for k in ALL_QUERY_CHANNELS:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                all_slot_correct += 1
            all_slot_total += 1

        if a_all:
            sample_a_all += 1
        if b_all:
            sample_b_all += 1
        if a_all and b_all:
            sample_all_all += 1
        n_samples += 1
        per_sample_a.append(bool(a_all))
        per_sample_b.append(bool(b_all))
        per_sample_all.append(bool(a_all and b_all))

    a_slot = block_a_slot_correct / block_a_slot_total if block_a_slot_total > 0 else 0.0
    b_slot = block_b_slot_correct / block_b_slot_total if block_b_slot_total > 0 else 0.0
    all_slot = all_slot_correct / all_slot_total if all_slot_total > 0 else 0.0

    a_joint = sample_a_all / n_samples if n_samples > 0 else 0.0
    b_joint = sample_b_all / n_samples if n_samples > 0 else 0.0
    all_joint = sample_all_all / n_samples if n_samples > 0 else 0.0

    print("  --- SLOT-LEVEL (per-channel mean) ---")
    print("  Block A: {:.1f}%  (n={})".format(a_slot*100, block_a_slot_total))
    print("  Block B: {:.1f}%  (n={})".format(b_slot*100, block_b_slot_total))
    print("  All:     {:.1f}%  (n={})".format(all_slot*100, all_slot_total))
    print("  --- SAMPLE-LEVEL JOINT (honest n) ---")
    print("  Block A all-correct: {:.1f}%  (n={})".format(a_joint*100, n_samples))
    print("  Block B all-correct: {:.1f}%  (n={})".format(b_joint*100, n_samples))
    print("  All-32 all-correct:  {:.1f}%  (n={})".format(all_joint*100, n_samples))

    return {
        # Legacy fields (slot-level) - kept so existing summary print works
        "block_a_acc": float(a_slot),
        "block_b_acc": float(b_slot),
        "all_acc": float(all_slot),
        # Explicit slot-level
        "block_a_acc_slot": float(a_slot),
        "block_b_acc_slot": float(b_slot),
        "all_acc_slot": float(all_slot),
        # Sample-level joint
        "block_a_acc_joint": float(a_joint),
        "block_b_acc_joint": float(b_joint),
        "all_acc_joint": float(all_joint),
        # Raw counts
        "block_a_slot_correct": block_a_slot_correct,
        "block_a_slot_total": block_a_slot_total,
        "block_b_slot_correct": block_b_slot_correct,
        "block_b_slot_total": block_b_slot_total,
        "sample_a_all_correct": sample_a_all,
        "sample_b_all_correct": sample_b_all,
        "sample_all_all_correct": sample_all_all,
        "n_samples": n_samples,
        # Per-sample arrays for exact Wilson CI computation
        "per_sample_block_a_all": per_sample_a,
        "per_sample_block_b_all": per_sample_b,
        "per_sample_all32_all":   per_sample_all,
        # Other metadata
        "n_param": n_param_ch,
        "n_ctx": n_ctx_ch,
        "final_loss": float(np.mean(losses[-500:])),
    }
'''


SEED_FLAG = (
    '    parser.add_argument("--seed", type=int, default=42,\n'
    '                        help="Random seed for reproducibility.")\n'
)

SEED_BLOCK = (
    '\n'
    '    import random as _random\n'
    '    _random.seed(args.seed)\n'
    '    np.random.seed(args.seed)\n'
    '    torch.manual_seed(args.seed)\n'
    '    if torch.cuda.is_available():\n'
    '        torch.cuda.manual_seed_all(args.seed)\n'
    '    print("[seed] Random seed set to {}".format(args.seed))\n'
    '\n'
)


def main():
    if not TARGET.exists():
        print("ERROR: {} not found. cd to its directory first.".format(TARGET))
        sys.exit(1)

    if not BACKUP.exists():
        shutil.copy(TARGET, BACKUP)
        print("[+] Backup created: {}".format(BACKUP))
    else:
        print("[!] Backup exists, leaving in place: {}".format(BACKUP))

    src = TARGET.read_text()

    # ---- 1. Replace the no-write-head eval block ----
    # Locate the block by its distinctive opening string and replace through
    # the closing return statement (return {'block_a_acc': 0.0, ...}).
    nowh_pattern = re.compile(
        r'        print\("  No write head needed.*?return \{[^}]*?\}\n',
        re.DOTALL,
    )
    if nowh_pattern.search(src):
        src = nowh_pattern.sub(NEW_NOWH_EVAL, src, count=1)
        print("[+] Replaced no-write-head eval block")
    else:
        print("[!] Could not locate no-write-head eval block; skipping")

    # ---- 2. Replace the post-training eval block ----
    # The block starts at `write_head.eval()` AFTER the training loop and ends
    # at the existing `return {'block_a_acc': float(a_acc), ...}` block.
    # Use the unique sentinel `block_a_correct = 0; block_a_total = 0` which
    # appears only in the post-training eval (the no-WH block uses different vars).
    posttrain_pattern = re.compile(
        r'    write_head\.eval\(\)\n'
        r'    block_a_correct = 0; block_a_total = 0\n'
        r'.*?'
        r'    return \{[^}]*?\}\n',
        re.DOTALL,
    )
    if posttrain_pattern.search(src):
        src = posttrain_pattern.sub(NEW_POSTTRAIN_EVAL, src, count=1)
        print("[+] Replaced post-training eval block")
    else:
        print("[!] Could not locate post-training eval block; skipping")

    # ---- 3. Add --seed flag ----
    n_eval_re = re.compile(
        r'(parser\.add_argument\(\s*"--n_eval"[^)]*\)\s*\n)'
    )
    if n_eval_re.search(src) and "--seed" not in src:
        src = n_eval_re.sub(r"\1" + SEED_FLAG, src, count=1)
        print("[+] Added --seed flag")
    elif "--seed" in src:
        print("[!] --seed already present; not adding again")
    else:
        print("[!] Could not find --n_eval line; --seed not added")

    # ---- 4. Inject seed setup after parse_args ----
    parse_re = re.compile(r"(args\s*=\s*parser\.parse_args\(\)\s*\n)")
    if parse_re.search(src) and "[seed] Random seed set" not in src:
        src = parse_re.sub(r"\1" + SEED_BLOCK, src, count=1)
        print("[+] Added seed initialization")
    elif "[seed] Random seed set" in src:
        print("[!] Seed init already present")
    else:
        print("[!] Could not find parse_args; seed init not added")

    # ---- 5. Validate ----
    try:
        ast.parse(src)
        print("[+] AST parse OK")
    except SyntaxError as e:
        print("ERROR: patched file has syntax error: {}".format(e))
        print("File NOT written. Original is still intact.")
        sys.exit(1)

    TARGET.write_text(src)
    print("[+] Wrote patched {}".format(TARGET))
    print()
    print("Test (small n=5, single ratio, fast smoke test):")
    print("  python fdm_split_ratio_sweep.py --n_eval 5 --n_steps 100 --splits 16 --seed 42")
    print()
    print("Real run (n=200, all 9 ratios, ~9 hours):")
    print("  tmux new -s sweep \"python fdm_split_ratio_sweep.py \\")
    print("      --n_eval 200 --seed 42 \\")
    print("      --output_dir /workspace/FDM_IN_WEIGHTS/split_ratio_sweep_n200 \\")
    print("      2>&1 | tee sweep_n200.log\"")


if __name__ == "__main__":
    main()
