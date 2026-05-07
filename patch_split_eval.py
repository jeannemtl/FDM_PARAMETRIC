#!/usr/bin/env python3
"""
Patch fdm_split_source_unified_clean.py to:
1. Replace evaluate_split_source with a version that reports BOTH
   slot-level accuracy AND sample-level joint accuracy.
2. Add --skip_train flag (explicitly skip training, eval only).
3. Add --seed flag for reproducibility.
4. Save both metric types in results.json.

Usage:
    cd /workspace/FDM_IN_WEIGHTS
    python patch_split_eval.py
    # creates fdm_split_source_unified_clean.py.bak as backup

After patching, evaluate-only mode at n=500:
    python fdm_split_source_unified_clean.py \
        --model /workspace/FDM_IN_WEIGHTS/two_block_model \
        --skip_train \
        --n_eval 500 \
        --seed 42 \
        --tag qwen3_n500
"""

import re
import shutil
import sys
from pathlib import Path

TARGET = Path("fdm_split_source_unified_clean.py")
BACKUP = Path("fdm_split_source_unified_clean.py.bak")

NEW_EVAL_FUNCTION = '''def evaluate_split_source(model, tokenizer, write_head, encoder, device,
                           block_a_channels, block_b_channels, n_eval=20):
    """Evaluate split-source hybrid.

    Reports BOTH:
      - Slot-level accuracy (per-channel mean across n_eval samples)
      - Sample-level joint accuracy (all channels in group correct on same sample)

    Sample-level joint is what should be reported as 'Joint (32)' in tables.
    Slot-level is the channel-by-channel mean and uses inflated n.
    """
    if write_head is not None:
        write_head.eval()

    # Slot-level counters
    block_a_correct = 0; block_a_total = 0
    block_b_correct = 0; block_b_total = 0

    # Sample-level joint counters
    sample_a_all = 0
    sample_b_all = 0
    sample_all_all = 0
    n_samples = 0

    for _ in tqdm(range(n_eval), desc="Eval", leave=False):
        mem = random_memory()
        answer = generate_with_hybrid(model, tokenizer, write_head, mem, encoder, device,
                                       block_a_channels, block_b_channels)

        a_all = True
        b_all = True

        for k in block_a_channels:
            hit = re.search(rf"\\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\\b", answer) is not None
            if hit:
                block_a_correct += 1
            else:
                a_all = False
            block_a_total += 1

        for k in block_b_channels:
            hit = re.search(rf"\\b{re.escape(CHANNEL_NAMES[k])}={re.escape(mem[k])}\\b", answer) is not None
            if hit:
                block_b_correct += 1
            else:
                b_all = False
            block_b_total += 1

        if a_all:
            sample_a_all += 1
        if b_all:
            sample_b_all += 1
        if a_all and b_all:
            sample_all_all += 1
        n_samples += 1

    # Slot-level (per-channel mean across all sample/channel slots)
    a_slot = block_a_correct / block_a_total if block_a_total else 0.0
    b_slot = block_b_correct / block_b_total if block_b_total else 0.0
    overall_slot = (block_a_correct + block_b_correct) / (block_a_total + block_b_total) \
                    if (block_a_total + block_b_total) else 0.0

    # Sample-level joint (all channels in group correct in same sample)
    a_joint = sample_a_all / n_samples if n_samples else 0.0
    b_joint = sample_b_all / n_samples if n_samples else 0.0
    all_joint = sample_all_all / n_samples if n_samples else 0.0

    print(f"\\n    --- SLOT-LEVEL (per-channel mean, inflated n) ---")
    print(f"    Block A slot-level:  {a_slot*100:.1f}%  (n={block_a_total} = {len(block_a_channels)} ch x {n_samples} samples)")
    print(f"    Block B slot-level:  {b_slot*100:.1f}%  (n={block_b_total} = {len(block_b_channels)} ch x {n_samples} samples)")
    print(f"    Overall slot-level:  {overall_slot*100:.1f}%  (n={block_a_total + block_b_total})")
    print(f"    --- SAMPLE-LEVEL JOINT (honest n, this is what 'Joint' should mean) ---")
    print(f"    Block A joint (all {len(block_a_channels)} correct/sample): {a_joint*100:.1f}%  (n={n_samples})")
    print(f"    Block B joint (all {len(block_b_channels)} correct/sample): {b_joint*100:.1f}%  (n={n_samples})")
    print(f"    ALL joint (all 32 correct/sample): {all_joint*100:.1f}%  (n={n_samples})")

    # Return dict with both metric types; preserve old 3-tuple via .values() if needed
    return {
        # Slot-level
        "a_slot": a_slot,
        "b_slot": b_slot,
        "overall_slot": overall_slot,
        "block_a_correct": block_a_correct,
        "block_a_total": block_a_total,
        "block_b_correct": block_b_correct,
        "block_b_total": block_b_total,
        # Sample-level joint
        "a_joint": a_joint,
        "b_joint": b_joint,
        "all_joint": all_joint,
        "sample_a_all": sample_a_all,
        "sample_b_all": sample_b_all,
        "sample_all_all": sample_all_all,
        "n_samples": n_samples,
    }


'''


def main():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found in current directory.")
        print("cd to the directory containing the script first.")
        sys.exit(1)

    src = TARGET.read_text()

    # 1. Backup
    if not BACKUP.exists():
        shutil.copy(TARGET, BACKUP)
        print(f"[+] Backup created: {BACKUP}")
    else:
        print(f"[!] Backup already exists: {BACKUP} (not overwriting)")

    # 2. Replace evaluate_split_source function
    # Match from 'def evaluate_split_source' through the next top-level def or '# ====' marker
    pattern = re.compile(
        r"def evaluate_split_source\(.*?\n(?=(?:def |# ===|\nclass ))",
        re.DOTALL,
    )
    if not pattern.search(src):
        print("ERROR: could not locate evaluate_split_source function.")
        sys.exit(1)

    new_src = pattern.sub(NEW_EVAL_FUNCTION, src, count=1)
    print("[+] Replaced evaluate_split_source function.")

    # 3. Add --skip_train and --seed flags after --n_eval line
    n_eval_arg_re = re.compile(
        r'(parser\.add_argument\(\s*"--n_eval"[^)]*\)\s*\n)',
    )
    extra_args = (
        '    parser.add_argument("--skip_train", action="store_true",\n'
        '                        help="Skip training and only run evaluation on the loaded model.")\n'
        '    parser.add_argument("--seed", type=int, default=42,\n'
        '                        help="Random seed for reproducibility.")\n'
    )
    if n_eval_arg_re.search(new_src):
        new_src = n_eval_arg_re.sub(r"\1" + extra_args, new_src, count=1)
        print("[+] Added --skip_train and --seed flags.")
    else:
        print("[!] Could not find --n_eval arg line; skipping flag addition.")

    # 4. Inject seed-setting and skip_train logic after args = parser.parse_args()
    parse_re = re.compile(r"(args\s*=\s*parser\.parse_args\(\)\s*\n)")
    seed_block = (
        '\n    # Reproducibility\n'
        '    import random as _random\n'
        '    _random.seed(args.seed)\n'
        '    np.random.seed(args.seed)\n'
        '    torch.manual_seed(args.seed)\n'
        '    if torch.cuda.is_available():\n'
        '        torch.cuda.manual_seed_all(args.seed)\n'
        '    print(f"[seed] Set random seed to {args.seed}")\n\n'
        '    # If --skip_train, force n_steps to 0\n'
        '    if args.skip_train:\n'
        '        args.n_steps = 0\n'
        '        print("[skip_train] Training disabled, eval-only mode.")\n\n'
    )
    if parse_re.search(new_src):
        new_src = parse_re.sub(r"\1" + seed_block, new_src, count=1)
        print("[+] Added seed setup and skip_train logic.")
    else:
        print("[!] Could not find parse_args call; manual edit needed.")

    # 5. Write back
    TARGET.write_text(new_src)
    print(f"[+] Wrote patched {TARGET}")
    print()
    print("Test with a small n first:")
    print("    python fdm_split_source_unified_clean.py \\")
    print("        --model /workspace/FDM_IN_WEIGHTS/two_block_model \\")
    print("        --skip_train --n_eval 50 --seed 42 --tag qwen3_test")
    print()
    print("If that works, run the real eval at n=500:")
    print("    tmux new -s split_eval 'python fdm_split_source_unified_clean.py \\")
    print("        --model /workspace/FDM_IN_WEIGHTS/two_block_model \\")
    print("        --skip_train --n_eval 500 --seed 42 --tag qwen3_n500 \\")
    print("        2>&1 | tee split_eval_qwen3_n500.log'")


if __name__ == "__main__":
    main()
