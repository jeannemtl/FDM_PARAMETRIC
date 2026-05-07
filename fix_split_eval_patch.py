#!/usr/bin/env python3
"""
Fix the broken patch from the first attempt.

The first patch had over-escaped \\n inside f-strings which produced
invalid Python. This script:
  1. Restores from .bak
  2. Re-applies the patch using AST-friendly raw injection
  3. Tests that the result parses cleanly before saving
"""

import ast
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("fdm_split_source_unified_clean.py")
BACKUP = Path("fdm_split_source_unified_clean.py.bak")

NEW_EVAL_FUNCTION = r'''def evaluate_split_source(model, tokenizer, write_head, encoder, device,
                           block_a_channels, block_b_channels, n_eval=20):
    """Evaluate split-source hybrid.

    Reports BOTH slot-level (per-channel mean, inflated n) and
    sample-level joint accuracy (all channels correct on same sample, honest n).
    """
    if write_head is not None:
        write_head.eval()

    block_a_correct = 0; block_a_total = 0
    block_b_correct = 0; block_b_total = 0

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
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
            if hit:
                block_a_correct += 1
            else:
                a_all = False
            block_a_total += 1

        for k in block_b_channels:
            pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"
            hit = re.search(pat, answer) is not None
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

    a_slot = block_a_correct / block_a_total if block_a_total else 0.0
    b_slot = block_b_correct / block_b_total if block_b_total else 0.0
    overall_slot = (block_a_correct + block_b_correct) / (block_a_total + block_b_total) if (block_a_total + block_b_total) else 0.0

    a_joint = sample_a_all / n_samples if n_samples else 0.0
    b_joint = sample_b_all / n_samples if n_samples else 0.0
    all_joint = sample_all_all / n_samples if n_samples else 0.0

    n_a = len(block_a_channels)
    n_b = len(block_b_channels)
    slot_total = block_a_total + block_b_total

    print("")
    print("    --- SLOT-LEVEL (per-channel mean, inflated n) ---")
    print("    Block A slot-level:  {:.1f}%  (n={} = {} ch x {} samples)".format(a_slot*100, block_a_total, n_a, n_samples))
    print("    Block B slot-level:  {:.1f}%  (n={} = {} ch x {} samples)".format(b_slot*100, block_b_total, n_b, n_samples))
    print("    Overall slot-level:  {:.1f}%  (n={})".format(overall_slot*100, slot_total))
    print("    --- SAMPLE-LEVEL JOINT (honest n) ---")
    print("    Block A joint (all {} correct/sample): {:.1f}%  (n={})".format(n_a, a_joint*100, n_samples))
    print("    Block B joint (all {} correct/sample): {:.1f}%  (n={})".format(n_b, b_joint*100, n_samples))
    print("    ALL joint (all 32 correct/sample): {:.1f}%  (n={})".format(all_joint*100, n_samples))

    return {
        "a_slot": a_slot,
        "b_slot": b_slot,
        "overall_slot": overall_slot,
        "block_a_correct": block_a_correct,
        "block_a_total": block_a_total,
        "block_b_correct": block_b_correct,
        "block_b_total": block_b_total,
        "a_joint": a_joint,
        "b_joint": b_joint,
        "all_joint": all_joint,
        "sample_a_all": sample_a_all,
        "sample_b_all": sample_b_all,
        "sample_all_all": sample_all_all,
        "n_samples": n_samples,
    }


'''

EXTRA_ARGS = (
    '    parser.add_argument("--skip_train", action="store_true",\n'
    '                        help="Skip training and only run evaluation.")\n'
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
    '    print("[seed] Set random seed to {}".format(args.seed))\n'
    '\n'
    '    if args.skip_train:\n'
    '        args.n_steps = 0\n'
    '        print("[skip_train] Training disabled, eval-only mode.")\n'
    '\n'
)


def main():
    if not TARGET.exists():
        print("ERROR: {} not found".format(TARGET))
        sys.exit(1)
    if not BACKUP.exists():
        print("ERROR: {} not found, cannot restore from backup".format(BACKUP))
        sys.exit(1)

    # Restore from backup
    shutil.copy(BACKUP, TARGET)
    print("[+] Restored from backup")

    src = TARGET.read_text()

    # Replace evaluate_split_source
    pattern = re.compile(
        r"def evaluate_split_source\(.*?\n(?=(?:def |# ===|\nclass ))",
        re.DOTALL,
    )
    if not pattern.search(src):
        print("ERROR: could not locate evaluate_split_source")
        sys.exit(1)
    src = pattern.sub(NEW_EVAL_FUNCTION, src, count=1)
    print("[+] Replaced evaluate_split_source")

    # Add --skip_train and --seed
    n_eval_re = re.compile(r'(parser\.add_argument\(\s*"--n_eval"[^)]*\)\s*\n)')
    if n_eval_re.search(src):
        src = n_eval_re.sub(r"\1" + EXTRA_ARGS, src, count=1)
        print("[+] Added flags")
    else:
        print("[!] could not find --n_eval line; flags not added")

    # Inject seed setup after parse_args
    parse_re = re.compile(r"(args\s*=\s*parser\.parse_args\(\)\s*\n)")
    if parse_re.search(src):
        src = parse_re.sub(r"\1" + SEED_BLOCK, src, count=1)
        print("[+] Added seed setup")
    else:
        print("[!] could not find parse_args; seed setup not added")

    # Validate by parsing
    try:
        ast.parse(src)
        print("[+] AST parse OK")
    except SyntaxError as e:
        print("ERROR: patched file has syntax error: {}".format(e))
        print("Not writing. Original is intact via backup.")
        sys.exit(1)

    TARGET.write_text(src)
    print("[+] Wrote {}".format(TARGET))
    print()
    print("Test:")
    print("  python fdm_split_source_unified_clean.py \\")
    print("    --model /workspace/FDM_IN_WEIGHTS/two_block_model \\")
    print("    --skip_train --n_eval 50 --seed 42 --tag qwen3_test")


if __name__ == "__main__":
    main()
