#!/usr/bin/env python3
"""
Fix the broken regex patterns left by patch_sweep.py.

The patch produced lines like:
    pat = r" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"

which should be:
    pat = r"\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\b"

The \b word boundaries got eaten by the patch's heredoc/regex escaping.

Run from the directory containing fdm_split_ratio_sweep.py.
Validates with AST parse before writing.
"""

import ast
import re
import sys
from pathlib import Path

TARGET = Path("fdm_split_ratio_sweep.py")

if not TARGET.exists():
    print("ERROR: {} not found. cd to its directory.".format(TARGET))
    sys.exit(1)

src = TARGET.read_text()

BROKEN = 'pat = r" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"'
FIXED  = 'pat = r"\\b" + re.escape(CHANNEL_NAMES[k]) + r"=" + re.escape(mem[k]) + r"\\b"'

count = src.count(BROKEN)
if count == 0:
    print("[!] No broken regex lines found. Either already fixed, or pattern is different.")
    print("    Searching for any remaining 'pat = r\"' lines for diagnosis:")
    for i, line in enumerate(src.split("\n"), 1):
        if "pat = r\"" in line:
            print("  line {}: {}".format(i, line.rstrip()))
    sys.exit(0 if "Already fixed" else 1)

print("[+] Found {} broken regex lines".format(count))

new_src = src.replace(BROKEN, FIXED)

try:
    ast.parse(new_src)
    print("[+] AST parse OK after fix")
except SyntaxError as e:
    print("ERROR: Fixed source still has syntax error: {}".format(e))
    sys.exit(1)

TARGET.write_text(new_src)
print("[+] Wrote fixed {}".format(TARGET))

# Verify the replacement
with TARGET.open() as f:
    new = f.read()
remaining = new.count(BROKEN)
fixed_count = new.count(FIXED)
print("[+] Remaining broken: {}".format(remaining))
print("[+] Fixed regex lines present: {}".format(fixed_count))

print()
print("Quick sanity check before re-launching:")
print("  grep -n 'pat = r' fdm_split_ratio_sweep.py")
print()
print("Then re-run smoke test (under 1 minute, no eval needed):")
print("  python -c \"import fdm_split_ratio_sweep\" && echo 'imports OK'")
print()
print("Then re-launch the sweep:")
print("  tmux kill-session -t sweep 2>/dev/null  # in case it's still running")
print("  rm -rf split_ratio_sweep_n200  # clear bad partial output")
print("  tmux new -s sweep \"python fdm_split_ratio_sweep.py \\")
print("      --n_eval 200 --seed 42 \\")
print("      --output_dir /workspace/FDM_IN_WEIGHTS/split_ratio_sweep_n200 \\")
print("      2>&1 | tee sweep_n200.log\"")
