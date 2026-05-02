#!/usr/bin/env python3
"""Compute Wilson 95% CIs and produce LaTeX cells from sweep_results_merged.json."""

import json
import math
from pathlib import Path


def wilson_half_margin(k, n, alpha=0.05):
    if n == 0:
        return 0.0
    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lower = max(0.0, centre - half)
    upper = min(1.0, centre + half)
    return max(p - lower, upper - p)


def fmt_cell(k, n):
    if n == 0:
        return "---"
    p = k / n
    hm = wilson_half_margin(k, n)
    return f"{p*100:.1f}\\% $\\pm$ {hm*100:.1f}"


def main():
    path = Path("split_ratio_sweep_n200/sweep_results_merged.json")
    if not path.exists():
        print(f"ERROR: {path} not found")
        return

    results = json.load(path.open())

    print("=" * 70)
    print("RAW DATA SUMMARY")
    print("=" * 70)
    print(f"{'Param':>5} {'Ctx':>5} {'A k/n':>15} {'B k/n':>15} {'Joint k/n':>15}")
    print("-" * 70)
    for r in results:
        if r["n_param"] == 0:
            a_str = "---"
            slot_k = r.get("slot_correct", 0)
            slot_n = r.get("slot_total", 0)
            sample_k = r.get("sample_all_correct", 0)
            sample_n = r.get("n_samples", 0)
            b_str = f"{slot_k}/{slot_n}"
            j_str = f"{sample_k}/{sample_n}"
        else:
            a_k = r.get("block_a_slot_correct", 0)
            a_n = r.get("block_a_slot_total", 0)
            b_k = r.get("block_b_slot_correct", 0)
            b_n = r.get("block_b_slot_total", 0)
            j_k = r.get("sample_all_all_correct", 0)
            j_n = r.get("n_samples", 0)
            a_str = f"{a_k}/{a_n}"
            b_str = f"{b_k}/{b_n}" if b_n > 0 else "---"
            j_str = f"{j_k}/{j_n}"
        print(f"{r['n_param']:>5} {r['n_ctx']:>5} {a_str:>15} {b_str:>15} {j_str:>15}")

    print()
    print("=" * 70)
    print("LATEX TABLE 5 BODY")
    print("=" * 70)
    for r in results:
        n_param = r["n_param"]
        n_ctx = r["n_ctx"]

        if n_param == 0:
            slot_k = r.get("slot_correct", 0)
            slot_n = r.get("slot_total", 0)
            j_k = r.get("sample_all_correct", 0)
            j_n = r.get("n_samples", 0)
            a_cell, b_cell = "---", fmt_cell(slot_k, slot_n)
            joint_cell = fmt_cell(j_k, j_n)
        elif n_ctx == 0:
            a_k = r.get("block_a_slot_correct", 0)
            a_n = r.get("block_a_slot_total", 0)
            j_k = r.get("sample_all_all_correct", 0)
            j_n = r.get("n_samples", 0)
            a_cell, b_cell = fmt_cell(a_k, a_n), "---"
            joint_cell = fmt_cell(j_k, j_n)
        else:
            a_k = r.get("block_a_slot_correct", 0)
            a_n = r.get("block_a_slot_total", 0)
            b_k = r.get("block_b_slot_correct", 0)
            b_n = r.get("block_b_slot_total", 0)
            j_k = r.get("sample_all_all_correct", 0)
            j_n = r.get("n_samples", 0)
            a_cell = fmt_cell(a_k, a_n)
            b_cell = fmt_cell(b_k, b_n)
            joint_cell = fmt_cell(j_k, j_n)

        if n_param == 28:
            print("\\midrule")
        print(f"{n_param:>2} & {n_ctx:>2} & {a_cell} & {b_cell} & {joint_cell} \\\\")

    print()
    print("=" * 70)
    print("INDEPENDENCE CHECK (joint vs slot^32)")
    print("=" * 70)
    print(f"{'Ratio':>7} {'slot':>8} {'slot^32':>10} {'actual':>10} {'ratio':>10}")
    print("-" * 70)
    for r in results:
        slot = r.get("all_acc_slot", 0)
        joint = r.get("all_acc_joint", 0)
        slot_pow_32 = slot ** 32
        ratio_str = f"{joint / slot_pow_32:.2f}x" if slot_pow_32 > 1e-10 else "n/a"
        print(f"{r['n_param']:>2}/{r['n_ctx']:>2} {slot*100:>7.1f}% "
              f"{slot_pow_32*100:>9.1f}% {joint*100:>9.1f}% {ratio_str:>10}")


if __name__ == "__main__":
    main()
