"""Measure tokenized lengths using ModernBERT-base tokenizer across splits.

Checks length percentiles on Frege formatted text, and checks truncation risk
specifically for attack rows with position == 'end'.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SPLITS = ["train", "val", "val_template", "test_template", "test_real", "test_ood"]
MODEL_ID = "answerdotai/ModernBERT-base"


def main():
    print(f"Loading tokenizer: {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\nToken length distribution (Frege format) per split:")
    header = (
        f"{'Split':<15} {'Rows':>6} {'Min':>6} {'Median':>8} {'P95':>6} {'P99':>6} {'Max':>6} "
        f"{'EndAttacks':>12} {'>512':>6} {'% > 512':>9}"
    )
    print(header)
    print("-" * len(header))

    total_end_attacks = 0
    total_end_attacks_over_512 = 0

    for name in SPLITS:
        path = PROCESSED / f"{name}.parquet"
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        df = pd.read_parquet(path)

        # Batch tokenization
        enc = tokenizer(df["text"].tolist(), add_special_tokens=True, truncation=False, return_attention_mask=False)
        lengths = np.array([len(ids) for ids in enc["input_ids"]])
        df["token_len"] = lengths

        c_min = int(np.min(lengths))
        c_med = float(np.median(lengths))
        c_p95 = float(np.percentile(lengths, 95))
        c_p99 = float(np.percentile(lengths, 99))
        c_max = int(np.max(lengths))

        end_attacks = df[(df["label"] == 1) & (df["position"] == "end")]
        n_end = len(end_attacks)
        n_over_512 = int((end_attacks["token_len"] > 512).sum())
        pct_over = (n_over_512 / n_end * 100.0) if n_end > 0 else 0.0

        total_end_attacks += n_end
        total_end_attacks_over_512 += n_over_512

        print(
            f"{name:<15} {len(df):>6} {c_min:>6} {c_med:>8.1f} {c_p95:>6.1f} {c_p99:>6.1f} {c_max:>6} "
            f"{n_end:>12} {n_over_512:>6} {pct_over:>8.2f}%"
        )

    print("-" * len(header))
    total_pct = (total_end_attacks_over_512 / total_end_attacks * 100.0) if total_end_attacks > 0 else 0.0
    print(
        f"{'TOTAL':<15} {'':>6} {'':>6} {'':>8} {'':>6} {'':>6} {'':>6} "
        f"{total_end_attacks:>12} {total_end_attacks_over_512:>6} {total_pct:>8.2f}%"
    )

    print(
        f"\nExact % of attack rows with position == 'end' exceeding 512 tokens: "
        f"{total_end_attacks_over_512}/{total_end_attacks} = {total_pct:.4f}%\n"
    )


if __name__ == "__main__":
    main()
