"""Strict ground-truth audit of the deepset_real split implementation.

Verifies:
1. Zero 'deepset_real' leakage into train, val, test, or test_ood.
2. Strict doc_id separation between test_real and all other splits.
3. Benign twins correctly accompanied deepset_real attacks into test_real.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def main():
    print("=" * 60)
    print("STARTING STRICT GROUND-TRUTH AUDIT: deepset_real SPLITS")
    print("=" * 60)

    # 1. Load splits using pandas
    split_names = ["train", "val", "test", "test_ood", "test_real"]
    splits = {}
    print("\n[STEP 1] Loading Parquet splits from data/processed/ ...")
    for name in split_names:
        file_path = DATA_DIR / f"{name}.parquet"
        assert file_path.exists(), f"Missing split file: {file_path}"
        splits[name] = pd.read_parquet(file_path)
        print(f"  - Loaded {name:<10}: {len(splits[name]):>6} rows, {splits[name]['doc_id'].nunique():>5} unique doc_ids")

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    test_ood = splits["test_ood"]
    test_real = splits["test_real"]

    # 2. Check 1: Zero deepset_real leakage
    print("\n[STEP 2] Asserting zero 'deepset_real' leakage into other splits ...")
    for name, df in [("train", train), ("val", val), ("test", test), ("test_ood", test_ood)]:
        count = (df["attack_family"] == "deepset_real").sum()
        print(f"  - Occurrences of 'deepset_real' in {name:<10} 'attack_family': {count}")
        assert count == 0, f"LEAKAGE DETECTED: 'deepset_real' appeared {count} times in {name}!"
    print("  -> CONFIRMED: Zero 'deepset_real' leakage across train, val, test, and test_ood.")

    # 3. Check 2: Strict doc_id separation
    print("\n[STEP 3] Asserting strict doc_id separation between test_real and other splits ...")
    real_doc_ids = set(test_real["doc_id"].unique())
    print(f"  - Unique doc_ids in test_real: {len(real_doc_ids)}")
    for name, df in [("train", train), ("val", val), ("test", test), ("test_ood", test_ood)]:
        split_doc_ids = set(df["doc_id"].unique())
        shared = real_doc_ids.intersection(split_doc_ids)
        print(f"  - Intersection(test_real, {name:<10}): {len(shared)} shared doc_ids")
        assert len(shared) == 0, f"DOC_ID OVERLAP DETECTED: {len(shared)} doc_ids shared between test_real and {name}: {shared}"
    print("  -> CONFIRMED: test_real doc_ids are strictly disjoint from all other splits.")

    # 4. Check 3: Benign twins moved correctly
    print("\n[STEP 4] Asserting benign twins moved correctly into test_real ...")
    benign_rows = test_real[test_real["label"] == 0]
    attack_rows = test_real[test_real["attack_family"] == "deepset_real"]
    n_benign = len(benign_rows)
    n_attack = len(attack_rows)
    print(f"  - Total rows in test_real: {len(test_real)}")
    print(f"  - Benign rows (label == 0) in test_real: {n_benign}")
    print(f"  - deepset_real attack rows in test_real: {n_attack}")

    assert n_benign > 0, "FAILED: test_real contains 0 benign rows (label == 0)!"
    assert n_attack > 0, "FAILED: test_real contains 0 deepset_real attack rows!"

    benign_doc_ids = set(benign_rows["doc_id"].unique())
    attack_doc_ids = set(attack_rows["doc_id"].unique())
    print(f"  - Unique doc_ids among benign rows in test_real: {len(benign_doc_ids)}")
    print(f"  - Unique doc_ids among deepset_real rows in test_real: {len(attack_doc_ids)}")

    diff_benign_attack = benign_doc_ids - attack_doc_ids
    diff_attack_benign = attack_doc_ids - benign_doc_ids
    print(f"  - Benign doc_ids not in attack doc_ids: {len(diff_benign_attack)}")
    print(f"  - Attack doc_ids not in benign doc_ids: {len(diff_attack_benign)}")

    assert benign_doc_ids == attack_doc_ids, (
        f"MISMATCH: benign and deepset_real doc_ids do not match! "
        f"benign-only: {len(diff_benign_attack)}, attack-only: {len(diff_attack_benign)}"
    )
    print("  -> CONFIRMED: Benign rows share the exact same doc_ids as deepset_real attacks in test_real.")

    print("\n" + "=" * 60)
    print("ALL AUDIT ASSERTIONS PASSED (100% VERIFIED)")
    print("=" * 60)


if __name__ == "__main__":
    main()
