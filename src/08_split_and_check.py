"""Day 6 (v0.2): split by document, hold out one attack family, enforce template-level
and hard-negative holdout, and deduplicate deepset payloads across splits.

  test_ood = ALL examples of documents whose attack twin uses the held-out family.
             The model never sees this family in training (out-of-distribution test).
  rest     = split by doc_id 80/10/10 into train/val/test, so one passage (and its twins)
             never appears in two splits.

v0.2 additions:
  - Rows with template_test_only=True are moved out of train into test/val.
  - Rows with hn_test_only=True are moved out of train into test/val.
  - Deepset payloads that appear in test/val are removed from train (payload dedup).

Run from the repo root:  python src/08_split_and_check.py [--holdout phishing]
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, REPORTS, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", default="phishing", help="attack family kept out of train/val/test")
ap.add_argument("--val", type=float, default=0.1)
ap.add_argument("--test", type=float, default=0.1)
args = ap.parse_args()

df = pd.read_parquet(INTERIM / "formatted.parquet")

# --- ensure holdout columns exist (backward compat with v0.1 data) ------------
if "template_test_only" not in df.columns:
    df["template_test_only"] = False
if "hn_test_only" not in df.columns:
    df["hn_test_only"] = False

hold_docs = set(df.loc[df["attack_family"] == args.holdout, "doc_id"])
if not hold_docs:
    raise SystemExit(f"holdout family '{args.holdout}' not found. Available: {sorted(df.attack_family.unique())}")

ood = df[df["doc_id"].isin(hold_docs)]
rest = df[~df["doc_id"].isin(hold_docs)]

ids = np.array(sorted(rest["doc_id"].unique()))
np.random.default_rng(SEED).shuffle(ids)
n_val, n_test = int(len(ids) * args.val), int(len(ids) * args.test)
groups = {"val": ids[:n_val], "test": ids[n_val:n_val + n_test], "train": ids[n_val + n_test:]}
splits = {k: rest[rest["doc_id"].isin(v)].reset_index(drop=True) for k, v in groups.items()}
splits["test_ood"] = ood.reset_index(drop=True)

# --- v0.2: enforce template-level and hard-negative holdout -------------------
# Move rows with template_test_only=True or hn_test_only=True out of train
test_only_mask = splits["train"]["template_test_only"] | splits["train"]["hn_test_only"]
moved = splits["train"][test_only_mask]
if len(moved):
    # Append to test (these docs are already unique to train's doc_id pool,
    # so we move the whole doc to avoid doc_id leakage)
    moved_doc_ids = set(moved["doc_id"])
    move_all = splits["train"][splits["train"]["doc_id"].isin(moved_doc_ids)]
    splits["train"] = splits["train"][~splits["train"]["doc_id"].isin(moved_doc_ids)].reset_index(drop=True)
    splits["test"] = pd.concat([splits["test"], move_all], ignore_index=True)
    print(f"v0.2: moved {len(move_all)} rows ({len(moved_doc_ids)} docs) with test-only templates/HN from train to test")

# --- v0.2: deduplicate deepset payloads across splits -------------------------
# Payloads in test/val/ood should NOT also appear in train
def strip_prefix(s):
    """Remove style prefix from payloads for dedup. Note: 'P.S. ' has no colon."""
    return s.str.replace(r'^(?:Note|Important|Reminder|Update): |^P\.S\. ', '', regex=True)

test_val_payloads = set()
for name in ("test", "val", "test_ood"):
    ds_mask = splits[name]["attack_family"] == "deepset_real"
    test_val_payloads |= set(strip_prefix(splits[name].loc[ds_mask, "payload"]))

if test_val_payloads:
    tr = splits["train"]
    ds_train = tr["attack_family"] == "deepset_real"
    tr_payloads_stripped = strip_prefix(tr.loc[ds_train, "payload"])
    dup_mask = ds_train & tr_payloads_stripped.isin(test_val_payloads)
    n_dup = dup_mask.sum()
    if n_dup:
        # Remove the duplicate deepset attacks from train, but keep the doc's benign/HN rows
        splits["train"] = tr[~dup_mask].reset_index(drop=True)
        print(f"v0.2: removed {n_dup} deepset_real rows from train (payload also in test/val)")

# ---- leakage checks -----------------------------------------------------------
report, problems = {"holdout_family": args.holdout, "splits": {}}, []
names = list(splits)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        for col in ("doc_id", "text", "data"):
            n = len(set(splits[a][col]) & set(splits[b][col]))
            if n:
                problems.append(f"{n} shared '{col}' values between {a} and {b}")
for name in ("train", "val", "test"):
    if (splits[name]["attack_family"] == args.holdout).any():
        problems.append(f"held-out family '{args.holdout}' leaked into {name}")

# v0.2: check holdout flags not remaining in train
for col in ("template_test_only", "hn_test_only"):
    n_leak = int(splits["train"][col].sum())
    if n_leak:
        problems.append(f"{n_leak} rows with {col}=True remain in train")

# v0.2: check payload leakage for deepset_real
tr_ds = strip_prefix(splits["train"].loc[splits["train"]["attack_family"] == "deepset_real", "payload"])
for name in ("test", "val"):
    te_ds = strip_prefix(splits[name].loc[splits[name]["attack_family"] == "deepset_real", "payload"])
    n_leak = len(set(tr_ds) & set(te_ds))
    if n_leak:
        problems.append(f"{n_leak} deepset_real payloads shared between train and {name}")

for name, s in splits.items():
    report["splits"][name] = {
        "rows": len(s), "docs": int(s["doc_id"].nunique()),
        "label_counts": {int(k): int(v) for k, v in s["label"].value_counts().items()},
        "attack_families": {k: int(v) for k, v in s.loc[s.label == 1, "attack_family"].value_counts().items()},
        "hard_negatives": {k: int(v) for k, v in s.loc[s.variant == "hard_negative", "hn_type"].value_counts().items()},
    }
    s.to_parquet(PROCESSED / f"{name}.parquet", index=False)

report["problems"] = problems
(REPORTS / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\n{'split':<9}{'rows':>7}{'docs':>7}{'benign':>8}{'attack':>8}")
for name, r in report["splits"].items():
    lc = r["label_counts"]
    print(f"{name:<9}{r['rows']:>7}{r['docs']:>7}{lc.get(0, 0):>8}{lc.get(1, 0):>8}")
print("\nLEAKAGE CHECK:", "PASSED" if not problems else "FAILED")
for p in problems:
    print("  -", p)
