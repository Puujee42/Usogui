"""Day 6: split by document, hold out one attack family, check for leakage.

  test_ood = ALL examples of documents whose attack twin uses the held-out family.
             The model never sees this family in training (out-of-distribution test).
  rest     = split by doc_id 80/10/10 into train/val/test, so one passage (and its twins)
             never appears in two splits.

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

print(f"{'split':<9}{'rows':>7}{'docs':>7}{'benign':>8}{'attack':>8}")
for name, r in report["splits"].items():
    lc = r["label_counts"]
    print(f"{name:<9}{r['rows']:>7}{r['docs']:>7}{lc.get(0, 0):>8}{lc.get(1, 0):>8}")
print("\nLEAKAGE CHECK:", "PASSED" if not problems else "FAILED")
for p in problems:
    print("  -", p)
