"""Split by DOCUMENT into 6 sets and check for leakage (v0.4, works with both pipeline schemas).

A doc_id and all its variants (benign / attack / hard negative) always stay together. Priority order:
  1. test_real      docs whose attack twin is a REAL deepset sentence (evaluation-only: zero-shot synthetic -> real)
  2. test_ood       docs whose attack twin belongs to the held-out FAMILY (default: phishing)
  3. test_template  docs with ANY row flagged as reserved wording (held-out attack template or harmless sentence)
  4. train/val/test the remaining docs, ~80/10/10 by number of documents

Flag columns understood: heldout_template (v0.2 schema) and template_test_only / hn_test_only (agent schema).
If NONE exists the script refuses to run, because a silent loss of the held-out-wording rule is exactly the
bug this version fixes. Pass --allow-no-template-holdout to run without it on purpose.

Run from the repo root:  python src/08_split_and_check.py [--holdout phishing]
tests/test_no_leakage.py and tests/mutation_check.py audit the result independently.
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
ap.add_argument("--allow-no-template-holdout", action="store_true")
args = ap.parse_args()

df = pd.read_parquet(INTERIM / "formatted.parquet")
if not (df["attack_family"] == args.holdout).any():
    raise SystemExit(f"holdout family '{args.holdout}' not found. Available: {sorted(df.attack_family.unique())}")

FLAGS = [c for c in ("heldout_template", "template_test_only", "hn_test_only") if c in df.columns]
if not FLAGS and not args.allow_no_template_holdout:
    raise SystemExit("No flag column (heldout_template / template_test_only / hn_test_only) found, so the held-out-wording "
                     "rule cannot be applied. Fix script 05/06 or pass --allow-no-template-holdout.")
df["_flag"] = df[FLAGS].astype(bool).any(axis=1) if FLAGS else False

real_docs = set(df.loc[df["attack_family"] == "deepset_real", "doc_id"])
ood_docs = set(df.loc[df["attack_family"] == args.holdout, "doc_id"]) - real_docs
tmpl_docs = set(df.loc[df["_flag"], "doc_id"]) - real_docs - ood_docs
rest = df[~df["doc_id"].isin(real_docs | ood_docs | tmpl_docs)]

ids = np.array(sorted(rest["doc_id"].unique()))
np.random.default_rng(SEED).shuffle(ids)
n_val, n_test = int(len(ids) * args.val), int(len(ids) * args.test)
groups = {"val": ids[:n_val], "test": ids[n_val:n_val + n_test], "train": ids[n_val + n_test:]}
splits = {k: rest[rest["doc_id"].isin(v)] for k, v in groups.items()}
splits["test_template"] = df[df["doc_id"].isin(tmpl_docs)]
splits["test_real"] = df[df["doc_id"].isin(real_docs)]
splits["test_ood"] = df[df["doc_id"].isin(ood_docs)]
splits = {k: v.drop(columns="_flag").reset_index(drop=True) for k, v in splits.items() if len(v)}

# ---- leakage checks (independent audit: tests/test_no_leakage.py) ---------------------------------
problems, names = [], list(splits)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        for col in ("doc_id", "text", "data"):
            n = len(set(splits[a][col]) & set(splits[b][col]))
            if n:
                problems.append(f"{n} shared '{col}' values between {a} and {b}")
for name in ("train", "val", "test"):
    s = splits[name]
    if (s["attack_family"] == args.holdout).any():
        problems.append(f"held-out family '{args.holdout}' leaked into {name}")
    for c in FLAGS:
        n = int(s[c].astype(bool).sum())
        if n:
            problems.append(f"{n} rows flagged '{c}' are in {name}")
for name, s in splits.items():
    if name != "test_real" and (s["attack_family"] == "deepset_real").any():
        problems.append(f"deepset_real rows found in {name}")

report = {"holdout_family": args.holdout, "flag_columns": FLAGS, "splits": {}, "problems": problems}
print(f"{'split':<14}{'rows':>6}{'docs':>6}{'benign':>8}{'attack':>8}   flagged rows")
for name, s in splits.items():
    lc = {int(k): int(v) for k, v in s["label"].value_counts().items()}
    fl = {c: int(s[c].astype(bool).sum()) for c in FLAGS}
    report["splits"][name] = {"rows": len(s), "docs": int(s["doc_id"].nunique()), "label_counts": lc, "flagged_rows": fl,
                              "attack_families": {k: int(v) for k, v in s.loc[s.label == 1, "attack_family"].value_counts().items()}}
    s.to_parquet(PROCESSED / f"{name}.parquet", index=False)
    print(f"{name:<14}{len(s):>6}{s['doc_id'].nunique():>6}{lc.get(0, 0):>8}{lc.get(1, 0):>8}   {fl}")
(REPORTS / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nLEAKAGE CHECK:", "PASSED" if not problems else "FAILED")
for p in problems:
    print("  -", p)