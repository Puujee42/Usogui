"""Independent evaluation on REAL attacks (test_real), reproducible without trusting any summary.

What it does
  1. trains the TF-IDF + logistic-regression baseline on train ONLY,
  2. picks a threshold on val ONLY (best F1) - or uses --threshold so you can compare with another script,
  3. scores test_real and reports, at that threshold:
       - AUC, row-level recall, FPR,
       - recall PER DISTINCT SENTENCE with a 95% bootstrap interval (resampling sentences, not rows),
       - the real sentences it misses most, so you can read what real attacks look like.

Why per sentence: one real sentence is inserted into several passages, so rows are NOT independent.
360 rows may be only ~114 distinct sentences; the honest sample size is the number of sentences.

Run from the repo root:  python src/10_real_eval.py [--threshold 0.42] [--boot 2000]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
SEED = 42
_PREFIX = re.compile(r"^\s*(?:note|important|p\.\s?s\.?|reminder|update)\s*[:\-\u2013\u2014]?\s*", re.IGNORECASE)

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, default=None, help="use this threshold instead of tuning on val")
ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples over distinct sentences")
args = ap.parse_args()


def norm(s) -> str:
    s = str(s)
    while True:
        t = _PREFIX.sub("", s, count=1)
        if t == s:
            break
        s = t
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".!?;: ")


tr, va, real = (pd.read_parquet(PROCESSED / f"{n}.parquet") for n in ("train", "val", "test_real"))
assert not (tr["attack_family"] == "deepset_real").any() and not (va["attack_family"] == "deepset_real").any(), \
    "deepset_real rows found in train/val - run the leakage audit first"

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(vec.fit_transform(tr["data"]), tr["label"])
score = lambda d: clf.predict_proba(vec.transform(d["data"]))[:, 1]  # noqa: E731

sv = score(va)
p, r, t = precision_recall_curve(va["label"], sv)
thr_f1 = float(t[np.argmax(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12))])
thr = thr_f1 if args.threshold is None else args.threshold
print(f"threshold: {thr:.3f} ({'best F1 on val' if args.threshold is None else 'given'}; best-F1-on-val would be {thr_f1:.3f})")

s, y = score(real), real["label"].to_numpy()
pred = s > thr
att = real[y == 1].copy()
att["score"], att["hit"], att["sent"] = s[y == 1], pred[y == 1], att["payload"].map(norm)

per = att.groupby("sent").agg(rate=("hit", "mean"), rows=("hit", "size"), score=("score", "mean")).sort_values("score")
rng = np.random.default_rng(SEED)
rates = per["rate"].to_numpy()
boots = [rng.choice(rates, size=len(rates), replace=True).mean() for _ in range(args.boot)]
lo, hi = np.percentile(boots, [2.5, 97.5])

print(f"\ntest_real: {len(real)} rows | {int((y == 1).sum())} attack rows = {len(per)} DISTINCT sentences | {int((y == 0).sum())} benign rows")
print(f"AUC                         = {roc_auc_score(y, s):.3f}")
print(f"row-level recall            = {att['hit'].mean():.3f}   (rows are not independent)")
print(f"per-sentence recall (macro) = {per['rate'].mean():.3f}   95% bootstrap CI over sentences: [{lo:.3f}, {hi:.3f}]")
print(f"sentences never detected    = {(per['rate'] == 0).sum()} of {len(per)}   |   always detected = {(per['rate'] == 1).sum()}")
print(f"FPR on benign rows          = {pred[y == 0].mean():.4f}")
if "hn_type" in real.columns:
    for k, g in real[y == 0].assign(pred=pred[y == 0]).groupby("hn_type"):
        print(f"   FPR[{k}] = {g['pred'].mean():.4f} (n={len(g)})")

print("\nREAL SENTENCES THE BASELINE MISSES MOST (lowest mean score):")
for sent, row in per.head(8).iterrows():
    print(f"  score {row['score']:.2f} | rows {int(row['rows'])} | {sent[:110]}")
print("\nREAL SENTENCES IT CATCHES MOST CONFIDENTLY:")
for sent, row in per.tail(4).iterrows():
    print(f"  score {row['score']:.2f} | rows {int(row['rows'])} | {sent[:110]}")
print("\nCAVEATS: the benign side of test_real is synthetic (squad passages), so this FPR is NOT a real-world FPR;"
      "\nonly the attack side is real. A real-world FPR needs real benign text (emails, web pages, user prompts).")
