"""TF-IDF baseline on EVERY split, with a threshold policy that is not arbitrary.   (independent of script 09)

Splits scored if present: test, test_template (unseen wording), test_real (real deepset sentences), test_ood (unseen family).
Thresholds come from VAL only:
  0.5            : the default
  val-F1 (mid)   : middle of the plateau of best-F1 thresholds (if val is easy, F1 is flat over a wide range,
                   so the 'argmax' is arbitrary - this script prints the plateau so you can see it)
  val-FPR<=1%    : 99th percentile of benign val scores  (the Week 20 target: false-positive rate under 1%)

Run from the repo root:  python src/11_baseline_all_splits.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
SEED = 42
tr, va = (pd.read_parquet(PROCESSED / f"{n}.parquet") for n in ("train", "val"))
EVAL = {n: pd.read_parquet(PROCESSED / f"{n}.parquet") for n in ("test", "test_template", "test_real", "test_ood")
        if (PROCESSED / f"{n}.parquet").exists()}


def fit(col):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(vec.fit_transform(tr[col]), tr["label"])
    return (lambda d: clf.predict_proba(vec.transform(d[col]))[:, 1]), vec, clf


def length_feats(s):
    s = pd.Series(s)
    return np.c_[np.log1p(s.str.len()), np.log1p(s.str.split().str.len())]


score, vec, clf = fit("data")
s_instr = fit("instruction")[0]
len_clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(length_feats(tr["data"]), tr["label"])
s_len = lambda d: len_clf.predict_proba(length_feats(d["data"]))[:, 1]  # noqa: E731


def f1_rec_fpr(y, s, t):
    pred = s > t
    tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), ((~pred) & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1), pred[y == 1].mean(), pred[y == 0].mean()


# ---- thresholds from VAL only -----------------------------------------------------------------
sv, yv = score(va), va["label"].to_numpy()
grid = np.linspace(0.02, 0.98, 97)
f1s = np.array([f1_rec_fpr(yv, sv, t)[0] for t in grid])
plateau = grid[f1s >= f1s.max() - 1e-3]
thr = {"0.5": 0.5, "val-F1 (mid)": float(np.median(plateau)), "val-FPR<=1%": float(np.quantile(sv[yv == 0], 0.99))}
print(f"val best F1 = {f1s.max():.3f}; thresholds within 0.001 of it span [{plateau.min():.2f}, {plateau.max():.2f}]")
if plateau.max() - plateau.min() > 0.3:
    print("  WARNING: val is (nearly) saturated - any threshold in that range is 'optimal', so a val-F1 threshold is arbitrary.\n"
          "  Use the FPR policy, and build a val set that contains unseen wording (val_template).")
print("thresholds:", {k: round(v, 3) for k, v in thr.items()}, "\n")

# ---- table ---------------------------------------------------------------------------------------
rows = []
for name, d in EVAL.items():
    y, s = d["label"].to_numpy(), score(d)
    for pn, t in thr.items():
        f1, rec, fpr = f1_rec_fpr(y, s, t)
        rows.append((name, len(d), round(roc_auc_score(y, s), 3), pn, round(t, 3), round(f1, 3), round(rec, 3), round(fpr, 4)))
res = pd.DataFrame(rows, columns=["split", "n", "AUC", "policy", "thr", "F1", "recall", "FPR"])
print(res.to_string(index=False))

T = thr["val-FPR<=1%"]
print(f"\nRECALL PER ATTACK FAMILY at the FPR<=1% threshold ({T:.3f}):")
for name, d in EVAL.items():
    for fam, g in d[d["label"] == 1].groupby("attack_family"):
        print(f"   {name:<14}{fam:<17}recall = {(score(g) > T).mean():.3f}  (n={len(g)})")

print("\nFALSE-POSITIVE RATE ON HARD NEGATIVES (harmless but suspicious-looking):")
for name, d in EVAL.items():
    hn = d[(d["label"] == 0) & (d["hn_type"] != "none")] if "hn_type" in d.columns else d.iloc[0:0]
    for k, g in hn.groupby("hn_type"):
        print(f"   {name:<14}{k:<19}FPR@0.5 = {(score(g) > 0.5).mean():.3f}   FPR@val-1% = {(score(g) > T).mean():.3f}  (n={len(g)})")

print("\nSANITY BASELINES (AUC; ~0.5 = no shortcut):")
for name, d in EVAL.items():
    print(f"   {name:<14} instruction-only = {roc_auc_score(d['label'], s_instr(d)):.3f}   length-only = {roc_auc_score(d['label'], s_len(d)):.3f}")

names, coef = np.array(vec.get_feature_names_out()), clf.coef_[0]
order = np.argsort(coef)
print("\nTop words -> BENIGN:", ", ".join(names[order[:12]]))
print("Top words -> ATTACK:", ", ".join(names[order[-12:]][::-1]))
print("\nREAD: test (seen wording) is expected to be ~perfect for TF-IDF; the evidence is test_template, test_real and test_ood.")