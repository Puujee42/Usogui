"""Day 6: re-run the shortcut check on YOUR dataset.

Models (all trained on train, scored on test and on test_ood):
  tfidf(data)          - words in the data (the injected sentence is in here, so high is expected)
  tfidf(instruction)   - SANITY CHECK: twins share the same instruction -> should be near chance
  length only          - should be near chance because twins differ by one short sentence
Also reported: false-positive rate on hard negatives, and recall per attack family.

Run from the repo root:  python src/09_shortcut_check_v2.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from common import FIGURES, PROCESSED, REPORTS, SEED

tr, val, te, ood = (pd.read_parquet(PROCESSED / f"{n}.parquet") for n in ("train", "val", "test", "test_ood"))


def length_feats(s):
    s = pd.Series(s)
    return np.c_[np.log1p(s.str.len()), np.log1p(s.str.split().str.len())]


def fit_text(col):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(vec.fit_transform(tr[col]), tr["label"])
    return lambda df: clf.predict_proba(vec.transform(df[col]))[:, 1], (vec, clf)


def fit_len():
    clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(length_feats(tr["data"]), tr["label"])
    return lambda df: clf.predict_proba(length_feats(df["data"]))[:, 1], None


models = {"tfidf(data)": fit_text("data"), "tfidf(instruction)": fit_text("instruction"), "length only": fit_len()}

# --- Threshold tuning on validation ------------------------------------------
score_data = models["tfidf(data)"][0]
val_prob = score_data(val)
thresholds = np.arange(0.10, 0.91, 0.01)
best_f1, best_thresh = -1.0, 0.5
for t in thresholds:
    _, _, f1_t, _ = precision_recall_fscore_support(val["label"], (val_prob >= t).astype(int),
                                                    average="binary", zero_division=0)
    if f1_t > best_f1:
        best_f1, best_thresh = f1_t, t
print(f"Optimal threshold (tuned on val): {best_thresh:.2f}")

# --- Main results table (both fixed 0.5 and tuned threshold) -----------------
rows = []
for mname, (score, _) in models.items():
    for sname, df in (("test", te), ("test_ood", ood)):
        prob = score(df)
        # Fixed 0.5 threshold
        pred_05 = (prob >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(df["label"], pred_05, average="binary", zero_division=0)
        auc = roc_auc_score(df["label"], prob) if df["label"].nunique() >= 2 else float("nan")
        rows.append((mname, sname, 0.5, accuracy_score(df["label"], pred_05), p, r, f1, auc))
        # Tuned threshold (only for tfidf(data))
        if mname == "tfidf(data)":
            pred_t = (prob >= best_thresh).astype(int)
            p_t, r_t, f1_t, _ = precision_recall_fscore_support(df["label"], pred_t,
                                                                  average="binary", zero_division=0)
            rows.append((mname, sname, round(best_thresh, 2), accuracy_score(df["label"], pred_t),
                          p_t, r_t, f1_t, auc))
res = pd.DataFrame(rows, columns=["model", "split", "threshold", "accuracy", "precision",
                                   "recall", "f1", "roc_auc"]).round(3)
res.to_csv(REPORTS / "shortcut_check_v2.csv", index=False)
print(res.to_string(index=False))

# --- False-positive rate on hard negatives ------------------------------------
score = models["tfidf(data)"][0]
hn = te[te["variant"] == "hard_negative"]
if len(hn):
    fpr = (score(hn) >= 0.5).mean()
    print(f"\nFalse-positive rate on hard negatives (tfidf(data), test, thresh=0.5): {fpr:.3f}")
    for t, g in hn.groupby("hn_type"):
        print(f"   {t:<18} FPR = {(score(g) >= 0.5).mean():.3f}  (n={len(g)})")

    fpr_t = (score(hn) >= best_thresh).mean()
    print(f"\nFalse-positive rate on hard negatives (tfidf(data), test, thresh={best_thresh:.2f}): {fpr_t:.3f}")
    for t, g in hn.groupby("hn_type"):
        print(f"   {t:<18} FPR = {(score(g) >= best_thresh).mean():.3f}  (n={len(g)})")

# --- Per-family recall (fixed 0.5) -------------------------------------------
print("\nRecall per attack family (tfidf(data), threshold=0.50):")
for sname, df in (("test", te), ("test_ood", ood)):
    for fam, g in df[df["label"] == 1].groupby("attack_family"):
        print(f"   {sname:<9}{fam:<18} recall = {(score(g) >= 0.5).mean():.3f}  (n={len(g)})")

# --- Per-family recall (tuned threshold) --------------------------------------
print(f"\nRecall per attack family (tfidf(data), threshold={best_thresh:.2f}):")
for sname, df in (("test", te), ("test_ood", ood)):
    for fam, g in df[df["label"] == 1].groupby("attack_family"):
        print(f"   {sname:<9}{fam:<18} recall = {(score(g) >= best_thresh).mean():.3f}  (n={len(g)})")

# --- Per-source recall (tuned threshold) --------------------------------------
print(f"\nRecall per source (tfidf(data), tuned threshold={best_thresh:.2f}):")
for sname, df in (("test", te), ("test_ood", ood)):
    for src, g in df[df["label"] == 1].groupby("source"):
        print(f"   {sname:<9}{src:<25} recall = {(score(g) >= best_thresh).mean():.3f}  (n={len(g)})")

# --- Top features -------------------------------------------------------------
vec, clf = models["tfidf(data)"][1]
names, coef = np.array(vec.get_feature_names_out()), clf.coef_[0]
order = np.argsort(coef)
print("\nTop words -> BENIGN:", ", ".join(names[order[:12]]))
print("Top words -> ATTACK:", ", ".join(names[order[-12:]][::-1]))
fig, ax = plt.subplots(figsize=(8, 5))
idx = np.r_[order[:12], order[-12:]]
ax.barh(names[idx], coef[idx], color=["tab:blue"] * 12 + ["tab:red"] * 12)
ax.set_title("tfidf(data): most influential words (blue = benign, red = attack)")
fig.tight_layout()
fig.savefig(FIGURES / "shortcut_v2_features.png", dpi=150)

s = res.set_index(["model", "split", "threshold"])
print("\nHOW TO READ THIS")
print(f"- instruction-only F1 = {s.loc[('tfidf(instruction)', 'test', 0.5), 'f1']} (should be near chance; if high, instructions leak the label)")
print(f"- length-only F1      = {s.loc[('length only', 'test', 0.5), 'f1']} (should be near chance; if high, length is a shortcut)")
print(f"- tfidf(data) F1: test {s.loc[('tfidf(data)', 'test', 0.5), 'f1']} vs OOD {s.loc[('tfidf(data)', 'test_ood', 0.5), 'f1']} (a big drop = it memorised families)")
