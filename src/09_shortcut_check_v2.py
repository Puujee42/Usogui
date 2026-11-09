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

tr, te, ood = (pd.read_parquet(PROCESSED / f"{n}.parquet") for n in ("train", "test", "test_ood"))


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

rows = []
for mname, (score, _) in models.items():
    for sname, df in (("test", te), ("test_ood", ood)):
        prob = score(df)
        pred = (prob >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(df["label"], pred, average="binary", zero_division=0)
        rows.append((mname, sname, accuracy_score(df["label"], pred), p, r, f1, roc_auc_score(df["label"], prob)))
res = pd.DataFrame(rows, columns=["model", "split", "accuracy", "precision", "recall", "f1", "roc_auc"]).round(3)
res.to_csv(REPORTS / "shortcut_check_v2.csv", index=False)
print(res.to_string(index=False))

score = models["tfidf(data)"][0]
hn = te[te["variant"] == "hard_negative"]
if len(hn):
    fpr = (score(hn) >= 0.5).mean()
    print(f"\nFalse-positive rate on hard negatives (tfidf(data), test): {fpr:.3f}")
    for t, g in hn.groupby("hn_type"):
        print(f"   {t:<18} FPR = {(score(g) >= 0.5).mean():.3f}  (n={len(g)})")

print("\nRecall per attack family (tfidf(data)):")
for sname, df in (("test", te), ("test_ood", ood)):
    for fam, g in df[df["label"] == 1].groupby("attack_family"):
        print(f"   {sname:<9}{fam:<18} recall = {(score(g) >= 0.5).mean():.3f}  (n={len(g)})")

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

s = res.set_index(["model", "split"])
print("\nHOW TO READ THIS")
print(f"- instruction-only F1 = {s.loc[('tfidf(instruction)', 'test'), 'f1']} (should be near chance; if high, instructions leak the label)")
print(f"- length-only F1      = {s.loc[('length only', 'test'), 'f1']} (should be near chance; if high, length is a shortcut)")
print(f"- tfidf(data) F1: test {s.loc[('tfidf(data)', 'test'), 'f1']} vs OOD {s.loc[('tfidf(data)', 'test_ood'), 'f1']} (a big drop = it memorised families)")
