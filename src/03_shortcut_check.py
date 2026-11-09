"""Day 5: shortcut check.

Question: can a trivial model separate benign from malicious text for the WRONG reason
(source, length or style instead of actual injection content)?

Experiments
  A1  squad contexts   (benign)  vs deepset injections   <- different sources
  A2  squad questions  (benign)  vs deepset injections   <- different sources, similar length
  B   deepset label 0  vs deepset label 1                <- same source (fair comparison)
Each is run with TF-IDF+LogReg and with a length-only model.

Read the result:  A >> B  means the classes differ by source/style, not by injection.

Run from the repo root:  python src/03_shortcut_check.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
RAW = Path("data/raw")
for d in ("figures", "reports"):
    Path(d).mkdir(exist_ok=True)

squad = pd.read_parquet(RAW / "squad_v2_train.parquet")
parts = {p.stem.split("_")[-1]: pd.read_parquet(p) for p in RAW.glob("prompt_injections_*.parquet")}
inj_all = pd.concat(parts.values(), ignore_index=True)
attacks = inj_all.loc[inj_all.label == 1, "text"].tolist()
n = len(attacks)

contexts = squad["context"].drop_duplicates().sample(n, random_state=SEED).tolist()
questions = squad["question"].drop_duplicates().sample(n, random_state=SEED).tolist()


def split_ab(benign):
    X = benign + attacks
    y = [0] * len(benign) + [1] * len(attacks)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def split_b():
    tr, te = parts["train"], parts["test"]
    return tr.text.tolist(), te.text.tolist(), tr.label.tolist(), te.label.tolist()


def length_feats(texts):
    s = pd.Series(texts)
    return np.c_[np.log1p(s.str.len()), np.log1p(s.str.split().str.len())]


def evaluate(name, Xtr, Xte, ytr, yte):
    rows = []
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(A, ytr)
    pred, prob = clf.predict(B), clf.predict_proba(B)[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    rows.append((name, "tfidf+logreg", accuracy_score(yte, pred), p, r, f1, roc_auc_score(yte, prob)))

    lclf = LogisticRegression(max_iter=2000, random_state=SEED).fit(length_feats(Xtr), ytr)
    lpred, lprob = lclf.predict(length_feats(Xte)), lclf.predict_proba(length_feats(Xte))[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(yte, lpred, average="binary", zero_division=0)
    rows.append((name, "length only", accuracy_score(yte, lpred), p, r, f1, roc_auc_score(yte, lprob)))

    # top features
    names, coef = np.array(vec.get_feature_names_out()), clf.coef_[0]
    order = np.argsort(coef)
    print(f"\n[{name}] top words -> BENIGN : {', '.join(names[order[:12]])}")
    print(f"[{name}] top words -> ATTACK : {', '.join(names[order[-12:]][::-1])}")
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.r_[order[:12], order[-12:]]
    ax.barh(names[idx], coef[idx], color=["tab:blue"] * 12 + ["tab:red"] * 12)
    ax.set_title(f"{name}: most influential words (blue = benign, red = attack)")
    fig.tight_layout()
    fig.savefig(f"figures/shortcut_features_{name.split()[0]}.png", dpi=150)
    plt.close(fig)
    return rows


results = []
Xtr, Xte, ytr, yte = split_ab(contexts)
results += evaluate("A1 squad contexts vs injections", Xtr, Xte, ytr, yte)
Xtr, Xte, ytr, yte = split_ab(questions)
results += evaluate("A2 squad questions vs injections", Xtr, Xte, ytr, yte)
Xtr, Xte, ytr, yte = split_b()
results += evaluate("B deepset benign vs injection", Xtr, Xte, ytr, yte)

df = pd.DataFrame(results, columns=["experiment", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]).round(3)
df.to_csv("reports/shortcut_check.csv", index=False)
print("\n" + df.to_string(index=False))

a = df[(df.experiment.str.startswith("A")) & (df.model == "tfidf+logreg")].f1.max()
b = df[(df.experiment.str.startswith("B")) & (df.model == "tfidf+logreg")].f1.max()
print(f"\nVerdict: cross-source F1 = {a}, same-source F1 = {b}")
if a - b > 0.05:
    print("-> Cross-source is much easier: classes differ by SOURCE/STYLE. Build matched pairs (Day 6).")
else:
    print("-> No big gap, but still check the top words and length-only rows before trusting it.")
