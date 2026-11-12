"""TF-IDF baseline on EVERY split, with a threshold policy that is not arbitrary.
(independent of script 09)

Splits scored if present: test, test_template (unseen wording), test_real (real deepset sentences), test_ood (unseen family).
Thresholds come from VAL only:
  0.5            : the default
  val-F1 (mid)   : middle of the plateau of best-F1 thresholds (if val is easy, F1 is flat over a wide range).
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix

def main():
    splits = ['train', 'val', 'test', 'test_template', 'test_real', 'test_ood']
    dfs = {}
    
    print("Loading datasets...")
    for s in splits:
        path = f"data/processed/{s}.parquet"
        if os.path.exists(path):
            dfs[s] = pd.read_parquet(path)
            print(f"  Loaded {s:<14} ({len(dfs[s])} rows)")
        else:
            print(f"  Skipped {s:<13} (file not found)")

    if 'train' not in dfs or 'val' not in dfs:
        print("\nERROR: Need at least train.parquet and val.parquet to run.")
        return

    train, val = dfs['train'], dfs['val']

    print("\nFitting TF-IDF + Logistic Regression on TRAIN data...")
    vec = TfidfVectorizer(max_features=10000)
    X_train = vec.fit_transform(train['text'])
    y_train = train['label']
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("Calibrating thresholds on VAL...")
    X_val = vec.transform(val['text'])
    y_val = val['label']
    val_probs = clf.predict_proba(X_val)[:, 1]

    # Find the F1 plateau
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_val, val_probs >= t) for t in thresholds]
    max_f1 = max(f1s)
    
    # Get all thresholds that achieve the maximum F1 score and find the midpoint
    best_thresholds = [t for t, f in zip(thresholds, f1s) if f == max_f1]
    mid_t = float(np.mean(best_thresholds))

    print(f"  Max Val F1: {max_f1:.4f}")
    print(f"  F1 Plateau covers thresholds: {min(best_thresholds):.2f} to {max(best_thresholds):.2f}")
    print(f"  val-F1 (mid) threshold set to: {mid_t:.3f}\n")

    print(f"{'Split':<15} | {'Thresh':<7} | {'AUC':<5} | {'F1':<5} | {'Recall':<6} | {'Prec':<5} | {'FPR':<5}")
    print("-" * 72)

    def evaluate(split_name, df, t, t_name):
        X = vec.transform(df['text'])
        y = df['label']
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= t).astype(int)
        
        # Protect against single-class splits
        if len(np.unique(y)) > 1:
            auc = roc_auc_score(y, probs)
            tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            auc = float('nan')
            fpr = float('nan')
            if np.unique(y)[0] == 0:
                # All benign, FP calculation
                fp = np.sum(preds == 1)
                tn = np.sum(preds == 0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        f1 = f1_score(y, preds)
        rec = recall_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        
        print(f"{split_name:<15} | {t_name:<7} | {auc:.3f} | {f1:.3f} | {rec:.3f} | {prec:.3f} | {fpr:.3f}")

    for s_name in ['val', 'test', 'test_template', 'test_real', 'test_ood']:
        if s_name in dfs:
            evaluate(s_name, dfs[s_name], 0.5, "0.50")
            evaluate(s_name, dfs[s_name], mid_t, f"{mid_t:.3f}")
            print("-" * 72)

if __name__ == "__main__":
    main()
