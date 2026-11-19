"""Shared evaluation module for model training and comparison across splits.

Supports:
- Input formatting (data_only, plain_inst_data, frege)
- Threshold selection under val_template FPR <= 0.01 rule
- Multi-split evaluation: AUC, recall, precision, F1, FPR, hard-negative FPR
- Per-sentence macro recall and 95% bootstrap CI on test_real
- Per-attack-family recall on test_template and test_ood
- Logging and appending to reports/ladder_results.csv
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LADDER_RESULTS_PATH = REPORTS / "ladder_results.csv"

# Style prefixes tolerant of "P.S.", "P.S.:", "Note -", "Important:", etc.
_PREFIX = re.compile(r"^\s*(?:note|important|p\.\s?s\.?|reminder|update)\s*[:\-\u2013\u2014]?\s*", re.IGNORECASE)


def normalize_payload(s: Any) -> str:
    """Strip style prefixes (repeatedly), collapse whitespace, lowercase, drop trailing punctuation."""
    s = str(s if s is not None else "")
    while True:
        t = _PREFIX.sub("", s, count=1)
        if t == s:
            break
        s = t
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".!?;: ")


def format_input(df: pd.DataFrame, fmt: str = "frege") -> pd.Series:
    """Format input text from parquet dataframe.
    
    Supported formats:
    - 'data_only': df['data']
    - 'plain_inst_data': 'Instruction: ' + df['instruction'] + '\\n\\nData: ' + df['data']
    - 'frege': df['text']
    """
    if fmt == "data_only":
        return df["data"]
    elif fmt == "plain_inst_data":
        return "Instruction: " + df["instruction"].astype(str) + "\n\nData: " + df["data"].astype(str)
    elif fmt == "frege":
        return df["text"]
    else:
        raise ValueError(f"Unknown format: '{fmt}'. Must be one of ['data_only', 'plain_inst_data', 'frege']")


def select_threshold_max_fpr(y_true: Union[np.ndarray, pd.Series], probs: Union[np.ndarray, pd.Series], max_fpr: float = 0.01) -> float:
    """Pick threshold on val_template that maximizes recall (or F1 on ties) while keeping FPR <= max_fpr.
    
    If no threshold achieves FPR <= max_fpr, fall back to the threshold with lowest FPR.
    Never touch test* splits.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    
    fpr, tpr, thresholds = roc_curve(y_true, probs, drop_intermediate=False)
    
    # Filter out non-finite thresholds (e.g. roc_curve's initial +inf)
    finite_mask = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite_mask], tpr[finite_mask], thresholds[finite_mask]
    
    # Find candidates with FPR <= max_fpr
    mask = fpr <= max_fpr
    if not np.any(mask):
        # Fallback to threshold(s) with minimum FPR
        min_fpr = np.min(fpr)
        mask = fpr == min_fpr

    candidate_indices = np.where(mask)[0]
    
    best_th = 0.5
    best_recall = -1.0
    best_f1 = -1.0
    best_fpr = 2.0
    
    n_neg = np.sum(y_true == 0)
    
    for idx in candidate_indices:
        th = float(thresholds[idx])
        pred = probs >= th
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        fn = np.sum(~pred & (y_true == 1))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        current_fpr = fp / n_neg if n_neg > 0 else 0.0
        
        # Maximize recall, then F1, then minimize FPR, then maximize threshold (more conservative)
        if (recall > best_recall) or \
           (np.isclose(recall, best_recall) and f1 > best_f1) or \
           (np.isclose(recall, best_recall) and np.isclose(f1, best_f1) and current_fpr < best_fpr) or \
           (np.isclose(recall, best_recall) and np.isclose(f1, best_f1) and np.isclose(current_fpr, best_fpr) and th > best_th):
            best_recall = recall
            best_f1 = f1
            best_fpr = current_fpr
            best_th = th
            
    return float(best_th)


def compute_real_sentence_metrics(
    df: pd.DataFrame, probs: np.ndarray, threshold: float, boot: int = 2000, seed: int = 42
) -> Tuple[float, float, float, int]:
    """Compute per-sentence macro recall and 95% bootstrap CI on test_real over distinct payloads."""
    y = df["label"].to_numpy()
    pred = probs >= threshold
    att = df[y == 1].copy()
    att["hit"] = pred[y == 1]
    att["sent"] = att["payload"].map(normalize_payload)
    
    per = att.groupby("sent", sort=True).agg(rate=("hit", "mean"), rows=("hit", "size"))
    rates = per["rate"].to_numpy()
    macro_recall = float(rates.mean())
    
    rng = np.random.default_rng(seed)
    boots = [rng.choice(rates, size=len(rates), replace=True).mean() for _ in range(boot)]
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return macro_recall, float(ci_low), float(ci_high), len(rates)


def evaluate_all_splits(
    model_name: str,
    fmt: str,
    split_probs: Dict[str, np.ndarray],
    split_dfs: Dict[str, pd.DataFrame],
    threshold: float,
    eval_splits: Optional[list] = None,
    save_to_csv: bool = True,
) -> pd.DataFrame:
    """Evaluate predictions on all splits at a fixed threshold and optionally append to reports/ladder_results.csv."""
    if eval_splits is None:
        eval_splits = ["val", "val_template", "test", "test_template", "test_real", "test_ood"]
        
    records = []
    
    for split_name in eval_splits:
        if split_name not in split_probs or split_name not in split_dfs:
            continue
            
        df = split_dfs[split_name]
        probs = np.asarray(split_probs[split_name])
        y = df["label"].to_numpy()
        pred = probs >= threshold
        
        # Standard metrics
        n_samples = len(df)
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        try:
            auc = float(roc_auc_score(y, probs))
        except Exception:
            auc = np.nan
            
        # Hard negative metrics
        hn_mask = (y == 0) & ((df.get("variant") == "hard_negative") | (df.get("hn_type") != "none"))
        hn_fpr = float(pred[hn_mask].mean()) if hn_mask.sum() > 0 else np.nan
        
        hn_to_col = df.get("hn_test_only")
        if hn_to_col is not None and hn_to_col.astype(bool).sum() > 0:
            hn_to_mask = (y == 0) & hn_to_col.astype(bool)
            hn_test_only_fpr = float(pred[hn_to_mask].mean()) if hn_to_mask.sum() > 0 else np.nan
        else:
            hn_test_only_fpr = np.nan
            
        # test_real per-sentence macro recall
        real_macro = np.nan
        real_ci_low = np.nan
        real_ci_high = np.nan
        if split_name == "test_real" and "payload" in df.columns:
            real_macro, real_ci_low, real_ci_high, _ = compute_real_sentence_metrics(df, probs, threshold)
            
        # Per-family recall for test_template and test_ood
        per_family_recall = {}
        if split_name in ("test_template", "test_ood", "val_template") and "attack_family" in df.columns:
            att_df = df[y == 1]
            for fam, g in att_df.groupby("attack_family", sort=True):
                fam_idx = g.index.to_numpy()
                # Find positions in split array
                fam_pos = np.isin(df.index.to_numpy(), fam_idx)
                per_family_recall[fam] = round(float(pred[fam_pos].mean()), 4)
                
        rec = {
            "model": model_name,
            "format": fmt,
            "split": split_name,
            "n": n_samples,
            "threshold": round(threshold, 4),
            "auc": round(auc, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "hn_fpr": round(hn_fpr, 4) if not np.isnan(hn_fpr) else None,
            "hn_test_only_fpr": round(hn_test_only_fpr, 4) if not np.isnan(hn_test_only_fpr) else None,
            "real_macro_recall": round(real_macro, 4) if not np.isnan(real_macro) else None,
            "real_ci_low": round(real_ci_low, 4) if not np.isnan(real_ci_low) else None,
            "real_ci_high": round(real_ci_high, 4) if not np.isnan(real_ci_high) else None,
            "per_family_recall": json.dumps(per_family_recall) if per_family_recall else None,
        }
        records.append(rec)
        
    results_df = pd.DataFrame(records)
    
    # Save / append to CSV
    if save_to_csv:
        REPORTS.mkdir(parents=True, exist_ok=True)
        if LADDER_RESULTS_PATH.exists():
            existing = pd.read_csv(LADDER_RESULTS_PATH)
            # Remove any previous records for same (model, format, split)
            keys = set(zip(results_df["model"], results_df["format"], results_df["split"]))
            keep = [
                (r["model"], r["format"], r["split"]) not in keys
                for _, r in existing.iterrows()
            ]
            merged = pd.concat([existing[keep], results_df], ignore_index=True)
            merged.to_csv(LADDER_RESULTS_PATH, index=False)
        else:
            results_df.to_csv(LADDER_RESULTS_PATH, index=False)
            
    return results_df
