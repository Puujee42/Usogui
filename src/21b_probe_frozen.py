"""Step 2 Diagnostic Probe: Frozen ModernBERT-base representation analysis.

Diagnoses:
  (a) Train & Val AUC for mean pooling with StandardScaler + LogisticRegression(C=1.0, max_iter=2000).
  (b) Regularization sweep on mean pooling: C in [0.1, 1, 10, 100, 1000], reporting train_auc, val_auc,
      val_thresh_fpr1pct, val_recall_at_fpr1pct, and val_fpr.
  (c) Pooling comparison (mean vs cls vs max) across C in [0.1, 1, 10, 100, 1000], reporting best val_auc
      and best val_recall_at_fpr1pct for each pooling method.
  (d) Payload-only probe: embed isolated attack payload strings (from train label=1, deduplicated) vs
      isolated harmless sentences (populated payloads from train hard negatives, deduplicated) using a
      clean 80/20 split by unique sentence (and also evaluating on unseen val sentences). Reports train AUC
      and held-out sentence AUC for mean, cls, and max pooling.

STRICT PROTOCOL RULE: Uses train.parquet and val.parquet ONLY.
Never touches any test* or val_template split.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_utils import format_input  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "reports" / "cache"
TRAIN_VAL_CACHE = CACHE_DIR / "step2_train_val_pools.npz"
PAYLOAD_CACHE = CACHE_DIR / "step2_payload_pools.npz"

MODEL_ID = "answerdotai/ModernBERT-base"
C_VALUES = [0.1, 1.0, 10.0, 100.0, 1000.0]
POOL_TYPES = ["mean", "cls", "max"]


def select_threshold_max_fpr(y_true: np.ndarray, probs: np.ndarray, max_fpr: float = 0.01) -> float:
    """Select threshold on probabilities that maximizes recall while ensuring FPR <= max_fpr."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    fpr, tpr, thresholds = roc_curve(y_true, probs, drop_intermediate=False)

    finite_mask = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite_mask], tpr[finite_mask], thresholds[finite_mask]

    mask = fpr <= max_fpr
    if not np.any(mask):
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

        if (recall > best_recall) or \
           (np.isclose(recall, best_recall) and f1 > best_f1) or \
           (np.isclose(recall, best_recall) and np.isclose(f1, best_f1) and current_fpr < best_fpr) or \
           (np.isclose(recall, best_recall) and np.isclose(f1, best_f1) and np.isclose(current_fpr, best_fpr) and th > best_th):
            best_recall = recall
            best_f1 = f1
            best_fpr = current_fpr
            best_th = th

    return float(best_th)


def extract_all_pools_for_texts(
    texts: list,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 64,
    max_len: int = 512,
) -> dict:
    """Extract mean, cls, and max pooled representations in a single forward pass."""
    model.eval()
    cls_list, mean_list, max_list = [], [], []
    n = len(texts)

    for i in range(0, n, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            lhs = outputs.last_hidden_state  # (B, L, D)

            # 1. CLS: first token
            cls_p = lhs[:, 0, :]

            # 2. Mean: mean over valid non-pad tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(lhs.size()).to(lhs.dtype)
            sum_embeddings = torch.sum(lhs * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_p = sum_embeddings / sum_mask

            # 3. Max: max over valid non-pad tokens (masking pad tokens to -inf)
            mask = attention_mask.unsqueeze(-1)
            lhs_masked = lhs.masked_fill(mask == 0, -float("inf"))
            max_p = torch.max(lhs_masked, dim=1).values

        cls_list.append(cls_p.float().cpu().numpy())
        mean_list.append(mean_p.float().cpu().numpy())
        max_list.append(max_p.float().cpu().numpy())

    return {
        "cls": np.concatenate(cls_list, axis=0),
        "mean": np.concatenate(mean_list, axis=0),
        "max": np.concatenate(max_list, axis=0),
    }


def load_or_extract_train_val_pools(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str,
    batch_size: int = 64,
    max_len: int = 512,
    force_extract: bool = False,
) -> dict:
    """Load cached train/val pools or extract them in a single forward pass."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_VAL_CACHE.exists() and not force_extract:
        print(f"Loading cached train/val pooled embeddings from {TRAIN_VAL_CACHE}...")
        try:
            data = np.load(TRAIN_VAL_CACHE)
            expected_keys = [f"{split}_{pool}" for split in ["train", "val"] for pool in POOL_TYPES]
            if all(k in data for k in expected_keys) and \
               len(data["train_mean"]) == len(train_df) and \
               len(data["val_mean"]) == len(val_df):
                print(f"Cache verified! Train: {len(data['train_mean'])} rows, Val: {len(data['val_mean'])} rows.")
                return {
                    "train": {p: data[f"train_{p}"] for p in POOL_TYPES},
                    "val": {p: data[f"val_{p}"] for p in POOL_TYPES},
                }
            else:
                print("Cache shape mismatch or missing keys, re-extracting...")
        except Exception as e:
            print(f"Could not load cache ({e}), re-extracting...")

    print(f"\nInitializing tokenizer & model ({MODEL_ID}) on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)

    train_texts = format_input(train_df, fmt="frege").tolist()
    val_texts = format_input(val_df, fmt="frege").tolist()

    print(f"Extracting 3 pooling types (mean, cls, max) for Train ({len(train_texts)} rows)...")
    t0 = time.time()
    train_pools = extract_all_pools_for_texts(
        texts=train_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_len=max_len,
    )
    t_train = time.time() - t0
    print(f"Train extracted in {t_train:.1f}s ({len(train_texts)/t_train:.1f} rows/s)")

    print(f"Extracting 3 pooling types (mean, cls, max) for Val ({len(val_texts)} rows)...")
    t0 = time.time()
    val_pools = extract_all_pools_for_texts(
        texts=val_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_len=max_len,
    )
    t_val = time.time() - t0
    print(f"Val extracted in {t_val:.1f}s ({len(val_texts)/t_val:.1f} rows/s)")

    np.savez_compressed(
        TRAIN_VAL_CACHE,
        train_mean=train_pools["mean"],
        train_cls=train_pools["cls"],
        train_max=train_pools["max"],
        val_mean=val_pools["mean"],
        val_cls=val_pools["cls"],
        val_max=val_pools["max"],
    )
    print(f"Cached train/val embeddings saved to {TRAIN_VAL_CACHE}.\n")

    return {"train": train_pools, "val": val_pools}


def load_or_extract_payload_pools(
    sentences: list,
    device: str,
    batch_size: int = 64,
    max_len: int = 128,
    force_extract: bool = False,
) -> dict:
    """Load cached payload pools or extract them in a single forward pass."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if PAYLOAD_CACHE.exists() and not force_extract:
        try:
            data = np.load(PAYLOAD_CACHE, allow_pickle=True)
            if "sentences" in data and len(data["sentences"]) == len(sentences) and \
               all(s1 == s2 for s1, s2 in zip(data["sentences"], sentences)):
                return {p: data[p] for p in POOL_TYPES}
        except Exception:
            pass

    print(f"\nExtracting isolated sentence embeddings ({len(sentences)} unique sentences)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)

    t0 = time.time()
    pools = extract_all_pools_for_texts(
        texts=sentences,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_len=max_len,
    )
    print(f"Payload sentences extracted in {time.time() - t0:.1f}s.")

    np.savez_compressed(
        PAYLOAD_CACHE,
        sentences=np.array(sentences, dtype=object),
        mean=pools["mean"],
        cls=pools["cls"],
        max=pools["max"],
    )
    return pools


def run_diagnostic_a(train_pools: dict, val_pools: dict, y_train: np.ndarray, y_val: np.ndarray):
    """(a) Train & Val AUC for mean pooling with StandardScaler + LogisticRegression(C=1.0, max_iter=2000)."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC (a): Baseline Mean Pooling (C=1.0)")
    print("=" * 80)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_pools["mean"])
    X_val = scaler.transform(val_pools["mean"])

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)

    train_probs = clf.predict_proba(X_train)[:, 1]
    val_probs = clf.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_probs)
    val_auc = roc_auc_score(y_val, val_probs)

    val_th = select_threshold_max_fpr(y_val, val_probs, max_fpr=0.01)
    val_pred = val_probs >= val_th
    val_recall = np.sum(val_pred & (y_val == 1)) / np.sum(y_val == 1)
    val_fpr = np.sum(val_pred & (y_val == 0)) / np.sum(y_val == 0)

    res_df = pd.DataFrame([{
        "Pooling": "mean",
        "C": 1.0,
        "Train AUC": f"{train_auc:.4f}",
        "Val AUC": f"{val_auc:.4f}",
        "Val Thresh (FPR<=1%)": f"{val_th:.4f}",
        "Val Recall @ <=1% FPR": f"{val_recall:.4f}",
        "Val FPR": f"{val_fpr:.4f}",
    }])
    print(res_df.to_string(index=False))
    return train_auc, val_auc


def run_diagnostic_b(train_pools: dict, val_pools: dict, y_train: np.ndarray, y_val: np.ndarray):
    """(b) Regularization sweep on mean pooling: C in [0.1, 1, 10, 100, 1000]."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC (b): Regularization Sweep on Mean Pooling (C in [0.1, 1, 10, 100, 1000])")
    print("=" * 80)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_pools["mean"])
    X_val = scaler.transform(val_pools["mean"])

    records = []
    for c in C_VALUES:
        clf = LogisticRegression(C=c, max_iter=2000, random_state=42)
        clf.fit(X_train, y_train)

        train_probs = clf.predict_proba(X_train)[:, 1]
        val_probs = clf.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, train_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        val_th = select_threshold_max_fpr(y_val, val_probs, max_fpr=0.01)
        val_pred = val_probs >= val_th
        val_recall = np.sum(val_pred & (y_val == 1)) / np.sum(y_val == 1)
        val_fpr = np.sum(val_pred & (y_val == 0)) / np.sum(y_val == 0)

        records.append({
            "C": c,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "val_thresh_fpr1pct": val_th,
            "val_recall_at_fpr1pct": val_recall,
            "val_fpr": val_fpr,
        })

    sweep_df = pd.DataFrame(records)
    print(sweep_df.to_string(index=False, formatters={
        "C": lambda x: f"{x:8.1f}",
        "train_auc": lambda x: f"{x:.4f}",
        "val_auc": lambda x: f"{x:.4f}",
        "val_thresh_fpr1pct": lambda x: f"{x:.4f}",
        "val_recall_at_fpr1pct": lambda x: f"{x:.4f}",
        "val_fpr": lambda x: f"{x:.4f}",
    }))
    return sweep_df


def run_diagnostic_c(train_pools: dict, val_pools: dict, y_train: np.ndarray, y_val: np.ndarray):
    """(c) Pooling comparison (mean vs cls vs max) across C in [0.1, 1, 10, 100, 1000]."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC (c): Pooling Comparison (mean vs cls vs max) across C in [0.1, 1, 10, 100, 1000]")
    print("=" * 80)

    all_records = []
    summary_records = []

    for pool in POOL_TYPES:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_pools[pool])
        X_val = scaler.transform(val_pools[pool])

        best_auc = -1.0
        best_auc_c = None
        best_recall = -1.0
        best_recall_c = None
        best_recall_fpr = 1.0

        for c in C_VALUES:
            clf = LogisticRegression(C=c, max_iter=2000, random_state=42)
            clf.fit(X_train, y_train)

            train_probs = clf.predict_proba(X_train)[:, 1]
            val_probs = clf.predict_proba(X_val)[:, 1]

            train_auc = roc_auc_score(y_train, train_probs)
            val_auc = roc_auc_score(y_val, val_probs)

            val_th = select_threshold_max_fpr(y_val, val_probs, max_fpr=0.01)
            val_pred = val_probs >= val_th
            val_fpr = np.sum(val_pred & (y_val == 0)) / np.sum(y_val == 0)
            val_recall = np.sum(val_pred & (y_val == 1)) / np.sum(y_val == 1)

            all_records.append({
                "pooling": pool,
                "C": c,
                "train_auc": train_auc,
                "val_auc": val_auc,
                "val_thresh_fpr1pct": val_th,
                "val_recall_at_fpr1pct": val_recall,
                "val_fpr": val_fpr,
            })

            if val_auc > best_auc:
                best_auc = val_auc
                best_auc_c = c

            # Best recall strictly under the FPR <= 1% constraint
            if val_fpr <= 0.010001:
                if val_recall > best_recall or (np.isclose(val_recall, best_recall) and val_fpr < best_recall_fpr):
                    best_recall = val_recall
                    best_recall_c = c
                    best_recall_fpr = val_fpr

        summary_records.append({
            "pooling": pool,
            "best_val_auc": best_auc,
            "best_auc_c": best_auc_c,
            "best_val_recall_at_fpr1pct": best_recall,
            "best_recall_c": best_recall_c,
            "val_fpr_at_best_recall": best_recall_fpr,
        })

    summary_df = pd.DataFrame(summary_records)
    print("--- Best Metrics Summary by Pooling Method ---")
    print(summary_df.to_string(index=False, formatters={
        "best_val_auc": lambda x: f"{x:.4f}",
        "best_auc_c": lambda x: f"{x:6.1f}",
        "best_val_recall_at_fpr1pct": lambda x: f"{x:.4f}",
        "best_recall_c": lambda x: f"{x:6.1f}",
        "val_fpr_at_best_recall": lambda x: f"{x:.4f}",
    }))

    print("\n--- Detailed Grid (Pooling x Regularization C) ---")
    detail_df = pd.DataFrame(all_records)
    print(detail_df.to_string(index=False, formatters={
        "C": lambda x: f"{x:8.1f}",
        "train_auc": lambda x: f"{x:.4f}",
        "val_auc": lambda x: f"{x:.4f}",
        "val_thresh_fpr1pct": lambda x: f"{x:.4f}",
        "val_recall_at_fpr1pct": lambda x: f"{x:.4f}",
        "val_fpr": lambda x: f"{x:.4f}",
    }))
    return summary_df, detail_df


def run_diagnostic_d(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str,
    batch_size: int = 64,
    force_extract: bool = False,
):
    """(d) Payload-only probe: isolated attack payloads vs isolated harmless sentences."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC (d): Payload-Only Probe (Isolated Sentences)")
    print("=" * 80)

    # 1. Harvest train attack payloads (label=1, deduplicated)
    train_attack_series = train_df[train_df["label"] == 1]["payload"].dropna()
    train_attacks = sorted(list(set(s.strip() for s in train_attack_series if s.strip())))

    # 2. Harvest train harmless sentences (label=0, hard negatives with non-empty payload)
    train_hn_series = train_df[(train_df["label"] == 0) & (train_df["payload"].str.strip() != "")]["payload"].dropna()
    train_harmless = sorted(list(set(s.strip() for s in train_hn_series if s.strip())))

    print(f"Extracted from train: {len(train_attacks)} unique attack payloads, {len(train_harmless)} unique harmless sentences.")
    assert len(set(train_attacks) & set(train_harmless)) == 0, "Overlap found between attacks and harmless sentences!"

    # Combine and create labels
    all_train_sents = train_attacks + train_harmless
    all_train_labels = np.array([1] * len(train_attacks) + [0] * len(train_harmless))

    # Clean 80/20 train/held-out split stratified by unique sentence
    tr_sents, ho_sents, tr_y, ho_y = train_test_split(
        all_train_sents,
        all_train_labels,
        test_size=0.20,
        random_state=42,
        stratify=all_train_labels,
    )
    print(f"Clean 80/20 sentence split: Train={len(tr_sents)} (Attacks={sum(tr_y)}, Harmless={len(tr_y)-sum(tr_y)}) | "
          f"Held-out={len(ho_sents)} (Attacks={sum(ho_y)}, Harmless={len(ho_y)-sum(ho_y)})")

    # Harvest strictly unseen payloads from val.parquet for out-of-distribution sentence check
    val_attack_series = val_df[val_df["label"] == 1]["payload"].dropna()
    val_attacks_unseen = sorted(list(set(s.strip() for s in val_attack_series if s.strip()) - set(all_train_sents)))

    val_hn_series = val_df[(val_df["label"] == 0) & (val_df["payload"].str.strip() != "")]["payload"].dropna()
    val_harmless_unseen = sorted(list(set(s.strip() for s in val_hn_series if s.strip()) - set(all_train_sents)))

    val_unseen_sents = val_attacks_unseen + val_harmless_unseen
    val_unseen_y = np.array([1] * len(val_attacks_unseen) + [0] * len(val_harmless_unseen))
    print(f"Val unseen sentences (zero overlap with train): Total={len(val_unseen_sents)} "
          f"(Attacks={len(val_attacks_unseen)}, Harmless={len(val_harmless_unseen)})")

    # Order all unique sentences to extract embeddings in a single pass
    # (train sents + val unseen sents)
    combined_unique_sents = all_train_sents + val_unseen_sents
    pools = load_or_extract_payload_pools(
        sentences=combined_unique_sents,
        device=device,
        batch_size=batch_size,
        max_len=128,
        force_extract=force_extract,
    )

    # Index mapping
    sent_to_idx = {s: i for i, s in enumerate(combined_unique_sents)}
    tr_indices = [sent_to_idx[s] for s in tr_sents]
    ho_indices = [sent_to_idx[s] for s in ho_sents]
    val_unseen_indices = [sent_to_idx[s] for s in val_unseen_sents]

    records = []
    for pool in POOL_TYPES:
        X_all = pools[pool]
        X_tr = X_all[tr_indices]
        X_ho = X_all[ho_indices]
        X_val_unseen = X_all[val_unseen_indices]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_ho_s = scaler.transform(X_ho)
        X_val_unseen_s = scaler.transform(X_val_unseen)

        # Baseline C=1.0 Logistic Regression
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        clf.fit(X_tr_s, tr_y)

        tr_probs = clf.predict_proba(X_tr_s)[:, 1]
        ho_probs = clf.predict_proba(X_ho_s)[:, 1]
        val_unseen_probs = clf.predict_proba(X_val_unseen_s)[:, 1]

        tr_auc = roc_auc_score(tr_y, tr_probs)
        ho_auc = roc_auc_score(ho_y, ho_probs)
        val_unseen_auc = roc_auc_score(val_unseen_y, val_unseen_probs)

        # Also find best C for held-out AUC
        best_ho_auc = -1.0
        best_c = 1.0
        for c in C_VALUES:
            clf_c = LogisticRegression(C=c, max_iter=2000, random_state=42)
            clf_c.fit(X_tr_s, tr_y)
            c_ho_probs = clf_c.predict_proba(X_ho_s)[:, 1]
            c_auc = roc_auc_score(ho_y, c_ho_probs)
            if c_auc > best_ho_auc:
                best_ho_auc = c_auc
                best_c = c

        records.append({
            "pooling": pool,
            "train_auc (C=1.0)": tr_auc,
            "held_out_sentence_auc (C=1.0)": ho_auc,
            "val_unseen_sentence_auc (C=1.0)": val_unseen_auc,
            "best_held_out_auc": best_ho_auc,
            "best_C": best_c,
        })

    payload_df = pd.DataFrame(records)
    print("\n--- Diagnostic (d) Results Table ---")
    print(payload_df.to_string(index=False, formatters={
        "train_auc (C=1.0)": lambda x: f"{x:.4f}",
        "held_out_sentence_auc (C=1.0)": lambda x: f"{x:.4f}",
        "val_unseen_sentence_auc (C=1.0)": lambda x: f"{x:.4f}",
        "best_held_out_auc": lambda x: f"{x:.4f}",
        "best_C": lambda x: f"{x:6.1f}",
    }))
    return payload_df


def main():
    parser = argparse.ArgumentParser(description="Step 2 Diagnostic Probe: Frozen ModernBERT-base")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length for document encoding")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of embeddings even if cached")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 2 DIAGNOSTIC PROBE: FROZEN ModernBERT-base (train.parquet & val.parquet ONLY)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Batch Size: {args.batch_size} | Max Len: {args.max_len}")

    # Load train and val ONLY
    train_path = PROCESSED / "train.parquet"
    val_path = PROCESSED / "val.parquet"

    print(f"Loading {train_path} and {val_path}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    print(f"Train: {len(train_df)} rows (Positive={sum(y_train)}, Negative={len(y_train)-sum(y_train)})")
    print(f"Val:   {len(val_df)} rows (Positive={sum(y_val)}, Negative={len(y_val)-sum(y_val)})")

    # Extract or load cached full document embeddings for train and val
    pools = load_or_extract_train_val_pools(
        train_df=train_df,
        val_df=val_df,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        force_extract=args.force_extract,
    )

    # (a) Baseline Mean Pooling
    run_diagnostic_a(pools["train"], pools["val"], y_train, y_val)

    # (b) Regularization Sweep on Mean Pooling
    run_diagnostic_b(pools["train"], pools["val"], y_train, y_val)

    # (c) Pooling Comparison (mean vs cls vs max)
    run_diagnostic_c(pools["train"], pools["val"], y_train, y_val)

    # (d) Payload-Only Probe
    run_diagnostic_d(
        train_df=train_df,
        val_df=val_df,
        device=device,
        batch_size=args.batch_size,
        force_extract=args.force_extract,
    )

    print("\n" + "=" * 80)
    print("ALL DIAGNOSTICS COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
