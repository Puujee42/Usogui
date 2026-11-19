"""Ladder Step 1 & Step 2 evaluation under the val_template <= 1% FPR rule.

Step 1: TF-IDF + Logistic Regression (format: frege)
Step 2: Frozen ModernBERT-base (bfloat16 + mean pooling) + StandardScaler + Logistic Regression (format: frege)

Extracts 768-d embeddings for all 7 splits, tunes the threshold strictly on val_template (FPR <= 0.01),
and evaluates all splits using eval_utils.py, appending to reports/ladder_results.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_utils import format_input, select_threshold_max_fpr, evaluate_all_splits  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
ALL_SPLITS = ["train", "val", "val_template", "test", "test_template", "test_real", "test_ood"]
EVAL_SPLITS = ["val", "val_template", "test", "test_template", "test_real", "test_ood"]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over valid tokens using attention mask."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def extract_embeddings_for_split(
    texts: list,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 64,
    max_len: int = 512,
) -> np.ndarray:
    """Batch inference extracting 768-d mean-pooled embeddings."""
    model.eval()
    embeddings = []
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
            pooled = mean_pool(outputs.last_hidden_state, attention_mask)
            
        embeddings.append(pooled.float().cpu().numpy())
        
    return np.concatenate(embeddings, axis=0)


def run_step1_tfidf(dfs: dict, fmt: str = "frege"):
    print("=" * 80)
    print("STEP 1: TF-IDF + LogisticRegression (apples-to-apples re-score under val_template FPR <= 0.01)")
    print("=" * 80)
    
    t0 = time.time()
    X_train = format_input(dfs["train"], fmt=fmt)
    y_train = dfs["train"]["label"].to_numpy()
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_vec = vec.fit_transform(X_train)
    
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    probs = {}
    for s in ALL_SPLITS:
        X = format_input(dfs[s], fmt=fmt)
        probs[s] = clf.predict_proba(vec.transform(X))[:, 1]
        
    thr = select_threshold_max_fpr(dfs["val_template"]["label"], probs["val_template"], max_fpr=0.01)
    fit_time = time.time() - t0
    print(f"Fitted in {fit_time:.2f}s | Selected threshold on val_template: {thr:.4f}\n")
    
    step1_results = evaluate_all_splits(
        model_name="Step 1 (TF-IDF + LogReg)",
        fmt=fmt,
        split_probs=probs,
        split_dfs=dfs,
        threshold=thr,
        eval_splits=EVAL_SPLITS,
        save_to_csv=True,
    )
    return step1_results, probs, thr


def run_step2_frozen_modernbert(dfs: dict, fmt: str = "frege", batch_size: int = 64, max_len: int = 512, force_extract: bool = False):
    print("=" * 80)
    print("STEP 2: Frozen ModernBERT-base + LogisticRegression (mean pooling, max_len=512, bf16)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "answerdotai/ModernBERT-base"
    print(f"Device: {device} | Model: {model_id} | Batch size: {batch_size} | max_len: {max_len}")
    
    cache_path = INTERIM / "frozen_modernbert_embeddings.pt"
    cached_embeddings = {}
    if cache_path.exists() and not force_extract:
        print(f"Loading cached embeddings from {cache_path}...")
        try:
            cached_embeddings = torch.load(cache_path, weights_only=False)
            # Verify all splits present
            if all(s in cached_embeddings and len(cached_embeddings[s]) == len(dfs[s]) for s in ALL_SPLITS):
                print("All splits found in cache and match current split sizes!")
            else:
                print("Cache incomplete or size mismatch, re-extracting missing...")
                cached_embeddings = {}
        except Exception as e:
            print(f"Could not load cache ({e}), re-extracting...")
            cached_embeddings = {}
            
    if not cached_embeddings:
        print(f"Loading tokenizer & model ({model_id})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModel.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            ).to(device)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(device)
            
        print("Extracting 768-d embeddings for all 7 splits:")
        t_extract_start = time.time()
        for s in ALL_SPLITS:
            t0 = time.time()
            texts = format_input(dfs[s], fmt=fmt).tolist()
            emb = extract_embeddings_for_split(
                texts=texts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
                max_len=max_len,
            )
            cached_embeddings[s] = emb
            elapsed = time.time() - t0
            print(f"  [{s:<14}] {len(texts):>5} rows extracted in {elapsed:5.1f}s ({len(texts)/elapsed:5.1f} rows/s)")
            
        print(f"Total extraction time: {time.time() - t_extract_start:.1f}s")
        # Save cache
        INTERIM.mkdir(parents=True, exist_ok=True)
        torch.save(cached_embeddings, cache_path)
        print(f"Cached embeddings saved to {cache_path}.")
        
    # Fit LogisticRegression on train
    print("\nFitting StandardScaler and LogisticRegression(max_iter=1000, random_state=42) on train embeddings...")
    t0 = time.time()
    X_train = cached_embeddings["train"]
    y_train = dfs["train"]["label"].to_numpy()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    print(f"Classifier fitted in {time.time() - t0:.2f}s.")
    
    # Predict probabilities for all splits
    probs = {}
    for s in ALL_SPLITS:
        X_scaled = scaler.transform(cached_embeddings[s])
        probs[s] = clf.predict_proba(X_scaled)[:, 1]
        
    # Select threshold on val_template ONLY
    thr = select_threshold_max_fpr(dfs["val_template"]["label"], probs["val_template"], max_fpr=0.01)
    print(f"Selected threshold on val_template: {thr:.4f}\n")
    
    step2_results = evaluate_all_splits(
        model_name="Step 2 (Frozen ModernBERT + LogReg)",
        fmt=fmt,
        split_probs=probs,
        split_dfs=dfs,
        threshold=thr,
        eval_splits=EVAL_SPLITS,
        save_to_csv=True,
    )
    return step2_results, probs, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()
    
    dfs = {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in ALL_SPLITS}
    
    # Run Step 1
    s1_res, s1_probs, s1_thr = run_step1_tfidf(dfs, fmt="frege")
    
    # Run Step 2
    s2_res, s2_probs, s2_thr = run_step2_frozen_modernbert(
        dfs, fmt="frege", batch_size=args.batch_size, max_len=args.max_len, force_extract=args.force_extract
    )
    
    # Print comparison table
    print("=" * 105)
    print("LADDER COMPARISON: Step 1 (TF-IDF) vs Step 2 (Frozen ModernBERT-base)")
    print(f"Rule: val_template FPR <= 0.01  |  Step 1 Thr: {s1_thr:.4f}  |  Step 2 Thr: {s2_thr:.4f}")
    print("=" * 105)
    
    cols = ["split", "n", "auc", "recall", "precision", "f1", "fpr", "hn_fpr", "hn_test_only_fpr", "real_macro_recall", "real_ci_low", "real_ci_high"]
    
    s1_disp = s1_res[cols].copy()
    s1_disp.insert(1, "model", "Step 1 (TF-IDF)")
    
    s2_disp = s2_res[cols].copy()
    s2_disp.insert(1, "model", "Step 2 (Frozen ModernBERT)")
    
    # Interleave rows by split
    combined = []
    for split in EVAL_SPLITS:
        r1 = s1_disp[s1_disp["split"] == split].iloc[0].to_dict()
        r2 = s2_disp[s2_disp["split"] == split].iloc[0].to_dict()
        combined.append(r1)
        combined.append(r2)
        
    comp_df = pd.DataFrame(combined)
    print(comp_df.to_string(index=False))
    print("=" * 105)
    
    # Print detailed breakdowns for test_real, test_template, test_ood
    print("\nDETAILED BREAKDOWN:")
    for split in ["test_real", "test_template", "test_ood"]:
        r1 = s1_res[s1_res["split"] == split].iloc[0]
        r2 = s2_res[s2_res["split"] == split].iloc[0]
        print(f"\n--- {split.upper()} ---")
        if split == "test_real":
            print(f"  Step 1 Macro Recall: {r1['real_macro_recall']:.4f}  (95% CI: [{r1['real_ci_low']:.4f}, {r1['real_ci_high']:.4f}])")
            print(f"  Step 2 Macro Recall: {r2['real_macro_recall']:.4f}  (95% CI: [{r2['real_ci_low']:.4f}, {r2['real_ci_high']:.4f}])")
            print(f"  Step 1 Row Recall:   {r1['recall']:.4f}  |  FPR: {r1['fpr']:.4f}  |  hn_test_only FPR: {r1['hn_test_only_fpr']}")
            print(f"  Step 2 Row Recall:   {r2['recall']:.4f}  |  FPR: {r2['fpr']:.4f}  |  hn_test_only FPR: {r2['hn_test_only_fpr']}")
        else:
            print(f"  Step 1 Per-Family Recall: {r1['per_family_recall']}")
            print(f"  Step 2 Per-Family Recall: {r2['per_family_recall']}")
            print(f"  Step 1 Overall Recall:    {r1['recall']:.4f}  |  FPR: {r1['fpr']:.4f}  |  hn_test_only FPR: {r1['hn_test_only_fpr']}")
            print(f"  Step 2 Overall Recall:    {r2['recall']:.4f}  |  FPR: {r2['fpr']:.4f}  |  hn_test_only FPR: {r2['hn_test_only_fpr']}")


if __name__ == "__main__":
    main()
