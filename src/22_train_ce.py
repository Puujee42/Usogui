"""Ladder Step 3: Fine-tune ModernBERT-base with Cross-Entropy Loss.

Architecture:
- Backbone: answerdotai/ModernBERT-base (sdpa attention, gradient checkpointing)
- Pooling: Mean pooling over valid non-pad tokens (or cls pooling if --pooling cls)
- Head: Linear(768, 2)
- Precision: bfloat16 mixed precision autocast, fp32 master weights
- Optimizer: AdamW(lr=2e-5, weight_decay=0.01), linear warmup (6%) + cosine decay
- Gradient clipping: 1.0

Power/Laptop GPU Speedups (Dell G15 5530 30W cap):
- Deterministic length-bucketed batching with dynamic padding for training
- Fast inference with length sorting (batch_size=64) and order restoration for evaluation

Validation & Selection Protocol:
- NEVER touch test* splits during training
- Evaluate after each epoch strictly on val and val_template
- Best checkpoint chosen by val_template recall at FPR <= 0.01 (tie-break by val_template AUC)

Final Evaluation:
- All evaluation splits: val, val_template, test, test_template, test_real, test_ood
- Log to reports/ladder_results.csv
- Save test_real and test_template 768-d embeddings to reports/embeddings/ for UMAP
"""
import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_utils import evaluate_all_splits, format_input, select_threshold_max_fpr  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"
FIGURES = ROOT / "figures"
EMBEDDINGS = REPORTS / "embeddings"

for _p in (REPORTS, CHECKPOINTS, FIGURES, EMBEDDINGS):
    _p.mkdir(parents=True, exist_ok=True)

ALL_SPLITS = ["train", "val", "val_template", "test", "test_template", "test_real", "test_ood"]
EVAL_SPLITS = ["val", "val_template", "test", "test_template", "test_real", "test_ood"]


def set_seed(seed: int):
    """Set random seed across Python, NumPy, and PyTorch for deterministic execution."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModernBertClassifier(nn.Module):
    """ModernBERT backbone + pooling + Linear(768, 2) classification head."""

    def __init__(self, encoder: AutoModel, pooling: str = "mean", num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        hidden_size = encoder.config.hidden_size  # 768
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        lhs = outputs.last_hidden_state  # shape: (B, L, D)

        if self.pooling == "cls":
            pooled = lhs[:, 0, :]
        else:  # mean pooling over valid non-pad tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(lhs.size()).to(lhs.dtype)
            sum_embeddings = torch.sum(lhs * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

        logits = self.classifier(pooled)
        return logits, pooled


def get_length_bucketed_batches(
    lengths: np.ndarray,
    batch_size: int = 32,
    bucket_size: int = 256,
    seed: int = 42,
    epoch: int = 0,
) -> List[np.ndarray]:
    """Generates batches of indices grouped by sequence length to minimize padding overhead.
    
    1. Adds small random jitter to lengths to avoid identical groupings each epoch.
    2. Sorts indices by jittered length into mega-buckets of size ~bucket_size.
    3. Shuffles locally within each mega-bucket.
    4. Slices each bucket into mini-batches of size batch_size.
    5. Shuffles the list of batches across the epoch.
    """
    rng = np.random.default_rng(seed + epoch)
    n = len(lengths)
    indices = np.arange(n)

    jitter = rng.uniform(-2.0, 2.0, size=n)
    sorted_indices = indices[np.argsort(lengths + jitter)]

    batches = []
    for bucket_start in range(0, n, bucket_size):
        bucket = sorted_indices[bucket_start : bucket_start + bucket_size].copy()
        rng.shuffle(bucket)
        for i in range(0, len(bucket), batch_size):
            batch = bucket[i : i + batch_size]
            batches.append(batch)

    rng.shuffle(batches)
    return batches


def collate_batch(
    batch_indices: np.ndarray,
    input_ids_list: List[List[int]],
    attention_mask_list: List[List[int]],
    labels: np.ndarray,
    pad_token_id: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically pads sequences in a batch to the maximum length in that specific batch."""
    b_max_len = max(len(input_ids_list[i]) for i in batch_indices)
    b_ids = []
    b_mask = []
    b_labels = []

    for i in batch_indices:
        ids = input_ids_list[i]
        mask = attention_mask_list[i]
        pad_len = b_max_len - len(ids)
        b_ids.append(ids + [pad_token_id] * pad_len)
        b_mask.append(mask + [0] * pad_len)
        b_labels.append(labels[i])

    t_ids = torch.tensor(b_ids, dtype=torch.long, device=device)
    t_mask = torch.tensor(b_mask, dtype=torch.long, device=device)
    t_labels = torch.tensor(b_labels, dtype=torch.long, device=device)
    return t_ids, t_mask, t_labels


def run_inference_on_split(
    df: pd.DataFrame,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    fmt: str = "frege",
    batch_size: int = 64,
    max_len: int = 512,
    return_embeddings: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Runs inference on a dataframe split.
    
    Optimized for throughput: sorts by length for batch_size=64 inference under
    torch.inference_mode() and bfloat16 autocast, then inverts the permutation
    to restore the original row order before returning probabilities and embeddings!
    """
    model.eval()
    texts = format_input(df, fmt=fmt).tolist()
    n = len(texts)
    if n == 0:
        empty_emb = np.empty((0, 768), dtype=np.float32) if return_embeddings else None
        return np.empty(0, dtype=np.float32), empty_emb

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_attention_mask=True,
    )
    input_ids_list = encoded["input_ids"]
    attention_mask_list = encoded["attention_mask"]
    lengths = np.array([len(x) for x in input_ids_list])

    # Sort indices by length
    sorted_indices = np.argsort(lengths)

    all_probs_sorted = []
    all_embeddings_sorted = []

    pad_token_id = tokenizer.pad_token_id or 50283

    with torch.inference_mode():
        for start_idx in range(0, n, batch_size):
            batch_idxs = sorted_indices[start_idx : start_idx + batch_size]
            b_max_len = max(len(input_ids_list[i]) for i in batch_idxs)

            b_ids = [input_ids_list[i] + [pad_token_id] * (b_max_len - len(input_ids_list[i])) for i in batch_idxs]
            b_mask = [attention_mask_list[i] + [0] * (b_max_len - len(attention_mask_list[i])) for i in batch_idxs]

            t_ids = torch.tensor(b_ids, dtype=torch.long, device=device)
            t_mask = torch.tensor(b_mask, dtype=torch.long, device=device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, pooled = model(t_ids, t_mask)
                probs = torch.softmax(logits, dim=-1)[:, 1]

            all_probs_sorted.append(probs.float().cpu().numpy())
            if return_embeddings:
                all_embeddings_sorted.append(pooled.float().cpu().numpy())

    all_probs_sorted = np.concatenate(all_probs_sorted, axis=0)

    # Invert sorted permutation to restore exact original order
    inverse_indices = np.empty_like(sorted_indices)
    inverse_indices[sorted_indices] = np.arange(n)

    restored_probs = all_probs_sorted[inverse_indices]

    if return_embeddings:
        all_embeddings_sorted = np.concatenate(all_embeddings_sorted, axis=0)
        restored_embeddings = all_embeddings_sorted[inverse_indices]
    else:
        restored_embeddings = None

    return restored_probs, restored_embeddings


def plot_training_history(history_df: pd.DataFrame, save_path: Path):
    """Plot training loss and validation recall/AUC across epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history_df["epoch"]

    # Left: Training Loss
    ax1.plot(epochs, history_df["train_loss"], marker="o", color="tab:blue", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss (Cross-Entropy)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Right: Validation Metrics
    ax2.plot(epochs, history_df["val_template_recall"], marker="s", color="tab:green", label="val_template Recall @ FPR<=1%")
    ax2.plot(epochs, history_df["val_template_auc"], marker="^", color="tab:purple", label="val_template AUC")
    ax2.plot(epochs, history_df["val_recall"], marker="x", color="tab:orange", linestyle="--", label="val Recall @ FPR<=1%")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Validation Metrics")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    config_file = ROOT / "configs" / "train_config.yaml"
    cfg = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="Ladder Step 3: ModernBERT-base Cross-Entropy Fine-Tuning")
    parser.add_argument("--format", choices=["frege", "data_only", "plain_inst_data"], default="frege", help="Input text format")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 4)), help="Number of training epochs")
    parser.add_argument("--pooling", choices=["mean", "cls"], default=cfg.get("pooling", "mean"), help="Pooling method")
    parser.add_argument("--batch-size", type=int, default=int(cfg.get("batch_size", 32)), help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max-len", type=int, default=int(cfg.get("max_len", 512)), help="Maximum token length")
    parser.add_argument("--lr", type=float, default=float(cfg.get("lr_ce", 2.0e-5)), help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=float(cfg.get("weight_decay", 0.01)), help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=float(cfg.get("warmup_ratio", 0.06)), help="Linear warmup ratio")
    parser.add_argument("--grad-clip", type=float, default=float(cfg.get("grad_clip", 1.0)), help="Gradient clipping norm")
    parser.add_argument("--model-name", default=cfg.get("model_name", "answerdotai/ModernBERT-base"), help="Pretrained model ID")
    parser.add_argument("--dry-run", action="store_true", help="Runs 3 train steps + tiny eval on 64 rows per split without saving production artifacts")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("LADDER STEP 3: ModernBERT-base Cross-Entropy Fine-Tuning")
    print(f"Format: {args.format} | Seed: {args.seed} | Epochs: {args.epochs} | Pooling: {args.pooling}")
    print(f"Device: {device} | Model: {args.model_name} | Batch size: {args.batch_size} | LR: {args.lr}")
    if args.dry_run:
        print(">>> DRY-RUN MODE: 3 train steps + 64 rows/eval split (no production artifacts overwritten) <<<")
    print("=" * 80)

    # Load data splits
    dfs = {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in ALL_SPLITS}
    if args.dry_run:
        # Sample balanced rows so binary classification metrics are non-trivial
        train_df = pd.concat([
            dfs["train"][dfs["train"]["label"] == 0].head(48),
            dfs["train"][dfs["train"]["label"] == 1].head(48),
        ]).reset_index(drop=True)

        eval_dfs = {}
        for s in EVAL_SPLITS:
            s_df = dfs[s]
            s_0 = s_df[s_df["label"] == 0].head(32)
            s_1 = s_df[s_df["label"] == 1].head(32)
            eval_dfs[s] = pd.concat([s_0, s_1]).reset_index(drop=True)
    else:
        train_df = dfs["train"]
        eval_dfs = dfs

    print(f"Loaded splits: train={len(train_df)}, val={len(eval_dfs['val'])}, val_template={len(eval_dfs['val_template'])}, "
          f"test={len(eval_dfs['test'])}, test_template={len(eval_dfs['test_template'])}, "
          f"test_real={len(eval_dfs['test_real'])}, test_ood={len(eval_dfs['test_ood'])}")

    # Load tokenizer & model
    print(f"Loading tokenizer & encoder ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_token_id = tokenizer.pad_token_id or 50283

    encoder = AutoModel.from_pretrained(
        args.model_name,
        attn_implementation="sdpa",
    )
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = ModernBertClassifier(encoder=encoder, pooling=args.pooling).to(device)

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Pre-tokenize training data once
    print("Tokenizing training set...")
    train_texts = format_input(train_df, fmt=args.format).tolist()
    train_labels = train_df["label"].to_numpy()
    encoded_train = tokenizer(
        train_texts,
        truncation=True,
        max_length=args.max_len,
        padding=False,
        return_attention_mask=True,
    )
    train_ids = encoded_train["input_ids"]
    train_masks = encoded_train["attention_mask"]
    train_lens = np.array([len(x) for x in train_ids])

    num_training_steps = 3 if args.dry_run else math.ceil(len(train_df) / args.batch_size) * args.epochs
    num_warmup_steps = 1 if args.dry_run else int(args.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_vt_recall = -1.0
    best_vt_auc = -1.0
    best_threshold = 0.5
    best_epoch = -1
    history_records = []
    ckpt_dir = ROOT / "checkpoints" / f"ce_{args.format}_seed{args.seed}"

    num_epochs = 1 if args.dry_run else args.epochs
    global_step = 0

    print(f"\nStarting training ({num_epochs} epoch(s), {num_training_steps} total steps, {num_warmup_steps} warmup steps)...")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_total = 0.0
        train_steps = 0

        batches = get_length_bucketed_batches(
            train_lens,
            batch_size=args.batch_size,
            bucket_size=256,
            seed=args.seed,
            epoch=epoch,
        )

        for step, batch_indices in enumerate(batches):
            if args.dry_run and global_step >= 3:
                break

            t_ids, t_mask, t_labels = collate_batch(
                batch_indices, train_ids, train_masks, train_labels, pad_token_id, device
            )

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(t_ids, t_mask)
                loss = criterion(logits, t_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item()
            train_steps += 1
            global_step += 1

            if (step + 1) % 25 == 0 or (step + 1) == len(batches) or args.dry_run:
                elapsed = time.time() - epoch_start
                current_lr = scheduler.get_last_lr()[0]
                print(f"  [Epoch {epoch}/{num_epochs} | Step {step+1:>3}/{len(batches)}] "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Elapsed: {elapsed:5.1f}s")

        avg_train_loss = train_loss_total / max(train_steps, 1)
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} finished in {epoch_time:.1f}s | Avg Train Loss: {avg_train_loss:.4f}")

        # Validation: Evaluate ONLY on val and val_template (NEVER touch test* splits during training)
        t_val_start = time.time()
        vt_probs, _ = run_inference_on_split(
            eval_dfs["val_template"], model, tokenizer, device,
            fmt=args.format, batch_size=args.eval_batch_size, max_len=args.max_len,
        )
        v_probs, _ = run_inference_on_split(
            eval_dfs["val"], model, tokenizer, device,
            fmt=args.format, batch_size=args.eval_batch_size, max_len=args.max_len,
        )

        y_vt = eval_dfs["val_template"]["label"].to_numpy()
        thr = select_threshold_max_fpr(y_vt, vt_probs, max_fpr=0.01)

        pred_vt = vt_probs >= thr
        tp_vt = int(np.sum(pred_vt & (y_vt == 1)))
        fn_vt = int(np.sum(~pred_vt & (y_vt == 1)))
        vt_recall = tp_vt / (tp_vt + fn_vt) if (tp_vt + fn_vt) > 0 else 0.0
        try:
            vt_auc = float(roc_auc_score(y_vt, vt_probs))
        except Exception:
            vt_auc = 0.0

        y_v = eval_dfs["val"]["label"].to_numpy()
        pred_v = v_probs >= thr
        tp_v = int(np.sum(pred_v & (y_v == 1)))
        fn_v = int(np.sum(~pred_v & (y_v == 1)))
        v_recall = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
        try:
            v_auc = float(roc_auc_score(y_v, v_probs))
        except Exception:
            v_auc = 0.0

        val_elapsed = time.time() - t_val_start
        print(f"  [Validation in {val_elapsed:.1f}s] Selected Threshold (FPR<=1%): {thr:.4f}")
        print(f"  val_template Recall @ FPR<=1%: {vt_recall:.4f} | val_template AUC: {vt_auc:.4f}")
        print(f"  val Recall: {v_recall:.4f} | val AUC: {v_auc:.4f}")

        # Checkpoint selection: maximize val_template recall @ FPR <= 1%, tie-break by val_template AUC
        is_best = False
        if (vt_recall > best_vt_recall) or (np.isclose(vt_recall, best_vt_recall) and vt_auc > best_vt_auc):
            is_best = True
            best_vt_recall = vt_recall
            best_vt_auc = vt_auc
            best_threshold = thr
            best_epoch = epoch
            print(f"  >>> New Best Checkpoint at Epoch {epoch}! <<<")

            if not args.dry_run:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_template_recall": vt_recall,
                    "val_template_auc": vt_auc,
                    "threshold": thr,
                    "pooling": args.pooling,
                    "format": args.format,
                    "seed": args.seed,
                }, ckpt_dir / "best_model.pt")
                tokenizer.save_pretrained(ckpt_dir)

        history_records.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_template_recall": round(vt_recall, 4),
            "val_template_auc": round(vt_auc, 4),
            "val_recall": round(v_recall, 4),
            "val_auc": round(v_auc, 4),
            "threshold": round(thr, 4),
            "epoch_time_sec": round(epoch_time, 1),
            "is_best": is_best,
        })

    # Save history and plot
    history_df = pd.DataFrame(history_records)
    if not args.dry_run:
        history_path = REPORTS / f"ce_{args.format}_seed{args.seed}_history.csv"
        history_df.to_csv(history_path, index=False)
        fig_path = FIGURES / f"ce_{args.format}_seed{args.seed}.png"
        plot_training_history(history_df, fig_path)
        print(f"\nHistory saved to {history_path}")
        print(f"Plot saved to {fig_path}")

    # =========================================================================
    # Final Evaluation (of the best checkpoint across all evaluation splits)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION OF BEST CHECKPOINT (Epoch {best_epoch}, Threshold: {best_threshold:.4f})")
    print("=" * 80)

    if not args.dry_run and (ckpt_dir / "best_model.pt").exists():
        ckpt = torch.load(ckpt_dir / "best_model.pt", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_threshold = ckpt.get("threshold", best_threshold)

    model.eval()
    all_eval_probs = {}
    saved_embeddings = {}

    for s in EVAL_SPLITS:
        t0 = time.time()
        need_embs = (s in ("test_real", "test_template"))
        probs, embs = run_inference_on_split(
            eval_dfs[s],
            model,
            tokenizer,
            device,
            fmt=args.format,
            batch_size=args.eval_batch_size,
            max_len=args.max_len,
            return_embeddings=need_embs,
        )
        all_eval_probs[s] = probs
        if need_embs:
            saved_embeddings[s] = embs
        print(f"  [{s:<14}] {len(eval_dfs[s]):>5} rows evaluated in {time.time() - t0:5.2f}s")

    model_label = f"Step 3 (CE Fine-Tuned, seed={args.seed})"
    results_df = evaluate_all_splits(
        model_name=model_label,
        fmt=args.format,
        split_probs=all_eval_probs,
        split_dfs=eval_dfs,
        threshold=best_threshold,
        eval_splits=EVAL_SPLITS,
        save_to_csv=not args.dry_run,
    )

    cols_to_print = ["split", "n", "threshold", "auc", "recall", "precision", "f1", "fpr", "hn_fpr", "hn_test_only_fpr",
                     "real_macro_recall", "real_ci_low", "real_ci_high", "real_delta_mean", "real_delta_ci_low", "real_delta_ci_high"]
    available_cols = [c for c in cols_to_print if c in results_df.columns]
    print("\nLADDER STEP 3 EVALUATION RESULTS:")
    print(results_df[available_cols].to_string(index=False))

    # Save test_real and test_template embeddings for UMAP
    if not args.dry_run and "test_real" in saved_embeddings and "test_template" in saved_embeddings:
        emb_file = EMBEDDINGS / f"ce_{args.format}_seed{args.seed}.npz"
        np.savez_compressed(
            emb_file,
            test_real_embeddings=saved_embeddings["test_real"],
            test_real_label=eval_dfs["test_real"]["label"].to_numpy(),
            test_real_attack_family=eval_dfs["test_real"]["attack_family"].to_numpy(),
            test_template_embeddings=saved_embeddings["test_template"],
            test_template_label=eval_dfs["test_template"]["label"].to_numpy(),
            test_template_attack_family=eval_dfs["test_template"]["attack_family"].to_numpy(),
        )
        print(f"\nSaved test_real and test_template embeddings to {emb_file}")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"\nPeak GPU memory: {peak_mb:.2f} MB")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Verified end-to-end forward, backward, length-bucketed batching, eval, and thresholding.")


if __name__ == "__main__":
    main()
