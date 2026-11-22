"""Step 5 format ablation: paired comparison between CE checkpoints trained on different input formats.

Read-only with respect to data, tests, thresholds and existing reports. It writes only to
reports/compare_formats/ (per-row probability caches and a per-sentence table).

For each format it loads checkpoints/ce_<format>_seed<seed>/best_model.pt, uses the threshold
stored in that checkpoint (selected on val_template during training, never on a test split),
scores test_template / test_real / test_ood and reports:
  - paired bootstrap over the 119 distinct real sentences: frege minus each other format
    (eval_utils.compute_paired_real_bootstrap: same normalize_payload, 2000 resamples, seed 42)
  - test_real macro recall + 95% CI, test_ood recall, FPR
  - hard-negative FPR per hn_type per split
It also checks that the recomputed recall/FPR match the rows in reports/ladder_results.csv.

Usage (from the project root, after the training queue):
    python scripts/compare_formats.py                      # frege vs data_only, plain_inst_data, seed 42
    python scripts/compare_formats.py --seed 1337
    python scripts/compare_formats.py --formats frege      # score / verify one format only
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval_utils import (  # noqa: E402
    compute_paired_real_bootstrap,
    compute_real_sentence_metrics,
    normalize_payload,
)

_spec = importlib.util.spec_from_file_location("train_ce", ROOT / "src" / "22_train_ce.py")
train_ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_ce)  # guarded by __main__, so this only defines functions/classes

SPLITS = ["test_template", "test_real", "test_ood"]
OUT = ROOT / "reports" / "compare_formats"
LADDER = ROOT / "reports" / "ladder_results.csv"
EVAL_BATCH = 64  # same as 22_train_ce.py --eval-batch-size default, so bf16 padding matches the run


def load_probs(fmt: str, seed: int, dfs: dict, device: str):
    """Return ({split: probs}, threshold) for one checkpoint, using a cache under reports/compare_formats/."""
    ckpt_dir = ROOT / "checkpoints" / f"ce_{fmt}_seed{seed}"
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path} (has the {fmt} seed {seed} run finished?)")
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")  # CPU: avoids the extra ~1.7 GiB on GPU
    if ck.get("format") != fmt:
        raise ValueError(f"{ckpt_path} was trained with format={ck.get('format')!r}, expected {fmt!r}")
    thr = float(ck["threshold"])
    cache = OUT / f"probs_ce_{fmt}_seed{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        if float(z["threshold"]) == thr and all(s in z.files and len(z[s]) == len(dfs[s]) for s in SPLITS):
            print(f"[{fmt}] using cached probabilities {cache.relative_to(ROOT)}")
            return {s: z[s] for s in SPLITS}, thr, int(ck["epoch"])
    print(f"[{fmt}] scoring {ckpt_path.relative_to(ROOT)} (epoch {ck['epoch']}, threshold {thr:.6g})", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    enc = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa")
    model = train_ce.ModernBertClassifier(enc, pooling=ck.get("pooling", "mean"))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    del ck
    probs = {}
    for s in SPLITS:
        probs[s], _ = train_ce.run_inference_on_split(
            dfs[s], model, tok, device, fmt=fmt, batch_size=EVAL_BATCH, max_len=512, return_embeddings=False
        )
        print(f"[{fmt}]   scored {s}", flush=True)
    del model, enc
    if device == "cuda":
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, threshold=thr, **probs)
    return probs, thr, None


def check_against_ladder(fmt: str, seed: int, probs: dict, thr: float, dfs: dict):
    """Compare recomputed recall/FPR with the row 22_train_ce.py wrote; returns list of mismatches."""
    if not LADDER.exists():
        return ["ladder_results.csv not found"]
    lad = pd.read_csv(LADDER)
    rows = lad[(lad["model"] == f"Step 3 (CE Fine-Tuned, seed={seed})") & (lad["format"] == fmt)]
    bad = []
    for s in SPLITS:
        r = rows[rows["split"] == s]
        if r.empty:
            bad.append(f"{s}: no ladder row")
            continue
        y = dfs[s]["label"].to_numpy()
        pred = probs[s] >= thr
        for name, val in (("recall", pred[y == 1].mean()), ("fpr", pred[y == 0].mean())):
            if abs(round(float(val), 4) - float(r[name].iloc[0])) > 1e-4:
                bad.append(f"{s} {name}: recomputed {val:.4f} vs ladder {float(r[name].iloc[0]):.4f}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=["frege", "data_only", "plain_inst_data"],
                    help="first format is the reference; deltas are reference minus each other format")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dfs = {s: pd.read_parquet(ROOT / "data" / "processed" / f"{s}.parquet") for s in SPLITS}
    real = dfs["test_real"]

    missing = [f for f in args.formats
               if not (ROOT / "checkpoints" / f"ce_{f}_seed{args.seed}" / "best_model.pt").exists()]
    if missing:
        sys.exit(f"missing checkpoints for seed {args.seed}: {missing}. Run the training queue first.")

    res = {}
    for fmt in args.formats:
        probs, thr, _ = load_probs(fmt, args.seed, dfs, device)
        res[fmt] = (probs, thr)

    # ---- reproduction check ------------------------------------------------------------------
    print("\nREPRODUCTION CHECK vs reports/ladder_results.csv")
    all_ok = True
    for fmt, (probs, thr) in res.items():
        bad = check_against_ladder(fmt, args.seed, probs, thr, dfs)
        all_ok &= not bad
        print(f"  {fmt:<16} " + ("OK" if not bad else "MISMATCH: " + "; ".join(bad)))

    # ---- per-format summary ------------------------------------------------------------------
    print(f"\nPER-FORMAT RESULTS (seed {args.seed}; each threshold chosen on its own val_template run)")
    rows = []
    for fmt, (probs, thr) in res.items():
        mac, lo, hi, n = compute_real_sentence_metrics(real, probs["test_real"], thr)
        r = {"format": fmt, "threshold": thr, "real_macro": mac, "real_ci_low": lo, "real_ci_high": hi, "n_sent": n}
        for s in SPLITS:
            y = dfs[s]["label"].to_numpy()
            pred = probs[s] >= thr
            r[f"{s}_recall"] = pred[y == 1].mean()
            r[f"{s}_fpr"] = pred[y == 0].mean()
        rows.append(r)
    summ = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string(index=False))

    # ---- paired deltas -----------------------------------------------------------------------
    ref = args.formats[0]
    print(f"\nPAIRED BOOTSTRAP over distinct real sentences (normalize_payload; 2000 resamples; seed 42): {ref} minus X")
    for other in args.formats[1:]:
        pr, tr = res[ref]
        po, to = res[other]
        dm, dl, dh = compute_paired_real_bootstrap(real, pr["test_real"], tr, po["test_real"], to)
        print(f"  {ref} - {other:<16} {dm:+.4f} [{dl:+.4f}, {dh:+.4f}]   lower bound > 0: {'YES' if dl > 0 else 'NO'}")

    # ---- hard-negative FPR per hn_type -------------------------------------------------------
    print("\nHARD-NEGATIVE FPR per hn_type (label 0 rows), FP/N")
    hrows = []
    for fmt, (probs, thr) in res.items():
        for s in SPLITS:
            df = dfs[s].assign(pred=probs[s] >= thr)
            neg = df[df["label"] == 0]
            is_hn = neg["hn_type"].notna() & (neg["hn_type"] != "none")
            groups = list(neg[is_hn].groupby("hn_type")) + [("(plain benign)", neg[~is_hn])]
            for t, g in groups:
                hrows.append({"format": fmt, "split": s, "hn_type": t,
                              "cell": f"{g['pred'].mean():.4f} ({int(g['pred'].sum())}/{len(g)})"})
    hn = pd.DataFrame(hrows).pivot_table(index=["split", "hn_type"], columns="format", values="cell", aggfunc="first")
    print(hn[[f for f in args.formats]].to_string())

    # ---- per-sentence table ------------------------------------------------------------------
    y = real["label"].to_numpy()
    att = real[y == 1].copy()
    att["sent"] = att["payload"].map(normalize_payload)
    for fmt, (probs, thr) in res.items():
        att[fmt] = probs["test_real"][y == 1] >= thr
    per = att.groupby("sent")[list(res)].mean()
    per.insert(0, "rows", att.groupby("sent").size())
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"real_sentences_seed{args.seed}.csv"
    per.to_csv(out_csv)
    print(f"\nnever detected (rate 0) per format: " + ", ".join(f"{f}={int((per[f] == 0).sum())}" for f in res))
    print(f"per-sentence detection rates written to {out_csv.relative_to(ROOT)}")
    if not all_ok:
        print("\nWARNING: reproduction check failed; do not trust the comparison until the mismatch is explained.")
        sys.exit(2)


if __name__ == "__main__":
    main()
