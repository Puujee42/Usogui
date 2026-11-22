# CLAUDE.md — Frege prompt-injection detector (thesis project)

Read this fully before touching anything. It is the project's memory.

## 1. What this project is
Final-year thesis (Шинэ Монгол Технологийн Коллеж): a prompt-injection detector.
Idea: treat the **instruction as a function and the data as its argument** (Frege).
Injection = the *data* slot trying to act like an instruction.

Input format (exact):
```
[FUNCTION_START] instruction [FUNCTION_END] [ARGUMENT_START] data [ARGUMENT_END]
```
- Forged markers inside raw data are **escaped** in `text`. Never remove the escaping.
- Model: ModernBERT-base, ladder of experiments below. Hardware: RTX 3050 6 GB laptop, Python 3.14, PyTorch 2.15+cu132.

## 2. Working rules (most important)
1. **Never change data, tests or thresholds to make a number look better.** If a number disagrees with a claim, report the fact and stop.
2. **Never edit `tests/` or `tests/mutation_check.py` to make them pass.** A failing audit means the data or the build script is wrong.
3. **Never tune anything on `test_*` splits.** Threshold is chosen on `val_template` only.
4. Do not silently rewrite scripts you did not create. If a script must change, say what and why, show the diff, run the tests.
5. Report what you ran with the **complete output**, not a summary. Separate *measured* from *expected*.
6. Do not overclaim. Frozen-encoder result ≠ "fine-tuning is necessary". A result is only a result for what was actually run.
7. Keep Mongolian text in decks/notes; code, comments and logs in English.
8. Never put system info (local IP, hostname screenshots) into any deliverable.

## 3. Dataset v0.2 (verified)
Files: `data/interim/formatted.parquet` (15,547 rows, before split) → split by `src/08_split_and_check.py` → **15,437 rows in 7 splits**
(110 held-out hard-negative rows are dropped by 08 because of conflicts — this is expected, not a bug).

| split | purpose |
|---|---|
| train | training |
| val | seen-wording validation |
| test | seen-wording test (expected near-perfect for TF-IDF; weak evidence) |
| val_template | **threshold selection**: 1,009 docs, 12 held-out wordings |
| test_template | **unseen-wording test**: 1,148 docs, 13 held-out wordings |
| test_real | real deepset attacks: 119 distinct sentences (767 attack rows) |
| test_ood | held-out attack family (phishing) |

Hard guarantees (checked by `tests/` and `src/12_verify_agent_claims.py`):
- no shared `doc_id` or identical text between any two splits
- `val_template` and `test_template` share 0 docs and 0 held-out wordings
- train/val/test contain no held-out-flagged rows or wordings
- every `deepset_real` row lives only in `test_real`
- "119 distinct sentences" is counted **after** `normalize_payload` (448 raw strings → 119). Always reuse that function.
- `test_real` / `test_ood` also contain template held-out wordings: eval-only, allowed.
- Benign side of `test_real` is synthetic (SQuAD passages): its FPR is **not** a real-world FPR.

Schema notes: columns include `doc_id`, `text`, `label` (0/1), `attack_family`, `hn_type`
(`benign_imperative`, `benign_link`, `keyword_natural`), `wording_id`, held-out flag columns, `position`, `task`.
Unflagged wording ids (`none`, `atk:*:0`, `hn:*:N`) are normal training wordings and legitimately appear in several splits.

## 4. Evaluation protocol (do not change)
- Threshold policy: **highest recall subject to FPR ≤ 1% on `val_template`**, fixed before touching test splits.
- Headline metric: **macro recall per distinct real sentence on `test_real`** with 95% bootstrap CI over the 119 sentences; also AUC, `test_template` recall, `test_ood` recall.
- Model comparison: **paired bootstrap over the 119 sentences**, Δ vs Step 1. Success = lower bound of the Δ interval > 0 (not "> 0.631").
- Always also print: hard-negative FPR per `hn_type`, instruction-only and length-only sanity baselines (AUC ≈ 0.5 expected).
- 3 seeds for final claims; report mean ± spread.

## 5. Experiment ladder
| Step | Model | Status / result |
|---|---|---|
| 1 | TF-IDF + LogReg | `test_template` 98.5%, `test_real` 49.2% [41.5–56.6], `test_ood` 18.1% |
| 2 | Frozen ModernBERT + LogReg (C=0.1) | 53.9% / 30.2% [24.8–35.9] / 3.6%; paired Δ −0.190 [−0.253, −0.132] |
| 3 | CE fine-tune (`src/22_train_ce.py`) | set up, dry-run OK (3.34 GiB); **full run pending** |
| 4 | SupCon (768→128 projection, batch 32–64) | not started |
| 5 | Format ablation: `data_only` / `plain_inst_data` / `frege` | not started |
Diagnostics: C=1.0→0.1 raised val AUC 0.9570→0.9632; long SQuAD passages dilute frozen embeddings (isolated strings 1.000 vs passages 0.963/0.970). Treat as hypothesis until Step 3 confirms.

## 6. Hardware / training settings
- `max_len = 512` (median 192, p99 407; 0.3% rows truncate, 0 attack sentences cut).
- bf16 + sdpa + **gradient_checkpointing=True (required**; batch 16 OOMs without it).
- batch 32 ≈ 2.8–3.3 GiB, batch 64 ≈ 4.2 GiB of ~5 GiB free → SupCon needs no GradCache.
- GPU is power-capped at **30 W / P5** by the 150 W charger.
- Latency: GPU kernel 7.8–25 ms; torch.compile wall-clock 70 / 141 / 148 ms (192/407/512 tok).
  Targets documented honestly: **<50 ms at stock TDP, <250 ms at the 30 W cap**. Report both; never present the capped number as stock.

## 7. Repo map
- `src/04…09` build + split + shortcut checks; `08_split_and_check.py` refuses to run without held-out flag columns.
- `src/10_real_eval.py` real-sentence report; `11_baseline_all_splits.py`; `12_verify_agent_claims.py` independent audit.
- `src/20_smoke_test.py`, `21`, `21b` (probe on train/val only), `22_train_ce.py`; `eval_utils.py`; `configs/train_config.yaml`.
- `tests/test_frege_format.py`, `tests/test_no_leakage.py`, `tests/mutation_check.py` (every planted leak must be CAUGHT).
- `notes/dataset-spec.md`, `notes/hardware-limit.md`, `reports/` (logs, `ladder_results.csv`).

## 8. Commands (fish shell, CachyOS)
```fish
source .venv/bin/activate.fish
python -m pytest tests -q -rs          # expect 26 passed, 0 skipped
python tests/mutation_check.py         # expect every planted leak CAUGHT
python src/12_verify_agent_claims.py   # expect: ALL LEAK CHECKS PASSED
python src/22_train_ce.py --format frege --seed 42
```
Run the three audit commands **after any change to data/split code** and before any training.

## 9. Known open issues / to do
- Step 3 full run results not yet reviewed → then Step 5 ablation, Step 4 SupCon, 3 seeds, UMAP, error analysis, latency measurement.
- Real-world FPR is unmeasured (needs real benign text: emails, web pages, user prompts).
- 30 of 119 real sentences were never detected by TF-IDF; error analysis should group them (non-English, short commands, "blame X" style).
- Confirm the 13 template wordings inside `test_real`/`test_ood` are `test_template`'s, not `val_template`'s.
- Remove any terminal screenshot with system info from the master deck's closing slide.

## 10. Style for reports back to the user
Short, honest, in this order: what ran → full output → table (claim | claimed | found | verdict) → what is NOT verified → next step.
