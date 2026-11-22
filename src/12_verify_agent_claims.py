"""
Independent check of the claims in the agent's summary (dataset v0.2).

Run from the project root:
    python src/12_verify_agent_claims.py            # auto-finds the split parquet files
    python src/12_verify_agent_claims.py --dir data/processed

It checks, with plain pandas (no project code, so it cannot share a bug with it):
  1. which splits exist and how many rows/docs each has       (claim: 7 splits, 15,547 rows)
  2. val_template vs test_template share 0 doc_ids            (claim)
  3. ... and 0 wording_ids                                    (claim: 12 vs 13 held-out wordings)
  4. no split pair shares a doc_id, and no identical text
  5. test_real has 119 distinct real sentences                (claim)
  6. no deepset_real payload appears outside test_real
  7. train never contains a wording that val/test_template hold out
Exit code 1 if any check fails.
"""
import argparse, glob, os, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=None)
args = ap.parse_args()

# ---- load every split ---------------------------------------------------
cands = [args.dir] if args.dir else ["data/processed", "data", "."]
files = []
for d in cands:
    files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
    if files:
        break
if not files:
    sys.exit("no .parquet files found - pass --dir")

frames = {}
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    df = pd.read_parquet(f)
    if "split" in df.columns and df["split"].nunique() > 1:      # one combined file
        for s, g in df.groupby("split"):
            frames[s] = g
    else:
        frames[name] = df
print("splits found:", {k: len(v) for k, v in frames.items()})

fails = []
def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        fails.append(msg)

def col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

total = sum(len(v) for v in frames.values())
check(len(frames) == 7, f"7 splits (found {len(frames)})")
check(total == 15547, f"15,547 rows in total (found {total})")

# ---- val_template / test_template --------------------------------------
vt, tt = frames.get("val_template"), frames.get("test_template")
if vt is None or tt is None:
    check(False, "val_template and test_template exist")
else:
    print(f"val_template  rows={len(vt)} docs={vt['doc_id'].nunique()}")
    print(f"test_template rows={len(tt)} docs={tt['doc_id'].nunique()}")
    check(vt["doc_id"].nunique() == 1009, "val_template has 1,009 docs")
    check(tt["doc_id"].nunique() == 1148, "test_template has 1,148 docs")
    check(not (set(vt["doc_id"]) & set(tt["doc_id"])), "val_template and test_template share 0 doc_ids")
    w = col(vt, "wording_id")
    if w and w in tt.columns:
        a, b = set(vt[w].dropna()), set(tt[w].dropna())
        print(f"held-out wording ids: val={len(a)} test={len(b)} shared={sorted(a & b)}")
        check(len(a) == 12, f"val_template has 12 wordings (found {len(a)})")
        check(len(b) == 13, f"test_template has 13 wordings (found {len(b)})")
        check(not (a & b), "val_template and test_template share 0 wording_ids")
        tr = frames.get("train")
        if tr is not None and w in tr.columns:
            leak = (a | b) & set(tr[w].dropna())
            check(not leak, f"train contains none of the held-out wordings ({sorted(leak)[:5]})")
    else:
        check(False, "wording_id column present in val_template and test_template")

# ---- every pair of splits ----------------------------------------------
names = list(frames)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = frames[names[i]], frames[names[j]]
        shared_docs = set(a["doc_id"]) & set(b["doc_id"])
        check(not shared_docs, f"{names[i]} / {names[j]}: no shared doc_id ({len(shared_docs)})")
        tcol = col(a, "text")
        if tcol:
            shared_text = set(a[tcol]) & set(b[tcol])
            check(not shared_text, f"{names[i]} / {names[j]}: no identical text ({len(shared_text)})")

# ---- real attacks -------------------------------------------------------
tr_real = frames.get("test_real")
fam = col(next(iter(frames.values())), "attack_family")
if tr_real is not None and fam:
    real = tr_real[tr_real[fam] == "deepset_real"]
    pcol = col(real, "payload", "attack_text", "injected_text")
    if pcol:
        n = real[pcol].str.strip().str.lower().nunique()
        check(n == 119, f"test_real has 119 distinct real sentences (found {n})")
        real_set = set(real[pcol].str.strip().str.lower())
        for k, v in frames.items():
            if k == "test_real" or pcol not in v.columns:
                continue
            o = v[v[fam] == "deepset_real"][pcol].dropna().str.strip().str.lower()
            check(not (set(o) & real_set), f"{k}: no real deepset payload outside test_real")
    else:
        print("NOTE  no payload column found - skipped distinct-sentence check "
              f"(columns: {list(real.columns)})")
    others = [k for k, v in frames.items() if k != "test_real" and (v[fam] == "deepset_real").any()]
    check(not others, f"deepset_real rows only in test_real (also in: {others})")

print()
print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED:\n  - " + "\n  - ".join(fails))
sys.exit(1 if fails else 0)