"""Does the leakage audit still bite?   Run from the repo root:  python tests/mutation_check.py

Copies tests/test_no_leakage.py and data/processed/ to a temp folder, plants ONE known leak at a time and
runs the audit on it. A mutation is CAUGHT if the audit fails. If any is MISSED, the audit is too weak
(for example somebody edited it to make it pass) and no 'no leakage' claim can be trusted.
Your real data is never modified.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC, AUDIT = ROOT / "data" / "processed", ROOT / "tests" / "test_no_leakage.py"
NAMES = ["train", "val", "test", "test_template", "test_real", "test_ood"]
PREFIX = re.compile(r"^\s*(?:note|important|p\.\s?s\.?|reminder|update)\s*[:\-\u2013\u2014]?\s*", re.I)


def fresh_copy(row, doc_id, suffix=" (copy)"):
    """A copy of `row` that is NOT an exact duplicate (new doc_id, slightly changed data/text)."""
    r = row.copy()
    r["doc_id"] = doc_id
    r["data"] = r["data"] + suffix
    r["text"] = r["text"].replace(" [ARGUMENT_END]", suffix + " [ARGUMENT_END]")
    return r


def add(d, split, row):
    d[split] = pd.concat([d[split], pd.DataFrame([row])], ignore_index=True)


def real_sentence_leak(target):
    def mutate(d):
        src = next((n for n in ("test_real", "test", "val", "train") if n in d and (d[n]["attack_family"] == "deepset_real").any()), None)
        if src is None:
            return False
        r = d[src][d[src]["attack_family"] == "deepset_real"].iloc[0].copy()
        old, new = str(r["payload"]), "P.S.: " + PREFIX.sub("", str(r["payload"]), count=1)   # prefix variant = the original bug
        host = d["val" if target == "train" else "train"].query("label == 0").iloc[0]          # different passage: only the payload leaks
        r["data"], r["text"] = host["data"] + " " + new, host["text"].replace(" [ARGUMENT_END]", " " + new + " [ARGUMENT_END]")
        r["payload"], r["doc_id"] = new, "mut_real_" + target
        add(d, target, r)
        return True
    return mutate


def heldout_family_in_train(d):
    if "test_ood" not in d:
        return False
    add(d, "train", fresh_copy(d["test_ood"][d["test_ood"]["label"] == 1].iloc[0], "mut_ood"))
    return True


def same_row_in_two_splits(d):
    add(d, "test", d["train"].iloc[0].copy())
    return True


def broken_marker(d):
    d["val"].loc[d["val"].index[0], "text"] = d["val"].iloc[0]["text"].replace("[ARGUMENT_END]", "")
    return True


def wrong_label(d):
    d["train"].loc[d["train"].index[0], "label"] = 1 - d["train"].iloc[0]["label"]
    return True


def flagged_row_in_train(d):
    for c in ("template_test_only", "hn_test_only"):
        if all(c in x.columns for x in d.values()):
            d["train"].loc[d["train"].index[0], c] = True
            return True
    return False


MUTATIONS = [
    ("real deepset sentence in TRAIN, with a changed prefix (the original bug)", real_sentence_leak("train")),
    ("real deepset sentence in VAL, with a changed prefix", real_sentence_leak("val")),
    ("held-out attack family leaks into train", heldout_family_in_train),
    ("same row present in train and test", same_row_in_two_splits),
    ("broken Frege markers in val", broken_marker),
    ("label contradicts attack_family", wrong_label),
    ("row flagged test-only sits in train", flagged_row_in_train),
]


def run_audit(d):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "tests").mkdir()
        (tmp / "data" / "processed").mkdir(parents=True)
        shutil.copy(AUDIT, tmp / "tests" / AUDIT.name)
        for n, df in d.items():
            df.to_parquet(tmp / "data" / "processed" / f"{n}.parquet", index=False)
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_no_leakage.py"],
                             cwd=tmp, capture_output=True, text=True)
    failed = re.findall(r"^FAILED tests/test_no_leakage.py::(\w+)", out.stdout, re.M)
    return out.returncode, failed, out.stdout


def main():
    if not (PROC / "train.parquet").exists():
        raise SystemExit("data/processed is missing - run scripts 04-08 first")
    load = lambda: {n: pd.read_parquet(PROC / f"{n}.parquet") for n in NAMES if (PROC / f"{n}.parquet").exists()}  # noqa: E731
    code, _, text = run_audit(load())
    if code != 0:
        raise SystemExit("The audit already FAILS on your real data - fix that first:\n" + text[-1500:])
    print("baseline: audit passes on unmodified data\n")
    missed = 0
    for title, mutate in MUTATIONS:
        d = load()
        if not mutate(d):
            print(f"  SKIPPED  {title}   (not applicable to your schema)")
            continue
        code, failed, _ = run_audit(d)
        if code == 1 and failed:
            print(f"  CAUGHT   {title}\n             by: {', '.join(failed)}")
        else:
            missed += 1
            print(f"  MISSED   {title}   <-- the audit did not fail (exit code {code})")
    print("\nVERDICT:", "the audit bites on every planted leak" if not missed else f"{missed} leak(s) MISSED - the audit is too weak")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
