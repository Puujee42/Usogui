"""Day 2: benign examples = a squad passage + a normal task instruction.

Run from the repo root:  python src/04_build_benign.py [--n-docs 3000]
"""
import argparse

import numpy as np
import pandas as pd

from common import INTERIM, SEED, TASKS, load_squad_docs, make_instruction

ap = argparse.ArgumentParser()
ap.add_argument("--n-docs", type=int, default=3000, help="how many unique passages to use")
args = ap.parse_args()

rng = np.random.default_rng(SEED)
docs = load_squad_docs()
docs = docs.sample(min(args.n_docs, len(docs)), random_state=SEED).reset_index(drop=True)
docs["task"] = rng.choice([t[0] for t in TASKS], size=len(docs))
docs["instruction"] = [make_instruction(t, q) for t, q in zip(docs["task"], docs["question"])]

out = pd.DataFrame({
    "doc_id": docs["doc_id"], "variant": "benign", "instruction": docs["instruction"],
    "data": docs["context"], "label": 0, "attack_family": "none", "hn_type": "none",
    "position": "none", "payload": "", "task": docs["task"], "source": "squad_v2",
    "template_test_only": False, "hn_test_only": False,
})
out.to_parquet(INTERIM / "benign.parquet", index=False)
print(f"benign examples: {len(out)}")
print(out["task"].value_counts().to_string())
