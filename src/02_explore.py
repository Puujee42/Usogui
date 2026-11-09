"""Day 4: look at the data WITHOUT cleaning it.

Writes:
  figures/length_distribution.png   - word-count histograms per group
  reports/data_stats.csv            - summary numbers
  notes/data-exploration-auto.md    - random examples to READ by hand

Run from the repo root:  python src/02_explore.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = Path("data/raw")
for d in ("figures", "reports", "notes"):
    Path(d).mkdir(exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

squad = pd.read_parquet(RAW / "squad_v2_train.parquet")
inj = pd.concat(
    [pd.read_parquet(p).assign(split=p.stem.split("_")[-1]) for p in sorted(RAW.glob("prompt_injections_*.parquet"))],
    ignore_index=True,
)

groups = {
    "squad contexts (unique)": squad["context"].drop_duplicates().tolist(),
    "squad questions": squad["question"].drop_duplicates().tolist(),
    "deepset benign (label 0)": inj.loc[inj.label == 0, "text"].tolist(),
    "deepset injection (label 1)": inj.loc[inj.label == 1, "text"].tolist(),
}


def stats(texts):
    s = pd.Series(texts)
    words = s.str.split().str.len()
    chars = s.str.len()
    non_ascii = s.apply(lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1))
    return {
        "n": len(s),
        "words_mean": round(words.mean(), 1),
        "words_median": float(words.median()),
        "words_max": int(words.max()),
        "chars_mean": round(chars.mean(), 1),
        "share_with_non_ascii": round((non_ascii > 0).mean(), 3),
    }


table = pd.DataFrame({k: stats(v) for k, v in groups.items()}).T
table.to_csv("reports/data_stats.csv")
print(table.to_string())
print("\ndeepset label counts per split:")
print(inj.groupby(["split", "label"]).size().unstack(fill_value=0))

# --- length histogram ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
bins = np.logspace(0, 3.2, 40)
for name, texts in groups.items():
    ax.hist([len(t.split()) for t in texts], bins=bins, alpha=0.5, label=name, density=True)
ax.set_xscale("log")
ax.set_xlabel("words per text (log scale)")
ax.set_ylabel("density")
ax.set_title("Text length by group")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/length_distribution.png", dpi=150)

# --- examples to read by hand -------------------------------------------------
with open("notes/data-exploration-auto.md", "w", encoding="utf-8") as f:
    f.write("# Data exploration - random examples (seed 42)\n\n")
    f.write("Read these by hand. Questions to answer in your own notes:\n")
    f.write("- How do attacks differ from normal text (length, language, wording)?\n")
    f.write("- Are there benign texts that look like attacks (imperatives, 'ignore', 'prompt')?\n")
    f.write("- Is the language mixed (e.g. German vs English)?\n\n")
    for name, texts in groups.items():
        f.write(f"## {name}\n\n")
        for t in rng.choice(texts, size=min(30, len(texts)), replace=False):
            f.write("- " + t.replace("\n", " ")[:300] + "\n")
        f.write("\n")
print("\nWrote figures/length_distribution.png, reports/data_stats.csv, notes/data-exploration-auto.md")
