"""Day 5: assemble everything into the Frege format, with marker escaping and validation.

Run from the repo root:  python src/07_frege_format.py
"""
import pandas as pd

from common import COLUMNS, INTERIM, is_well_formed, wrap

parts = [pd.read_parquet(INTERIM / f) for f in ("benign.parquet", "attacks.parquet", "hard_negatives.parquet")]
df = pd.concat(parts, ignore_index=True)[COLUMNS]

df["text"] = [wrap(i, d) for i, d in zip(df["instruction"], df["data"])]
bad = ~df["text"].map(is_well_formed)
if bad.any():
    raise SystemExit(f"{bad.sum()} examples have broken markers - escaping failed:\n{df[bad].head()}")

before = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"examples: {len(df)} (dropped {before - len(df)} exact duplicates)")
print("\nby variant:\n" + df["variant"].value_counts().to_string())
print("\nby label:\n" + df["label"].value_counts().to_string())

# how many forged markers were neutralised (visible as '(FUNCTION_START)' etc. in the data)
neutralised = df["data"].str.contains(r"\[\s*(?:FUNCTION|ARGUMENT)\s*_\s*(?:START|END)\s*\]", case=False, regex=True).sum()
print(f"\nexamples whose RAW data contained forged markers (escaped in 'text'): {neutralised}")

df.to_parquet(INTERIM / "formatted.parquet", index=False)
print("\nexample:\n" + df.loc[df["attack_family"] == "fake_markers", "text"].head(1).to_string(index=False)[:600])
