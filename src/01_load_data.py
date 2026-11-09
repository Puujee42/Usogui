"""Day 3: download the two datasets and save them as local parquet files.

Run from the repo root:  python src/01_load_data.py
"""
from pathlib import Path

from datasets import load_dataset

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)


def load_squad():
    # "squad_v2" has moved under the rajpurkar namespace on the Hub; try both names.
    for name in ("rajpurkar/squad_v2", "squad_v2"):
        try:
            print(f"loading {name} ...")
            return load_dataset(name)
        except Exception as e:  # noqa: BLE001
            print(f"  could not load {name}: {e}")
    raise SystemExit("squad_v2 failed to load - check your internet connection")


squad = load_squad()
inj = load_dataset("deepset/prompt-injections")

for split, ds in squad.items():
    ds.to_parquet(str(RAW / f"squad_v2_{split}.parquet"))
for split, ds in inj.items():
    ds.to_parquet(str(RAW / f"prompt_injections_{split}.parquet"))

print("\n=== squad_v2 ===")
for split, ds in squad.items():
    print(f"{split:>12}: {len(ds):>7} rows, columns = {ds.column_names}")
print("\n=== deepset/prompt-injections ===")
for split, ds in inj.items():
    print(f"{split:>12}: {len(ds):>7} rows, columns = {ds.column_names}")
print(f"\nSaved parquet files to {RAW.resolve()}")
