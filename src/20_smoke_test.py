import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="answerdotai/ModernBERT-base")
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--sizes", default="4,8,16,32,64", help="micro-batch sizes to try, comma separated")
ap.add_argument("--samples", type=int, default=3000)
ap.add_argument("--lens", default="256,384,512,768,1024", help="max_len options to compare in part A")
ap.add_argument("--no-model", action="store_true", help="only run the tokenizer analysis (part A)")
args = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model)
df = pd.read_parquet(PROCESSED / "train.parquet")
df = df.sample(min(args.samples, len(df)), random_state=42).reset_index(drop=True)

# ---------------- Part A: lengths and truncation ----------------------------------------------------------------
enc = tok(df["text"].tolist(), truncation=False, return_offsets_mapping=True)
lens = np.array([len(x) for x in enc["input_ids"]])
print(f"== PART A: token lengths of the Frege-formatted `text` ({len(df)} train rows, tokenizer: {args.model})")
print("   percentiles  p50 {:.0f} | p90 {:.0f} | p95 {:.0f} | p99 {:.0f} | max {}".format(*np.percentile(lens, [50, 90, 95, 99]), lens.max()))

att = [i for i in range(len(df)) if df.loc[i, "label"] == 1 and str(df.loc[i, "payload"]).strip()]
found = [(i, df.loc[i, "text"].find(str(df.loc[i, "payload"]).strip())) for i in att]
found = [(i, p) for i, p in found if p >= 0]          # rows whose payload was rewritten by marker escaping are skipped
print(f"   attack rows checked for payload visibility: {len(found)} of {len(att)} (forged-marker rows are skipped)")
print(f"\n   {'max_len':>8} {'rows truncated':>16} {'attack sentence cut off':>25}")
for L in [int(x) for x in args.lens.split(",")]:
    trunc = (lens > L).mean()
    cut = 0
    for i, p in found:
        off = enc["offset_mapping"][i]
        if len(off) > L:
            last_end = off[L - 2][1]                  # last content token kept (the final slot holds [SEP])
            cut += (p + len(str(df.loc[i, "payload"]).strip())) > last_end
    print(f"   {L:>8} {trunc:>15.1%} {cut / max(len(found), 1):>24.1%}")
print("   -> choose the smallest max_len whose 'attack sentence cut off' is ~0%.\n")

if args.no_model:
    raise SystemExit(0)

# ---------------- Part B: GPU memory, micro-batch, latency ---------------------------------------------------------
import torch  # noqa: E402
from transformers import AutoModel  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"== PART B: model {args.model} on {dev}; torch {torch.__version__}")
if dev == "cuda":
    free, total = torch.cuda.mem_get_info()
    print(f"   GPU {torch.cuda.get_device_name(0)} | free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB"
          f" (the desktop already holds the rest)")


class Clf(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc, self.head = enc, torch.nn.Linear(enc.config.hidden_size, 2)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))      # mean pooling


enc_model = AutoModel.from_pretrained(args.model)
print("   attention implementation:", getattr(enc_model.config, "_attn_implementation", "unknown"))
try:
    enc_model.gradient_checkpointing_enable()
    print("   gradient checkpointing: on")
except Exception as e:  # noqa: BLE001
    print("   gradient checkpointing NOT available:", e)
model = Clf(enc_model).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
texts, labels = df["text"].tolist(), torch.tensor(df["label"].tolist())


def batch(n, start=0):
    idx = [(start + k) % len(texts) for k in range(n)]
    b = tok([texts[i] for i in idx], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
    return b["input_ids"].to(dev), b["attention_mask"].to(dev), labels[idx].to(dev)


amp = torch.autocast(device_type=dev, dtype=torch.bfloat16)
best = None
print(f"\n   one full training step at max_len {args.max_len} (bf16 autocast, fp32 weights, AdamW):")
for n in [int(x) for x in args.sizes.split(",")]:
    try:
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model.train()
        ids, mask, y = batch(n)
        t0 = time.time()
        with amp:
            loss = torch.nn.functional.cross_entropy(model(ids, mask).float(), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if dev == "cuda":
            torch.cuda.synchronize()
            mem = f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak"
        else:
            mem = "cpu"
        print(f"   micro-batch {n:>3}: OK   {time.time() - t0:5.1f} s   {mem}   (padded to {ids.shape[1]} tokens)")
        best = n
    except torch.cuda.OutOfMemoryError:
        print(f"   micro-batch {n:>3}: OUT OF MEMORY")
        break
print(f"   -> largest micro-batch that fits: {best}\n")

model.eval()
for label, L in (("median-length", int(np.percentile(lens, 50))), ("p99-length", int(min(np.percentile(lens, 99), args.max_len)))):
    ids = torch.randint(1000, 2000, (1, L), device=dev)
    mask = torch.ones_like(ids)
    times = []
    with torch.inference_mode(), amp:
        for k in range(120):
            t0 = time.perf_counter()
            model(ids, mask)
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times[20:])
    print(f"   inference, 1 input, {label} ({L} tokens): median {np.median(t):.1f} ms | p95 {np.percentile(t, 95):.1f} ms   (target < 50 ms)")