"""GPU smoke test for ModernBERT-base on 6GB VRAM.

Verifies:
- bfloat16 loading with sdpa attention implementation
- Gradient checkpointing enabled
- Forward + backward pass with mean pooling and 768->128 projection head at seq_len=512
- Batch sizes 16 and 32 (graceful OOM catch)
- Peak allocated GPU memory (GB)
- Step latency (ms) and single-example inference latency (ms)
"""
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class ModernBERTWithProjection(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.bert = base_model
        self.head = nn.Linear(768, 128, dtype=torch.bfloat16)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)


def measure_single_example_inference(model: ModernBERTWithProjection, device: str, seq_len: int = 512) -> float:
    model.eval()
    dummy_ids = torch.randint(0, 1000, (1, seq_len), device=device, dtype=torch.long)
    dummy_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)

    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = model(dummy_ids, dummy_mask)
        torch.cuda.synchronize()

        n_runs = 30
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy_ids, dummy_mask)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    return ((t1 - t0) / n_runs) * 1000.0


def run_batch_test(model: ModernBERTWithProjection, batch_size: int, seq_len: int = 512, device: str = "cuda"):
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward pass
        output = model(input_ids, attention_mask)
        loss = output.sum()

        # Backward pass
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_latency_ms = (t1 - t0) * 1000.0
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # Zero grads to clean up
        model.zero_grad(set_to_none=True)

        return {
            "status": "SUCCESS",
            "batch_size": batch_size,
            "peak_mem_gb": peak_mem_gb,
            "step_latency_ms": step_latency_ms,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return {
                "status": "OOM",
                "batch_size": batch_size,
                "peak_mem_gb": None,
                "step_latency_ms": None,
                "error": str(e),
            }
        raise e


def main():
    if not torch.cuda.is_available():
        sys.exit("Error: CUDA is not available. GPU smoke test requires a GPU.")

    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Device: {device_name} ({total_mem_gb:.2f} GB total VRAM)")

    model_id = "answerdotai/ModernBERT-base"
    print(f"\nLoading {model_id} in bfloat16 on {device} with attn_implementation='sdpa'...")
    try:
        base_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device)
    except Exception:
        base_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)

    print("Enabling gradient checkpointing...")
    base_model.gradient_checkpointing_enable()

    model = ModernBERTWithProjection(base_model).to(device)

    # Measure single-example inference latency
    print("\nMeasuring single-example inference latency (seq_len=512)...")
    single_infer_ms = measure_single_example_inference(model, device=device, seq_len=512)
    print(f"Single-example inference latency: {single_infer_ms:.2f} ms")

    # Run forward + backward pass for batch sizes 16 and 32
    print("\nRunning forward + backward pass tests at seq_len=512:")
    results = {}
    for bs in [16, 32]:
        print(f"\nTesting batch_size={bs}...")
        res = run_batch_test(model, batch_size=bs, seq_len=512, device=device)
        results[bs] = res
        if res["status"] == "SUCCESS":
            print(f"  Batch size {bs}: SUCCESS")
            print(f"  Peak allocated GPU memory: {res['peak_mem_gb']:.3f} GB")
            print(f"  Step latency (forward + backward): {res['step_latency_ms']:.2f} ms")
        else:
            print(f"  Batch size {bs}: OUT OF MEMORY (caught gracefully)")

    # Summary table
    print("\n" + "=" * 65)
    print("GPU SMOKE TEST SUMMARY")
    print("=" * 65)
    print(f"GPU: {device_name} ({total_mem_gb:.2f} GB)")
    print(f"Dtype: torch.bfloat16 | Attention: SDPA | Gradient Checkpointing: ON")
    print(f"Single-example inference latency (seq_len=512): {single_infer_ms:.2f} ms")
    print("-" * 65)
    print(f"{'Batch Size':<12} {'Status':<10} {'Peak Memory (GB)':<20} {'Step Latency (ms)':<18}")
    print("-" * 65)
    for bs in [16, 32]:
        res = results[bs]
        if res["status"] == "SUCCESS":
            print(f"{bs:<12} {res['status']:<10} {res['peak_mem_gb']:<20.3f} {res['step_latency_ms']:<18.2f}")
        else:
            print(f"{bs:<12} {res['status']:<10} {'OOM':<20} {'N/A':<18}")
    print("=" * 65)


if __name__ == "__main__":
    main()
