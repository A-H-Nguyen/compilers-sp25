"""
perf_utils.py

Utility functions for profiling PyTorch models and scripts:
- Module-level forward timing via hooks
- CPU memory usage
- GPU memory usage
- Context manager for timing arbitrary code blocks
"""

import time
import psutil
import os
import torch
import pandas as pd

# ----------------------------
# Sparsity Profiling
# ----------------------------
def export_layerwise_sparsity(model, filename="sparsity_snapshot.csv", threshold=1e-8):
    """
    Compute and save the sparsity (percentage of near-zero parameters) for each layer.
    Args:
        model: the PyTorch model
        filename: CSV file to write results
        threshold: values with absolute magnitude below this are considered zero
    Returns:
        pandas.DataFrame of sparsity stats
    """
    import pandas as pd
    records = []
    for name, param in model.named_parameters():
        tensor = param.data
        total = tensor.numel()
        if total == 0:
            zeros = 0
            sparsity = 0.0
        else:
            zeros = (tensor.abs() < threshold).sum().item()
            sparsity = zeros / total * 100.0
        records.append({
            "layer_name": name,
            "num_elements": total,
            "num_zeros": zeros,
            "sparsity_pct": sparsity
        })
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    return df

# ----------------------------
# Forward Timing via Hooks
# ----------------------------
def register_forward_hooks(model):
    """
    Register pre- and post- forward hooks on all leaf modules to record timings.
    Returns a list of hook handles to remove later.
    Each module will record module._forward_times = [t1, t2, ...].
    """
    handles = []
    for module in model.modules():
        # Only attach to leaf modules (no children)
        if len(list(module.children())) == 0:
            # Pre-hook to record start time
            def pre_forward(module, inp):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                module._perf_start = time.perf_counter()
            handles.append(module.register_forward_pre_hook(pre_forward))

            # Post-hook to record elapsed time
            def post_forward(module, inp, out):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - getattr(module, "_perf_start", time.perf_counter())
                if not hasattr(module, "_forward_times"):
                    module._forward_times = []
                module._forward_times.append(elapsed)
            handles.append(module.register_forward_hook(post_forward))
    return handles

def report_forward_times(model, filename="forward_times.csv"):
    """
    Aggregate and save per-layer average forward time in milliseconds.
    """
    records = []
    for name, module in model.named_modules():
        times = getattr(module, "_forward_times", None)
        if times:
            avg_ms = sum(times) / len(times) * 1000.0
            records.append({
                'layer_name': name,
                'avg_forward_time_ms': avg_ms,
                'num_calls': len(times)
            })
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    return df

# ----------------------------
# Memory Profiling
# ----------------------------
def get_cpu_memory_mb():
    """
    Returns current process CPU memory usage (RSS) in MB.
    """
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)

def reset_gpu_memory_stats():
    """
    Reset peak GPU memory stats in PyTorch.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_gpu_peak_memory_mb():
    """
    Returns peak GPU memory allocated by PyTorch in MB.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0

# ----------------------------
# Context Manager for Timing
# ----------------------------
class Timer:
    """
    Context manager to time a block of code.
    Example:
        with Timer("training loop"):
            train()
    """
    def __init__(self, name="Block", logger=print):
        self.name = name
        self.logger = logger

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        self.logger(f"{self.name} elapsed: {elapsed * 1000:.3f} ms")
