import torch
import time
import pandas as pd
import psutil
import os

def make_timing_hook():
    def hook(module, input, output):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        end = time.perf_counter()
        if not hasattr(module, "_forward_times"):
            module._forward_times = []
        module._forward_times.append(end - start)
    return hook

def register_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module) and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_timing_hook()))
    return hooks

def report_forward_times(model, filename="forward_times.csv"):
    records = []
    for name, module in model.named_modules():
        if hasattr(module, "_forward_times"):
            times = module._forward_times
            avg_time = sum(times) / len(times)
            records.append({
                'layer_name': name,
                'avg_forward_time_ms': avg_time * 1000,
                'num_calls': len(times)
            })
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Saved forward timing report to {filename}")
    print(df)

def export_layerwise_sparsity(model, filename="sparsity_snapshot.csv"):
    records = []
    for name, param in model.named_parameters():
        total = param.numel()
        zeros = (param < 1e-8).sum().item()
        sparsity = (zeros / total) * 100 if total > 0 else 0
        records.append({
            'layer_name': name,
            'num_elements': total,
            'sparsity': sparsity
        })
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Saved sparsity snapshot to {filename}")
    print(df)

def get_cpu_memory_mb():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024**2
    print(f"CPU Memory Usage: {mem_usage:.2f} MB")
    return mem_usage

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        mem_usage = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Max GPU Memory Used: {mem_usage:.2f} MB")
        return mem_usage
    return 0.0
