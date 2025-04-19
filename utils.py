"""
For the analysis hooks, you can register them with this:

    forward_hooks = []
    for name, module in model.named_modules():
        forward_hooks.append(module.register_forward_hook(make_timing_hook()))

Or you can also check to see if the layer is of a specific type:

    forward_hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (basic.EGNN_Layer, layer_no.TimeConv, layer_no.TimeConv_x)):
            forward_hooks.append(module.register_forward_hook(make_timing_hook()))

You may have to copy this file into the directory of whichever workload you're looking at
"""

import torch
import time

# Calculate and save a running average of the runtime of a model layer
def make_timing_hook():
    def hook(module, input, output):
        if not hasattr(module, "_forward_time_total"):
            module._forward_time_total = 0.0
            module._forward_time_calls = 0
        # Record start time if not already recorded
        if not hasattr(module, "_start_time"):
            module._start_time = time.perf_counter()

        # Compute elapsed time for this call
        elapsed = time.perf_counter() - module._start_time

        # Update running totals
        module._forward_time_total += elapsed
        module._forward_time_calls += 1

    return hook

# This doesn't actually work, don't bother with it
def make_running_sparsity_hook():
    def hook(module, input, output):
        if not hasattr(module, "_sparsity_total"):
            module._sparsity_total = 0.0
            module._sparsity_calls = 0

        if not hasattr(module, "_activation_sparsity_total"):
            module._activation_sparsity_total = 0.0
            module._activation_sparsity_calls = 0

        if hasattr(module, 'weight'):
            weight = module.weight.data
            sparsity = (weight < 1e-8).sum().item() / weight.numel()
            module._sparsity_total += sparsity
            module._sparsity_calls += 1

        if isinstance(output, torch.Tensor):
            activation_sparsity = (output < 1e-8).sum().item() / output.numel()
            module._activation_sparsity_total += activation_sparsity
            module._activation_sparsity_calls += 1

    return hook

# Call this right before starting trainig
def start_timers(model):
    for module in model.modules():
        if hasattr(module, "_start_time"):
            module._start_time = time.perf_counter()

# Call this after training
def report_forward_times(model):
    print("\n=== Forward Pass Timing Summary ===")
    for name, module in model.named_modules():
        if hasattr(module, "_forward_time_total"):
            avg_time = module._forward_time_total / module._forward_time_calls
            print(f"{name}: {avg_time*1000:.4f} ms (avg over {module._forward_time_calls} calls)")

# This doesn't work either
def report_average_sparsities(model):
    print("\n=== Average Sparsity per Layer ===")
    for name, module in model.named_modules():
        if hasattr(module, "_sparsity_total") and module._sparsity_calls > 0:
            avg_sparsity = module._sparsity_total / module._sparsity_calls
            print(f"{name} weight sparsity: {avg_sparsity:.2%}")

        if hasattr(module, "_activation_sparsity_total") and module._activation_sparsity_calls > 0:
            avg_sparsity = module._sparsity_total / module._activation_sparsity_calls 
            print(f"{name} activation sparsity: {avg_sparsity:.2%}")

"""
Use this for calculating the per layer sparsity.
It will print a huge output, so just call it before and after training, depending on if the sparsity values actually change a lot.
"""
def layerwise_sparsity(model):
    print("Per-layer Model Sparsity:")
    print("=" * 55)
    
    for name, param in model.named_parameters():
        total = param.numel()
        zeros = torch.sum(param < 1e-8).item()
        sparsity = (zeros / total) * 100 if total > 0 else 0
        
        if (sparsity > 40.0):
            print(f"{name:<40} num elements: {total:>11.2f} sparsity: {sparsity:>11.2f}%")

    print("=" * 55)
