import argparse
import time
import torch
import pandas as pd
from models import Model
from dataloader import load_data
from utils import (
    set_seed,
    get_training_config,
    check_writable,
    get_evaluator,
)
from train_and_eval import run_transductive
from benchmark_utils import (
    register_hooks,
    export_layerwise_sparsity,
    report_forward_times,
    get_cpu_memory_mb,
    get_gpu_memory_mb,
)

# === New: Helper to compute sparsity ===
def compute_input_sparsity(tensor):
    total = tensor.numel()
    zeros = (tensor == 0).sum().item()
    sparsity = (zeros / total) * 100
    return sparsity


def run_experiment(name, dataset, teacher, compile_model=False):
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data
    g, labels, idx_train, idx_val, idx_test = load_data(dataset, dataset_path='./data', seed=42, labelrate_train=20, labelrate_val=30)
    feats = g.ndata["feat"]

    args = argparse.Namespace(
        model_config_path="./train.conf.yaml",
        teacher=teacher,
        dataset=dataset,
        feat_dim=feats.shape[1],
        label_dim=labels.int().max().item() + 1,
        output_path="outputs"
    )
    conf = get_training_config(args.model_config_path, args.teacher, args.dataset)
    conf.update(vars(args))
    conf["device"] = device

    model = Model(conf)
    if compile_model:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    model.to(device)

    # Register hooks
    hooks = register_hooks(model)

    # === New: Register input sparsity hook ===
    input_sparsities = []

    def input_sparsity_hook(module, input, output):
        if len(input) > 0 and isinstance(input[0], torch.Tensor):
            sparsity = compute_input_sparsity(input[0])
            input_sparsities.append(sparsity)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module) and len(list(module.children())) == 0:
            module.register_forward_hook(input_sparsity_hook)

    export_layerwise_sparsity(model, f"sparsity_before_{dataset}_{name}.csv")

    optimizer = torch.optim.Adam(model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"])
    criterion = torch.nn.NLLLoss()
    evaluator = get_evaluator(dataset)

    print(f"=== {name.upper()} MODE TRAINING ===")
    start_train = time.time()
    indices = (idx_train, idx_val, idx_test)
    out, _, _, h_list, dist, codebook = run_transductive(
        conf, model, g, feats, labels, indices,
        criterion, evaluator, optimizer, None, []
    )
    train_time = time.time() - start_train
    print(f"{name.capitalize()} mode training time: {train_time:.2f} seconds")

    export_layerwise_sparsity(model, f"sparsity_after_{dataset}_{name}.csv")
    report_forward_times(model, f"forward_times_{dataset}_{name}.csv")

    cpu_mem = get_cpu_memory_mb()
    gpu_mem = get_gpu_memory_mb()

    # === New: Report average input sparsity ===
    if len(input_sparsities) > 0:
        avg_input_sparsity = sum(input_sparsities) / len(input_sparsities)
        print(f"Average input sparsity during training: {avg_input_sparsity:.2f}%")
    else:
        avg_input_sparsity = None
        print("No input sparsity recorded.")

    for h in hooks:
        h.remove()

    return {
        'dataset': dataset,
        'mode': name,
        'train_time_sec': train_time,
        'cpu_mem_mb': cpu_mem,
        'gpu_mem_mb': gpu_mem,
        'avg_input_sparsity': avg_input_sparsity,
    }


def main():
    results = []
    datasets = ["citeseer", "cora", "pubmed"]
    teachers = ["GCN", "SAGE"]  # adjust if needed

    for dataset in datasets:
        for teacher in teachers:
            results.append(run_experiment(name="eager", dataset=dataset, teacher=teacher, compile_model=False))
            results.append(run_experiment(name="compiled", dataset=dataset, teacher=teacher, compile_model=True))

    df = pd.DataFrame(results)
    df.to_csv("summary_runtime_memory_vqgraph.csv", index=False)
    print("\n=== Summary ===")
    print(df)


if __name__ == "__main__":
    main()
