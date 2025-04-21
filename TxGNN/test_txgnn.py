from txgnn import TxData, TxGNN, TxEval
from utils import (
    register_hooks, report_forward_times,
    export_layerwise_sparsity, get_cpu_memory_mb,
    get_gpu_memory_mb
)
import time
import torch

def run_experiment(name, compile_model=False):
    # Load data
    TxDataObj = TxData(data_folder_path='./data')
    TxDataObj.prepare_split(split='complex_disease', seed=42)

    TxGNNObj = TxGNN(data=TxDataObj,
                     weight_bias_track=False,
                     proj_name='TxGNN',
                     exp_name='TxGNN_' + name,
                     device='cuda:0')

    # Initialize model
    TxGNNObj.model_initialize(
        n_hid=100, n_inp=100, n_out=100,
        proto=True, proto_num=3,
        attention=False,
        sim_measure='all_nodes_profile',
        agg_measure='rarity',
        num_walks=200, path_length=2
    )

    if compile_model:
        TxGNNObj.model = torch.compile(TxGNNObj.model)

    # Register hooks
    hooks = register_hooks(TxGNNObj.model)

    # Sparsity before training
    export_layerwise_sparsity(TxGNNObj.model, f"sparsity_before_{name}.csv")

    print(f"=== {name.upper()} MODE TRAINING ===")
    start_train = time.time()
    TxGNNObj.pretrain(n_epoch=2, learning_rate=1e-3, batch_size=1024, train_print_per_n=20)
    TxGNNObj.finetune(n_epoch=500, learning_rate=5e-4, train_print_per_n=5, valid_per_n=20,
                      save_name=f'finetune_{name}.pt')
    train_time = time.time() - start_train
    print(f"{name.capitalize()} mode training time: {train_time:.2f} seconds")

    # Sparsity after training
    export_layerwise_sparsity(TxGNNObj.model, f"sparsity_after_{name}.csv")

    # Report per-layer timing
    report_forward_times(TxGNNObj.model, f"forward_times_{name}.csv")

    # Evaluate
    print(f"=== {name.upper()} MODE EVALUATION ===")
    start_eval = time.time()
    TxEvalObj = TxEval(model=TxGNNObj)
    result = TxEvalObj.eval_disease_centric(
        disease_idxs='test_set',
        show_plot=False,
        verbose=True,
        save_result=True,
        return_raw=False,
        save_name=f"evaluation_results_{name}.pt"
    )
    eval_time = time.time() - start_eval
    print(f"{name.capitalize()} mode evaluation time: {eval_time:.2f} seconds")
    print(result)

    # Log memory
    cpu_mem = get_cpu_memory_mb()
    gpu_mem = get_gpu_memory_mb()

    # Clean up hooks
    for h in hooks:
        h.remove()

    return {
        'mode': name,
        'train_time_sec': train_time,
        'eval_time_sec': eval_time,
        'cpu_mem_mb': cpu_mem,
        'gpu_mem_mb': gpu_mem
    }


if __name__ == "__main__":
    import pandas as pd

    results = []
    results.append(run_experiment(name="eager", compile_model=False))
    results.append(run_experiment(name="compiled", compile_model=True))

    df = pd.DataFrame(results)
    df.to_csv("summary_runtime_memory.csv", index=False)
    print("\n=== Summary ===")
    print(df)
