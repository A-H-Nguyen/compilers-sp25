import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import pandas as pd

from models import Model
from dataloader import load_data
from train_and_eval import run_transductive, run_inductive
# Imports from the original VQGraph repo
from utils import (
    get_logger, get_evaluator, set_seed, get_training_config, check_writable,
    compute_min_cut_loss, graph_split, feature_prop,
)

# Imports from benchmark_utils.py (ours)
from benchmark_utils import (
    register_hooks,
    export_layerwise_sparsity,
    report_forward_times,
    get_cpu_memory_mb,
    get_gpu_memory_mb,
)


import time


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_level", type=int, default=20)
    parser.add_argument("--console_log", action="store_true")
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--num_exp", type=int, default=1)
    parser.add_argument("--exp_setting", type=str, default="tran")
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_results", action="store_true")

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--labelrate_train", type=int, default=20)
    parser.add_argument("--labelrate_val", type=int, default=30)
    parser.add_argument("--split_idx", type=int, default=0)

    parser.add_argument("--codebook_size", type=int, default=5000)
    parser.add_argument("--lamb_node", type=float, default=0.001)
    parser.add_argument("--lamb_edge", type=float, default=0.03)

    parser.add_argument("--model_config_path", type=str, default="./train.conf.yaml")
    parser.add_argument("--teacher", type=str, default="SAGE")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout_ratio", type=float, default=0)
    parser.add_argument("--norm_type", type=str, default="none")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--fan_out", type=str, default="5,5")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--feature_noise", type=float, default=0)
    parser.add_argument("--split_rate", type=float, default=0.2)
    parser.add_argument("--compute_min_cut", action="store_true")
    parser.add_argument("--feature_aug_k", type=int, default=0)

    return parser.parse_args()


def run(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    start_train = time.time()

    output_dir = Path(args.output_path, args.exp_setting, args.dataset, args.teacher, f"seed_{args.seed}")
    args.output_dir = output_dir
    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)

    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset, args.data_path,
        split_idx=args.split_idx, seed=args.seed,
        labelrate_train=args.labelrate_train, labelrate_val=args.labelrate_val
    )

    feats = g.ndata["feat"]
    args.feat_dim = feats.shape[1]
    args.label_dim = labels.int().max().item() + 1

    if 0 < args.feature_noise <= 1:
        feats = (1 - args.feature_noise) * feats + args.feature_noise * torch.randn_like(feats)

    conf = get_training_config(args.model_config_path, args.teacher, args.dataset) if args.model_config_path else {}
    conf = dict(args.__dict__, **conf)
    conf["device"] = device

    model = Model(conf)
    if args.use_compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    hooks = register_hooks(model)
    export_layerwise_sparsity(model, output_dir / "sparsity_before.csv")

    optimizer = optim.Adam(model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"])
    criterion = torch.nn.NLLLoss()
    evaluator = get_evaluator(conf["dataset"])

    if args.exp_setting == "tran":
        indices = (idx_train, idx_val, idx_test)
        if args.feature_aug_k > 0:
            feats = feature_prop(feats, g, args.feature_aug_k)

        out, _, _, h_list, dist, codebook = run_transductive(
            conf, model, g, feats, labels, indices,
            criterion, evaluator, optimizer, logger, []
        )
    else:
        raise NotImplementedError("Only --exp_setting tran is supported in this version.")

    total_train_time = time.time() - start_train
    export_layerwise_sparsity(model, output_dir / "sparsity_after.csv")
    report_forward_times(model, output_dir / "forward_times.csv")

    cpu_mem = get_cpu_memory_mb()
    gpu_mem = get_gpu_memory_mb()

    pd.DataFrame([{
        "compiled": args.use_compile,
        "train_time_sec": total_train_time,
        "cpu_mem_mb": cpu_mem,
        "gpu_mem_mb": gpu_mem
    }]).to_csv(output_dir / "summary_runtime_memory.csv", index=False)

    for h in hooks:
        h.remove()


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
