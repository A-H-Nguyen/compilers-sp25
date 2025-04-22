import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data, load_out_t
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    check_readable,
    compute_min_cut_loss,
    graph_split,
    feature_prop,
)
from train_and_eval import distill_run_transductive, distill_run_inductive
from benchmark_utils import (
    layerwise_sparsity,
    save_sparsity_report,
    register_hooks,
    report_forward_times,
    log_memory_usage,
)
import time


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--teacher", type=str, default="SAGE")
    parser.add_argument("--student", type=str, default="MLP")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--out_t_path", type=str, default="outputs")
    parser.add_argument("--exp_setting", type=str, default="tran")
    parser.add_argument("--use_compile", action="store_true")
    # Add all other args as needed...
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--model_config_path", type=str, default="./train.conf.yaml")



    return parser.parse_args()


def run(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    # Output directories
    out_dir = Path(args.output_path) / args.exp_setting / args.dataset / f"{args.teacher}_{args.student}"
    out_t_dir = Path(args.out_t_path) / args.exp_setting / args.dataset / args.teacher / f"seed_{args.seed}"
    check_writable(out_dir, overwrite=False)
    check_readable(out_t_dir)

    logger = get_logger(out_dir / "log", True, 20)
    logger.info(f"Output dir: {out_dir}")

    # Load data
    g, labels, idx_train, idx_val, idx_test = load_data(args.dataset, args.data_path, seed=args.seed, labelrate_train=20, labelrate_val=30)
    feats = g.ndata["feat"]

    args.feat_dim = feats.shape[1]
    args.label_dim = labels.int().max().item() + 1
    conf = get_training_config(args.model_config_path, args.student, args.dataset)
    conf.update(vars(args))
    conf["device"] = device

    # Ensure all needed keys exist in conf
    defaults = {
        "norm_type": "none",
        "dropout_ratio": 0.0,
        "hidden_dim": 64,
        "num_layers": 2,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "model_config_path": "./train.conf.yaml"
    }
    for key, value in defaults.items():
        if key not in conf:
            conf[key] = getattr(args, key, value)


    model = Model(conf)
    if args.use_compile:
        print("Compiling student model with torch.compile()...")
        model = torch.compile(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"])
    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(args.dataset)

    # Load teacher outputs
    out_t = load_out_t(out_t_dir, 'tea_soft_labels.npz')
    codebook = load_out_t(out_t_dir, 'codebook_embeddings.npz')
    tokens = load_out_t(out_t_dir, 'tea_soft_token_assignments.npz')

    # Log pre-training sparsity
    save_sparsity_report(model, out_dir / "sparsity_before.csv")

    # Register timing hooks
    hooks = register_hooks(model)

    # Training
    start_time = time.time()

    if args.exp_setting == "tran":
        distill_indices = (idx_train, torch.cat([idx_train, idx_val, idx_test]), idx_val, idx_test)
        out, _ = distill_run_transductive(conf, model, feats, labels, out_t, codebook, tokens, distill_indices, criterion_l, criterion_t, evaluator, optimizer, logger, [])
    else:
        raise NotImplementedError("Inductive not handled in this minimal version")

    train_duration = time.time() - start_time
    logger.info(f"Total training time: {train_duration:.2f}s")

    # Report memory and layer timings
    log_memory_usage(logger)
    report_forward_times(model, out_dir / "forward_times.csv")
    save_sparsity_report(model, out_dir / "sparsity_after.csv")

    for hook in hooks:
        hook.remove()

    # Save prediction
    np.savez(out_dir / "out", out.detach().cpu().numpy())


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
