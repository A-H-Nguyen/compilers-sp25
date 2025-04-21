import subprocess
import argparse
from itertools import product
import os

def run_vqgraph(mode, dataset, teacher, seed, device):
    output_path = f"outputs_{mode}"
    log_file = f"{output_path}_{dataset}_{teacher}_seed{seed}_{mode}.log"

    base_cmd = [
        "python3", "train_teacher.py",
        "--exp_setting", "tran",
        "--teacher", teacher,
        "--dataset", dataset,
        "--output_path", output_path,
        "--seed", str(seed),
        "--max_epoch", "500",
        "--patience", "1000",
        "--device", str(device)
    ]
    if mode == "compiled":
        base_cmd.append("--use_compile")

    print(f"[{mode.upper()}] Dataset: {dataset}, Teacher: {teacher}, Seed: {seed}")
    with open(log_file, "w") as f:
        subprocess.run(base_cmd, env=dict(os.environ, TORCH_LOGS="recompiles"), stdout=f, stderr=subprocess.STDOUT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["citeseer"])
    parser.add_argument("--teachers", nargs="+", default=["GCN"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    for dataset, teacher, seed in product(args.datasets, args.teachers, args.seeds):
        run_vqgraph("eager", dataset, teacher, seed, args.device)
        run_vqgraph("compiled", dataset, teacher, seed, args.device)

if __name__ == "__main__":
    main()
