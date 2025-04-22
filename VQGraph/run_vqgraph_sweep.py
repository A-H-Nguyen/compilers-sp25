import subprocess
import argparse
from itertools import product
import os
from pathlib import Path

def run_phase(phase, mode, dataset, teacher, seed, device):
    output_path = f"outputs_{mode}"
    Path("log").mkdir(parents=True, exist_ok=True)
    log_file = f"log/{phase}_{dataset}_{teacher}_seed{seed}_{mode}.log"

    if phase == "teacher":
        script = "train_teacher.py"
        base_cmd = [
            "python3", script,
            "--exp_setting", "tran",
            "--teacher", teacher,
            "--dataset", dataset,
            "--output_path", output_path,
            "--seed", str(seed),
            "--max_epoch", "500",
            "--patience", "1000",
            "--device", str(device)
        ]
    elif phase == "student":
        script = "train_student.py"
        base_cmd = [
            "python3", script,
            "--exp_setting", "tran",
            "--teacher", teacher,
            "--dataset", dataset,
            "--output_path", output_path,
            "--out_t_path", output_path,  # <- this is what student needs!
            "--seed", str(seed),
            "--device", str(device),
            "--max_epoch", "500",
            "--patience", "1000"
        ]

    else:
        raise ValueError(f"Invalid phase: {phase}")

    if mode == "compiled":
        base_cmd.append("--use_compile")

    print(f"[{phase.upper()} | {mode.upper()}] Dataset: {dataset}, Teacher: {teacher}, Seed: {seed}")
    with open(log_file, "w") as f:
        subprocess.run(
            base_cmd,
            env=dict(os.environ, TORCH_LOGS="recompiles"),
            stdout=f,
            stderr=subprocess.STDOUT
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["citeseer"])
    parser.add_argument("--teachers", nargs="+", default=["GCN"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    for dataset, teacher, seed in product(args.datasets, args.teachers, args.seeds):
        run_phase("teacher", "eager", dataset, teacher, seed, args.device)
        run_phase("teacher", "compiled", dataset, teacher, seed, args.device)
        #run_phase("student", "eager", dataset, teacher, seed, args.device)
        #run_phase("student", "compiled", dataset, teacher, seed, args.device)

if __name__ == "__main__":
    main()
