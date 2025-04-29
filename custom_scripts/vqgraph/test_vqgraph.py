import subprocess
import os
from pathlib import Path
import pandas as pd
from itertools import product

def run_phase(phase, dataset, teacher, mode, log_dir="log_vqgraph"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{log_dir}/{phase}_{dataset}_{teacher}_{mode}.log"
    output_path = f"outputs_{mode}"

    if phase == "teacher":
        cmd = [
            "python3", "train_teacher.py",
            "--exp_setting", "tran",
            "--teacher", teacher,
            "--dataset", dataset,
            "--output_path", output_path,
            "--seed", "42",
            "--max_epoch", "500",
            "--patience", "1000",
            "--device", "0"
        ]
    elif phase == "student":
        cmd = [
            "python3", "train_student.py",
            "--exp_setting", "tran",
            "--teacher", teacher,
            "--dataset", dataset,
            "--output_path", output_path,
            "--out_t_path", output_path,
            "--seed", "42",
            "--device", "0",
            "--max_epoch", "500",
            "--patience", "1000"
        ]
    else:
        raise ValueError(f"Invalid phase: {phase}")

    if mode == "compiled":
        cmd.append("--use_compile")

    print(f"[{phase.upper()} | {mode.upper()}] Dataset: {dataset}, Teacher: {teacher}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, env=dict(os.environ, TORCH_LOGS="recompiles"), stdout=f, stderr=subprocess.STDOUT)

    return {
        "phase": phase,
        "dataset": dataset,
        "teacher": teacher,
        "mode": mode,
        "log_file": log_file
    }

def main():
    datasets = ["citeseer", "cora", "pubmed"]
    teachers = ["GCN", "SAGE"]
    modes = ["eager", "compiled"]
    results = []

    for dataset, teacher, mode in product(datasets, teachers, modes):
        results.append(run_phase("teacher", dataset, teacher, mode))
        results.append(run_phase("student", dataset, teacher, mode))

    df = pd.DataFrame(results)
    df.to_csv("summary_vqgraph_runs.csv", index=False)
    print("\n=== Summary ===")
    print(df)

if __name__ == "__main__":
    main()
