import subprocess
import os
from pathlib import Path

def run_script(script_path, args, log_file):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: python {script_path} {' '.join(args)}")
    with open(log_file, "w") as f:
        subprocess.run(
            ["python", script_path] + args,
            env=dict(os.environ, TORCH_LOGS="recompiles"),
            stdout=f,
            stderr=subprocess.STDOUT
        )

def main():
    # Eager and Compiled modes
    modes = ["eager", "compiled"]

    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/TxGNN").mkdir(parents=True, exist_ok=True)
    Path("output/VQGraph").mkdir(parents=True, exist_ok=True)

    os.chdir("TxGNN")
    
    # TxGNN Benchmark
    for mode in modes:
        script = "./test_txgnn.py"
        args = []
        if mode == "compiled":
            args.append("--use_compile")

        log_file = f"../log/txgnn/txgnn_{mode}.log"
        print(f"\n=== Running TxGNN [{mode.upper()}] ===")
        run_script(script, args, log_file)

    os.chdir("..")

    # VQGraph Benchmark
    vq_datasets = ["citeseer", "cora", "pubmed"]
    teachers = ["GCN", "SAGE"]
    os.chdir("VQGraph")

    for dataset in vq_datasets:
        for teacher in teachers:
            for mode in modes:
                script = "./test_vqgraph.py"
                args = [
                    "--dataset", dataset,
                    "--teacher", teacher
                ]
                if mode == "compiled":
                    args.append("--use_compile")

                log_file = f"../log/vqgraph/vqgraph_{dataset}_{teacher}_{mode}.log"
                print(f"\n=== Running VQGraph [{dataset.upper()} | {teacher} | {mode.upper()}] ===")
                run_script(script, args, log_file)
    os.chdir("..")

    # tPatchGNN benchmark
    os.chdir("t-PatchGNN")
    script = "./tPatchGNN/run_models.py"
    tpatch_datasets = ["activity", "physionet", "ushcn"]
    for dataset in tpatch_datasets:
        args = [
            "--dataset", dataset,
            "--layerwise_csv",
            "--gpu", "0"
        ]
        log_file = f"../log/tpatchgnn/tpatchgnn_{dataset}.log"
        print(f"\n=== Running tPatchGNN [{dataset.upper()} | Compiled + Eager ] ===")
        run_script(script, args, log_file)

if __name__ == "__main__":
    main()
