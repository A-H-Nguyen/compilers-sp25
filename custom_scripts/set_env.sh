#!/bin/bash

set -e  # Exit immediately if any command fails

# 1. Install Miniconda locally
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm ./miniconda3/miniconda.sh

# 2. Setup conda environment
source ./miniconda3/bin/activate
conda init bash
source ~/.bashrc  # reload bashrc

# 3. Create new environment
conda create --yes --name gnn_benchmark_env python=3.8

# 4. Activate environment
conda activate gnn_benchmark_env

# 5. Install DGL (with CUDA backend)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

# 6. Clone and install TxGNN
rm -rf TxGNN
git clone https://github.com/mims-harvard/TxGNN.git
cp ./custom_scripts/txgnn/test_txgnn.py ./TxGNN/test_txgnn.py
cp ./custom_scripts/txgnn/utils.py ./TxGNN/utils.py
cd TxGNN
pip install -r requirements.txt
pip install .
#python setup.py install
cd ..
bash ./custom_scripts/update_txgnn_dataloader.sh

# 7. Clone and setup VQGraph
rm -rf VQGraph
git clone https://github.com/YangLing0818/VQGraph.git
cp ./custom_scripts/vqgraph/test_vqgraph.py ./VQGraph/test_vqgraph.py
cp ./custom_scripts/vqgraph/benchmark_utils.py ./VQGraph/benchmark_utils.py
cd VQGraph
pip install -r requirements.txt
cd ..

# 9. Clone and setup tPatchGNN
rm -rf t-PatchGNN
git clone https://github.com/usail-hkust/t-PatchGNN.git
cp ./custom_scripts/tPatchGNN/run_models.py ./t-PatchGNN/tPatchGNN/run_models.py
cp ./custom_scripts/tPatchGNN/perf_utils.py ./t-PatchGNN/tPatchGNN/perf_utils.py
cd t-PatchGNN
pip install -r requirements.txt
rm tPatchGNN/run_models.py
cd ..

# 8. Install CUDA libraries
conda install --yes cuda -c nvidia
#pip install pandas==1.3.0
conda install pandas=1.3.0 -c conda-forge

# 9. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

sed -i 's/from google_drive_downloader import GoogleDriveDownloader as gdd/import googledrivedownloader as gdd/' VQGraph/dataloader.py

echo "✅ Environment setup complete."
echo "✅ Activate environment anytime with: source ./miniconda3/bin/activate && conda activate gnn_benchmark_env"

