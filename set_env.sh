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

# 5. Install PyTorch
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install DGL (with CUDA backend)
!pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

# 7. Clone and install TxGNN
git clone https://github.com/mims-harvard/TxGNN.git
cd TxGNN
pip install -r requirements.txt
python setup.py install
cd ..
bash update_txgnn_dataloader.sh

# 8. Clone and setup VQGraph
git clone https://github.com/YangLing0818/VQGraph.git
cd VQGraph
pip install -r requirements.txt
cd ..

# 9. Install CUDA libraries (optional but safe)
conda install --yes cuda -c nvidia
pip install pandas==1.3.0

# 10. Move custom scripts
cp ./custom_scripts/txgnn/test_txgnn.py ./TxGNN/test_txgnn.py
cp ./custom_scripts/txgnn/utils.py ./TxGNN/utils.py
cp ./custom_scripts/vqgraph/test_vqgraph.py ./VQGraph/test_vqgraph.py
cp ./custom_scripts/vqgraph/benchmark_utils.py ./VQGraph/benchmark_utils.py

sed -i 's/from google_drive_downloader import GoogleDriveDownloader as gdd/import googledrivedownloader as gdd/' VQGraph/dataloader.py

echo "✅ Environment setup complete."
echo "✅ Activate environment anytime with: source ./miniconda3/bin/activate && conda activate gnn_benchmark_env"

