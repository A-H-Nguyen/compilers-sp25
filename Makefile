# Variables
ENV_NAME = gnn_benchmark_env
CONDA_PREFIX = ./miniconda3
ACTIVATE_CONDA = source $(CONDA_PREFIX)/bin/activate

# Targets
.PHONY: setup_env activate_env install_txgnn install_vqgraph benchmark_all clean

# 1. Install Miniconda + Conda Environment
setup_env:
	bash ./custom_scripts/set_env.sh

# 2. Activate Conda Env manually (prints instructions)
activate_env:
	@echo "Run the following to activate your environment:"
	@echo "  source $(CONDA_PREFIX)/bin/activate && conda activate $(ENV_NAME)"

# 3. Install TxGNN separately (optional)
install_txgnn:
	git clone https://github.com/mims-harvard/TxGNN.git
	cp ./custom_scripts/txgnn/test_txgnn.py ./TxGNN/test_txgnn.py
	cp ./custom_scripts/txgnn/utils.py ./TxGNN/utils.py
	#cd TxGNN && pip install -r requirements.txt && python setup.py install
	cd TxGNN && pip install -r requirements.txt && pip install .

# 4. Install VQGraph separately (optional)
install_vqgraph:
	git clone https://github.com/YangLing0818/VQGraph.git
	sed -i 's/from google_drive_downloader import GoogleDriveDownloader as gdd/import googledrivedownloader as gdd/' VQGraph/dataloader.py
	cp ./custom_scripts/vqgraph/test_vqgraph.py ./VQGraph/test_vqgraph.py
	cp ./custom_scripts/vqgraph/benchmark_utils.py ./VQGraph/benchmark_utils.py
	cd VQGraph && pip install -r requirements.txt

# 5. Run all benchmark sweeps (your run_sweep.py assumed)
benchmark_all:
	$(ACTIVATE_CONDA) && conda activate $(ENV_NAME) && \
	python ./custom_scripts/run_sweep.py

# 6. Clean conda env + folders
clean:
	rm -rf $(CONDA_PREFIX)
	rm -rf TxGNN VQGraph
	conda remove --name $(ENV_NAME) --all --yes || true
