# Step 1: Add two import lines after line 14
sed -i '14a import dgl.distributed\nimport dgl.dataloading' TxGNN/txgnn/TxGNN.py

# Step 2: Insert 3-line sampler modification after line 135, with padding
sed -i '135a \\        sampler = dgl.dataloading.as_edge_prediction_sampler(\n            sampler,\n            negative_sampler=Minibatch_NegSampler(self.G, 1, '\''fix_dst'\'')\n        )\n' TxGNN/txgnn/TxGNN.py

# Step 3: Replace EdgeDataLoader with DataLoader on line 152
sed -i '151s/dgl.dataloading.EdgeDataLoader/dgl.dataloading.DataLoader/' TxGNN/txgnn/TxGNN.py

# Step 4: Comment out line 153 (negative_sampler=...)
sed -i '153s/^/            #/' TxGNN/txgnn/TxGNN.py
