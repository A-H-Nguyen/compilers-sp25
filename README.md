# compilers-sp25
Final project for CS 380C: Advanced Topics in Compilers

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Workloads

### With idev:
```bash
TORCH_LOGS="" python3 -u ${WORK}/EGNO/compiled_mocap.py --config_by_file ${WORK}/compilers-sp25/config_mocap_no.json --outf ${WORK}/slurm-out/egno-mocap-out/compiled 2>&1 | tee ${WORK}/slurm-out/egno-mocap-out/compiled/idev/"torch_logs_$(date +'%Y-%m-%d_%H-%M-%S').txt"
```