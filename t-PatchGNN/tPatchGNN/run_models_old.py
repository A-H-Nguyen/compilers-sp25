import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim
from perf_utils import (
    register_forward_hooks,
    report_forward_times,
    export_layerwise_sparsity,
    get_cpu_memory_mb,
    reset_gpu_memory_stats,
    get_gpu_peak_memory_mb,
    Timer
)

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import *

parser = argparse.ArgumentParser('IMTS Forecasting')
parser.add_argument("--compile",       action="store_true",
                    help="Wrap model in torch.compile()")
parser.add_argument("--layerwise_csv", action="store_true",
                    help="Dump per-layer timing & sparsity CSVs")

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")

# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################

if __name__ == '__main__':
	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, patch_ts=True)
	input_dim = data_obj["input_dim"]
	
	### Model setting ###
	args.ndim = input_dim
	model = tPatchGNN(args).to(args.device)

	# ── A) per-layer hooks for timing & (later) input sparsity ──
	name = "compiled" if args.compile else "eager"
	hooks = register_forward_hooks(model)
	input_sparsities = []
	def input_sparsity_hook(module, inp, out):
		x = inp[0]
		if isinstance(x, torch.Tensor):
			input_sparsities.append((x==0).float().mean().item()*100)
	# attach to all leaf modules
	for m in model.modules():
		if len(list(m.children()))==0:
			m.register_forward_hook(input_sparsity_hook)

	# ── B) optionally compile both forward and forecasting ──
	if args.compile:
		#print("⟳ torch.compile(model)…")
		model = torch.compile(model)
		# also compile forecasting so compute_all_losses uses compiled
		model.forecasting = torch.compile(model.forecasting)

	if args.layerwise_csv:
		export_layerwise_sparsity(model, f"sparsity_before_{name}.csv")

	##################################################################
	
	# # Load checkpoint and evaluate the model
	# if args.load is not None:
	# 	utils.get_ckpt_model(ckpt_path, model, args.device)
	# 	exit()

	##################################################################

	if(args.n < 12000):
		args.state = "debug"
		log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
	else:
		log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log". \
			format(args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr)
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	best_val_mse = np.inf
	test_res = None
	print(f"=== {name.upper()} MODE TRAINING ===")
	total_train_time = 0.0
	for itr in range(args.epoch):
		st = time.time()

		### Training ###
		model.train()
		for _ in range(num_batches):
			optimizer.zero_grad()
			batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
			train_res = compute_all_losses(model, batch_dict)
			train_res["loss"].backward()
			optimizer.step()
		total_train_time += (time.time() - st)
		### Validation ###
		model.eval()
		with torch.no_grad():
			val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			
			### Testing ###
			if(val_res["mse"] < best_val_mse):
				best_val_mse = val_res["mse"]
				best_iter = itr
				test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
			
			#logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
			#logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
			#logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
			#	.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
			#if(test_res != None):
			#	logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
			#		.format(best_iter, test_res["loss"], test_res["mse"],\
			# 		 test_res["rmse"], test_res["mae"], test_res["mape"]*100))
			#logger.info("Time spent: {:.2f}s".format(time.time()-st))

		if(itr - best_iter >= args.patience):
			print("Exp has been early stopped!")
			break

	print(f"{name.capitalize()} mode training time: {total_train_time:.2f}s")
	#logger.info(f"Total training time: {total_train_time:.2f}s")
	if args.layerwise_csv:
		# per-layer parameter sparsity after training
		export_layerwise_sparsity(model, f"sparsity_after_{name}.csv")
		# per-layer forward times
		report_forward_times(model,     f"forward_times_{name}.csv")

	print(f"=== {name.upper()} MODE EVALUATION ===")
	
	model.eval()
	with torch.no_grad():
		eval_start = time.time()
		#val_res  = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
		test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
		eval_time = time.time() - eval_start
		print(f"{name.capitalize()} mode evaluation time: {eval_time:.2f}s")
		#logger.info(f"Total evaluation time: {eval_time:.2f}s")
		#print(f"Final Val MSE: {val_res['mse']:.5f}, Test MSE: {test_res['mse']:.5f}")

	try:
		avg_input = sum(input_sparsities) / len(input_sparsities)
		print(f"Average input sparsity: {avg_input:.2f}%")
 		#logger.info(f"Average input sparsity: {avg_input:.2f}%")
	except NameError:
		pass
