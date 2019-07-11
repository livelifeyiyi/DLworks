
from __future__ import absolute_import, division, print_function
import time
# from pynvml import *
import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features_ner, compute_metrics

if sys.version_info[0] == 2:
	import cPickle as pickle
else:
	import pickle

# nvmlInit()
logger = logging.getLogger(__name__)
print(time.time())


def main():
	parser = argparse.ArgumentParser()

	# # Required parameters
	parser.add_argument("--data_dir",
						default='./msra_ner',
						type=str,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	parser.add_argument("--bert_model", default='../bert-base-chinese/', type=str,
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
							 "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
							 "bert-base-multilingual-cased, bert-base-chinese.")
	parser.add_argument("--task_name",
						default='ner',
						type=str,
						help="The name of the task to train.")
	parser.add_argument("--output_dir",
						default='./out/',
						type=str,
						help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument('--cuda_device', default=None, type=str, help='e.g. 0')

	## Other parameters
	parser.add_argument("--cache_dir",
						default="",
						type=str,
						help="Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--max_seq_length",
						default=128,
						type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
							 "Sequences longer than this will be truncated, and sequences shorter \n"
							 "than this will be padded.")
	parser.add_argument("--do_train",
						default=True,
						action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval",
						action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--do_lower_case",
						action='store_true',
						help="Set this flag if you are using an uncased model.")
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
						default=5e-5,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
						default=3.0,
						type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",
						default=0.1,
						type=float,
						help="Proportion of training to perform linear learning rate warmup for. "
							 "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
						default=True,
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument('--overwrite_output_dir',
						action='store_true',
						help="Overwrite the content of the output directory")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16',
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
						type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
	parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
	args = parser.parse_args()

	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd
		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()
	if args.cuda_device:
		device = torch.device("cuda:%s" % args.cuda_device)
		n_gpu = 1
	else:
		if args.local_rank == -1 or args.no_cuda:
			device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
			n_gpu = torch.cuda.device_count()
		else:
			torch.cuda.set_device(args.local_rank)
			device = torch.device("cuda", args.local_rank)
			n_gpu = 1
			# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
			torch.distributed.init_process_group(backend='nccl')
	args.device = device

	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
		device, n_gpu, bool(args.local_rank != -1), args.fp16))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
			args.gradient_accumulation_steps))

	args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	if os.path.exists(args.output_dir) and os.listdir(
			args.output_dir) and args.do_train and not args.overwrite_output_dir:
		raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
	if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(args.output_dir)

	task_name = args.task_name.lower()

	if task_name not in processors:
		raise ValueError("Task not found: %s" % (task_name))

	processor = processors[task_name]()  # nerProcessor
	# output_mode = output_modes[task_name]

	label_list = processor.get_labels()
	num_labels = len(label_list) + 1

	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	stime = time.time()
	tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	model = BertForTokenClassification.from_pretrained(args.bert_model, num_labels=num_labels)

	etime = time.time()
	logger.info("Model loaded in %s seconds" % (etime-stime))

	model.to(device)
	# gpu_handle = nvmlDeviceGetHandleByIndex(int(args.cuda_device))
	# gpu_info = nvmlDeviceGetMemoryInfo(gpu_handle)
	# logger.info("GPU usage:%s" % (gpu_info.used / 2 ** 30))

	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model,
														  device_ids=[args.local_rank],
														  output_device=args.local_rank,
														  find_unused_parameters=True)
	elif n_gpu > 1:
		model = torch.nn.DataParallel(model)

	global_step = 0
	nb_tr_steps = 0
	tr_loss = 0

	if args.do_train:

		# Prepare data loader
		train_examples = processor.get_train_examples(args.data_dir)
		cached_train_features_file = os.path.join(args.data_dir, 'train_{0}_{1}_{2}'.format(
			list(filter(None, args.bert_model.split('/'))).pop(),
			str(args.max_seq_length),
			str(task_name)))
		try:
			with open(cached_train_features_file, "rb") as reader:
				train_features = pickle.load(reader)
		except:
			train_features = convert_examples_to_features_ner(
				train_examples, label_list, args.max_seq_length, tokenizer)
			# if args.local_rank == -1 or torch.distributed.get_rank() == 0:
			# 	logger.info("  Saving train features into cached file %s", cached_train_features_file)
			# 	with open(cached_train_features_file, "wb") as writer:
			# 		pickle.dump(train_features, writer)

		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
		all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
		all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

		train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
		if args.local_rank == -1:
			train_sampler = RandomSampler(train_data)
		else:
			train_sampler = DistributedSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
		num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

		param_optimizer = list(model.named_parameters())
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		if args.fp16:
			try:
				from apex.optimizers import FP16_Optimizer
				from apex.optimizers import FusedAdam
			except ImportError:
				raise ImportError(
					"Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

			optimizer = FusedAdam(optimizer_grouped_parameters,
								  lr=args.learning_rate,
								  bias_correction=False,
								  max_grad_norm=1.0)
			if args.loss_scale == 0:
				optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
			else:
				optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
			warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
												 t_total=num_train_optimization_steps)

		else:
			optimizer = BertAdam(optimizer_grouped_parameters,
								 lr=args.learning_rate,
								 warmup=args.warmup_proportion,
								 t_total=num_train_optimization_steps)

		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", len(train_examples))
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num steps = %d", num_train_optimization_steps)

		model.train()
		# gpu_handle = nvmlDeviceGetHandleByIndex(int(args.cuda_device))
		# gpu_info = nvmlDeviceGetMemoryInfo(gpu_handle)
		# logger.info("Model start training, GPU usage:%s" % (gpu_info.used / 2 ** 30))

		for _ in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
			epoch_stime = time.time()
			tr_loss = 0
			nb_tr_examples, nb_tr_steps = 0, 0
			for step, batch in enumerate(
					tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, label_ids = batch

				# define a new function to compute loss values for both output_modes
				loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
					  labels=label_ids)

				# loss_fct = CrossEntropyLoss()
				# loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

				if n_gpu > 1:
					loss = loss.mean()  # mean() to average on multi-gpu.
				if args.gradient_accumulation_steps > 1:
					loss = loss / args.gradient_accumulation_steps

				if args.fp16:
					optimizer.backward(loss)
				else:
					loss.backward()

				tr_loss += loss.item()
				nb_tr_examples += input_ids.size(0)
				nb_tr_steps += 1
				if (step + 1) % args.gradient_accumulation_steps == 0:
					if args.fp16:
						# modify learning rate with special warm up BERT uses
						# if args.fp16 is False, BertAdam is used that handles this automatically
						lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
						for param_group in optimizer.param_groups:
							param_group['lr'] = lr_this_step
					optimizer.step()
					optimizer.zero_grad()
					global_step += 1
					# if args.local_rank in [-1, 0]:
					# 	tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
					# 	tb_writer.add_scalar('loss', loss.item(), global_step)
			# epoch_etime = time.time()
			# gpu_handle = nvmlDeviceGetHandleByIndex(int(args.cuda_device))
			# gpu_info = nvmlDeviceGetMemoryInfo(gpu_handle)
			# logger.info(
			# 	"Training epoch cost %s seconds, GPU usage:%s" % (epoch_etime - epoch_stime, gpu_info.used / 2 ** 30))
		# logits = model(input_ids=all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask,
		# 			   labels=all_label_ids)
		pass

main()