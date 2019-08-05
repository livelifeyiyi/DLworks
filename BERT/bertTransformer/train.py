#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertConfig, optimization

import bertTransformer.distributed as distributed
# from bertTransformer.models import data_loader, model_builder
# from bertTransformer.models.data_loader import load_dataset
from bertTransformer.models.model_builder import Summarizer
from bertTransformer.models.trainer import build_trainer
from bertTransformer.others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def multi_main(args):
	""" Spawns 1 process per GPU """
	init_logger()

	nb_gpu = args.world_size
	mp = torch.multiprocessing.get_context('spawn')

	# Create a thread to listen for errors in the child processes.
	error_queue = mp.SimpleQueue()
	error_handler = ErrorHandler(error_queue)

	# Train with multiprocessing.
	procs = []
	for i in range(nb_gpu):
		device_id = i
		procs.append(mp.Process(target=run, args=(args,
												  device_id, error_queue,), daemon=True))
		procs[i].start()
		logger.info(" Starting process pid: %d  " % procs[i].pid)
		error_handler.add_child(procs[i].pid)
	for p in procs:
		p.join()


def run(args, device_id, error_queue):
	""" run process """
	setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

	try:
		gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
		print('gpu_rank %d' % gpu_rank)
		if gpu_rank != args.gpu_ranks[device_id]:
			raise AssertionError("An error occurred in \
                  Distributed initialization")

		train(args, device_id)
	except KeyboardInterrupt:
		pass  # killed by parent, do nothing
	except Exception:
		# propagate exception to parent process, keeping original traceback
		import traceback
		error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
	"""A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

	def __init__(self, error_queue):
		""" init error handler """
		import signal
		import threading
		self.error_queue = error_queue
		self.children_pids = []
		self.error_thread = threading.Thread(
			target=self.error_listener, daemon=True)
		self.error_thread.start()
		signal.signal(signal.SIGUSR1, self.signal_handler)

	def add_child(self, pid):
		""" error handler """
		self.children_pids.append(pid)

	def error_listener(self):
		""" error listener """
		(rank, original_trace) = self.error_queue.get()
		self.error_queue.put((rank, original_trace))
		os.kill(os.getpid(), signal.SIGUSR1)

	def signal_handler(self, signalnum, stackframe):
		""" signal handler """
		for pid in self.children_pids:
			os.kill(pid, signal.SIGINT)  # kill children processes
		(rank, original_trace) = self.error_queue.get()
		msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
		msg += original_trace
		raise Exception(msg)


def wait_and_validate(args, device_id):
	timestep = 0
	if (args.test_all):
		cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
		cp_files.sort(key=os.path.getmtime)
		xent_lst = []
		for i, cp in enumerate(cp_files):
			step = int(cp.split('.')[-2].split('_')[-1])
			xent = validate(args, device_id, cp, step)
			xent_lst.append((xent, cp))
			max_step = xent_lst.index(min(xent_lst))
			if (i - max_step > 10):
				break
		xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
		logger.info('PPL %s' % str(xent_lst))
		for xent, cp in xent_lst:
			step = int(cp.split('.')[-2].split('_')[-1])
			test(args, device_id, cp, step)
	else:
		while (True):
			cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
			cp_files.sort(key=os.path.getmtime)
			if cp_files:
				cp = cp_files[-1]
				time_of_cp = os.path.getmtime(cp)
				if (not os.path.getsize(cp) > 0):
					time.sleep(60)
					continue
				if (time_of_cp > timestep):
					timestep = time_of_cp
					step = int(cp.split('.')[-2].split('_')[-1])
					validate(args, device_id, cp, step)
					test(args, device_id, cp, step)

			cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
			cp_files.sort(key=os.path.getmtime)
			if (cp_files):
				cp = cp_files[-1]
				time_of_cp = os.path.getmtime(cp)
				if (time_of_cp > timestep):
					continue
			else:
				time.sleep(300)


def validate(args, device_id, pt, epoch):
	device = "cpu" if args.visible_gpus == '-1' else "cuda"
	if (pt != ''):
		test_from = pt
	else:
		test_from = args.test_from
	logger.info('Loading checkpoint from %s' % test_from)
	checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
	opt = vars(checkpoint['opt'])
	for k in opt.keys():
		if (k in model_flags):
			setattr(args, k, opt[k])
	print(args)

	config = BertConfig.from_json_file(args.bert_config_name)
	model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
	model.load_cp(checkpoint)
	model.eval()
	valid_dataset = torch.load(args.bert_data_path + 'valid.data')

	trainer = build_trainer(args, device_id, model, None)
	stats = trainer.validate(valid_dataset, epoch)
	return stats.xent()


def test(args, device_id, pt):
	device = "cpu" if args.visible_gpus == '-1' else "cuda"
	if pt != '':
		test_from = pt
	else:
		test_from = args.best_model
	logger.info('Loading checkpoint from %s' % test_from)
	checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
	opt = vars(checkpoint['opt'])
	for k in opt.keys():
		if k in model_flags:
			setattr(args, k, opt[k])
	# print(args)

	config = BertConfig.from_json_file(args.bert_config_path)
	model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
	model.load_cp(checkpoint)
	model.eval()

	logger.info("Test dataset......")
	test_dataset = torch.load(args.bert_data_path + 'test.data')
	trainer = build_trainer(args, device_id, model, None)
	trainer.test(model, test_dataset, device)

	logger.info("Valid dataset......")
	test_dataset = torch.load(args.bert_data_path + 'valid.data')
	trainer = build_trainer(args, device_id, model, None)
	trainer.test(model, test_dataset, device)


# def baseline(args, cal_lead=False, cal_oracle=False):
# 	test_dataset = torch.load(args.bert_data_path + 'test.data')
#
# 	trainer = build_trainer(args, device_id, None, None)
# 	#
# 	if (cal_lead):
# 		trainer.test(model, test_dataset, device)
# 	elif (cal_oracle):
# 		trainer.test(model, test_dataset, device)


def train(args, device_id):
	init_logger(args.log_file)

	device = "cpu" if args.visible_gpus == '-1' else "cuda"
	logger.info('Device ID %d' % device_id)
	logger.info('Device %s' % device)
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = True

	if device_id >= 0:
		# torch.cuda.set_device(device_id)
		torch.cuda.manual_seed(args.seed)

	torch.manual_seed(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = True

	# def train_iter_fct():
	# 	return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
	# 								  shuffle=True, is_test=False)

	train_dataset = torch.load(args.bert_data_path + 'train_ht.data')
	if args.do_use_second_dataset:
		train_dataset += torch.load(args.second_dataset_path + 'train.data')
	logger.info('Loading training dataset from %s, number of examples: %d' %
				(args.bert_data_path, len(train_dataset)))
	train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
	model = Summarizer(args, device, load_pretrained_bert=True)
	# if args.train_from != '':
	# 	logger.info('Loading checkpoint from %s' % args.train_from)
	# 	checkpoint = torch.load(args.train_from,
	# 							map_location=lambda storage, loc: storage)
	# 	opt = vars(checkpoint['opt'])
	# 	for k in opt.keys():
	# 		if (k in model_flags):
	# 			setattr(args, k, opt[k])
	# 	model.load_cp(checkpoint)
	# 	optim = model_builder.build_optim(args, model, checkpoint)
	# else:
	# 	optim = model_builder.build_optim(args, model, None)
	_params = filter(lambda p: p.requires_grad, model.parameters())
	optim = optimization.BertAdam(_params, lr=args.lr, weight_decay=args.l2reg)

	logger.info(model)
	trainer = build_trainer(args, device_id, model, optim)
	trainer.train(train_dataloader, device)

	if args.do_test:
		model = trainer.model
		model.eval()
		test_dataset = torch.load(args.bert_data_path + 'valid.data')
		logger.info('Loading valid dataset from %s, number of examples: %d' %
					(args.bert_data_path, len(test_dataset)))
		trainer = build_trainer(args, device_id, model, None)
		trainer.test(model, test_dataset, device)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--encoder", default='transformer', type=str,
						choices=['classifier', 'transformer', 'rnn', 'baseline'])
	parser.add_argument("--model_name", default="avg", choices=['avg', 'pool'])
	parser.add_argument("--mode", default='train', type=str, choices=['train', 'validate', 'test'])
	parser.add_argument("--do_eval", default=False, action='store_true')
	parser.add_argument("--do_test", default=False, action='store_true')
	parser.add_argument("--bert_data_path", default='../data/')
	parser.add_argument("--model_path", default='../models/')
	parser.add_argument("--result_path", default='../results/cnndm')
	parser.add_argument("--pretrained_dir", default='../bert-base-chinese', help='pre-trained bert model')
	parser.add_argument("--bert_config_path", default='bert_config.json')

	parser.add_argument("--batch_size", default=3, type=int)
	parser.add_argument("--train_epochs", default=3, type=int)

	parser.add_argument("--max_seq_length", default=512, type=int)
	parser.add_argument("--lr", default=1, type=float)
	parser.add_argument('--l2reg', default=0.01, type=float)
	parser.add_argument("--dropout", default=0.1, type=float)

	parser.add_argument("--polarities_dim", default=3, type=int, help="The dimension of the output classes")

	parser.add_argument('--visible_gpus', default='-1', type=str)
	parser.add_argument('--gpu_ranks', default='1', type=str)
	parser.add_argument('--log_file', default='../logs/cnndm.log')
	parser.add_argument('--dataset', default='')
	parser.add_argument('--seed', default=666, type=int)

	parser.add_argument("--check_steps", default=500, type=int)
	parser.add_argument("--best_model", default='')
	parser.add_argument("--inter_layers", default=2, type=int, help="Number of layers in transformer decoder")

	parser.add_argument("--do_use_second_dataset", default=False)
	parser.add_argument("--second_dataset_path", default="./bert_data_char_title_entity/")

	parser.add_argument("--optim", default='adam', type=str)
	parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
	parser.add_argument("-hidden_size", default=128, type=int)
	parser.add_argument("-ff_size", default=512, type=int)
	parser.add_argument("-heads", default=4, type=int)
	parser.add_argument("-rnn_size", default=512, type=int)

	parser.add_argument("-param_init", default=0, type=float)
	parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)

	parser.add_argument("-beta1", default=0.9, type=float)
	parser.add_argument("-beta2", default=0.999, type=float)
	parser.add_argument("-decay_method", default='', type=str)
	parser.add_argument("-warmup_steps", default=50, type=int)
	parser.add_argument("-max_grad_norm", default=0, type=float)

	parser.add_argument("-accum_count", default=1, type=int)
	parser.add_argument("-world_size", default=1, type=int)
	parser.add_argument("-report_every", default=1, type=int)
	parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

	parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
	parser.add_argument("-train_from", default='')
	parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
	parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

	args = parser.parse_args()
	args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]

	init_logger(args.log_file)
	device = "cpu" if args.visible_gpus == '-1' else "cuda"
	device_id = int(args.visible_gpus) if device == "cuda" else -1
	os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
	if args.world_size > 1:
		multi_main(args)
	elif args.mode == 'train':
		train(args, device_id)
	elif args.mode == 'validate':
		wait_and_validate(args, device_id)
	# elif (args.mode == 'lead'):
	# 	baseline(args, cal_lead=True)
	# elif (args.mode == 'oracle'):
	# 	baseline(args, cal_oracle=True)
	elif args.mode == 'test':
		cp = args.best_model
		# 	try:
		# 		step = int(cp.split('.')[-2].split('_')[-1])
		# 	except:
		# 		step = 0
		test(args, device_id, cp)
