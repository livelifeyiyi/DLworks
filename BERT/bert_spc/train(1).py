# -*- coding: utf-8 -*-
# file: train.py
import json
import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy

from pytorch_pretrained_bert import BertModel, optimization
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import Tokenizer4Bert, ABSADataset, ABSADataset_sentence_pair, SADataset

# from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN
# from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
	def __init__(self, opt):
		self.opt = opt

		if 'bert' in opt.model_name:
			tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
			bert = BertModel.from_pretrained(opt.pretrained_bert_name)
			self.model = opt.model_class(bert, opt).to(opt.device)
		# else:
		#     tokenizer = build_tokenizer(
		#         fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
		#         max_seq_len=opt.max_seq_len,
		#         dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
		#     embedding_matrix = build_embedding_matrix(
		#         word2idx=tokenizer.word2idx,
		#         embed_dim=opt.embed_dim,
		#         dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
		#     self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
		if 'pair' in opt.model_name:
			if not self.opt.do_eval:
				self.trainset = ABSADataset_sentence_pair(opt.dataset_file['train'], tokenizer)
			self.testset = ABSADataset_sentence_pair(opt.dataset_file['test'], tokenizer)
		elif 'SA' in opt.model_name:
			if not self.opt.do_eval:
				self.trainset = SADataset(opt.dataset_file['train'], tokenizer)
			self.testset = SADataset(opt.dataset_file['test'], tokenizer)
		else:
			if not self.opt.do_eval:
				self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
			self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
		assert 0 <= opt.valset_ratio < 1
		if not self.opt.do_eval:
			if opt.valset_ratio > 0:
				valset_len = int(len(self.trainset) * opt.valset_ratio)
				self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
			else:
				self.valset = self.testset

		if opt.device.type == 'cuda':
			logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
		self._print_args()

	def _print_args(self):
		n_trainable_params, n_nontrainable_params = 0, 0
		for p in self.model.parameters():
			n_params = torch.prod(torch.tensor(p.shape))
			if p.requires_grad:
				n_trainable_params += n_params
			else:
				n_nontrainable_params += n_params
		logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
		logger.info('> training arguments:')
		for arg in vars(self.opt):
			logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

	def _reset_params(self):
		for child in self.model.children():
			if type(child) != BertModel:  # skip bert params
				for p in child.parameters():
					if p.requires_grad:
						if len(p.shape) > 1:
							self.opt.initializer(p)
						else:
							stdv = 1. / math.sqrt(p.shape[0])
							torch.nn.init.uniform_(p, a=-stdv, b=stdv)

	def _train(self, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
		max_val_acc = 0
		max_val_f1 = 0
		global_step = 0
		path = None
		for epoch in range(self.opt.num_epoch):
			logger.info('>' * 100)
			logger.info('epoch: {}'.format(epoch))
			n_correct, n_total, loss_total = 0, 0, 0
			# switch model to training mode
			self.model.train()

			for i_batch, sample_batched in enumerate(train_data_loader):
				global_step += 1
				# clear gradient accumulators
				optimizer.zero_grad()

				inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
				outputs = self.model(inputs)
				targets = sample_batched['polarity'].to(self.opt.device)

				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()

				n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
				n_total += len(outputs)
				loss_total += loss.item() * len(outputs)
				if global_step % self.opt.log_step == 0:
					train_acc = n_correct / n_total
					train_loss = loss_total / n_total
					logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
				if i_batch % 200 == 0:
					logger.info("**********ground_truth data test**********")
					test_acc, test_f1, pred_labels, label_ids = self._evaluate_acc_f1(test_data_loader)
					logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
					self.predict_vote(pred_labels, label_ids)
					if not os.path.exists('evaluate_state_dict'):
						os.mkdir('evaluate_state_dict')
					path = 'evaluate_state_dict/epoch{0}_batch{1}_test_acc{2}'.format(epoch, i_batch, round(test_acc, 4))
					torch.save(self.model.state_dict(), path)
				# f1 = metrics.f1_score(targets.cpu(), torch.argmax(outputs, -1).cpu(), labels=[0, 1, 2],
				#                     #                       average='macro')

			val_acc, val_f1, full_pred, full_label_ids = self._evaluate_acc_f1(val_data_loader)
			logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
			if val_acc > max_val_acc:
				max_val_acc = val_acc
				if not os.path.exists('evaluate_state_dict'):
					os.mkdir('evaluate_state_dict')
			path = 'evaluate_state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, epoch, round(val_acc, 4))
			torch.save(self.model.state_dict(), path)
			logger.info('>> saved: {}'.format(path))
			if val_f1 > max_val_f1:
				max_val_f1 = val_f1

			# self.model.eval()
			logger.info("**********ground_truth data test**********")
			test_acc, test_f1, pred_labels, label_ids = self._evaluate_acc_f1(test_data_loader)
			logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
			self.predict_vote(pred_labels, label_ids)
			if not self.opt.do_eval:
				logger.info("**********souhu dev data test**********")
				test_acc, test_f1, pred_labels, label_ids = self._evaluate_acc_f1(val_data_loader)
				logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
				self.predict_vote(pred_labels, label_ids)
		return path

	def _evaluate_acc_f1(self, data_loader):
		n_correct, n_total = 0, 0
		t_targets_all, t_outputs_all = None, None
		# switch model to evaluation mode
		self.model.eval()
		with torch.no_grad():
			full_logits = []
			full_pred = []
			full_label_ids = []
			for t_batch, t_sample_batched in enumerate(data_loader):
				t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
				t_targets = t_sample_batched['polarity'].to(self.opt.device)
				t_outputs = self.model(t_inputs)

				n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
				n_total += len(t_outputs)
				full_pred.extend(torch.argmax(t_outputs, -1).tolist())
				full_logits.extend(t_outputs.tolist())
				full_label_ids.extend(t_targets.tolist())

				if t_targets_all is None:
					t_targets_all = t_targets
					t_outputs_all = t_outputs
				else:
					t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
					t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

		acc = n_correct / n_total
		f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
		# with open('pred_res', encoding='utf-8', mode='a+') as tfile:
		#             tfile.write(str((torch.argmax(t_outputs_all, -1) == t_targets_all).cpu())+'\n')
		if self.opt.output_name:
			with open(self.opt.output_name, "a+") as fw:
				json.dump({"logits": full_logits, "pred_labels": full_pred, "label_ids": full_label_ids}, fw)

		return acc, f1, full_pred, full_label_ids

	def predict_vote(self, pred_labels, label_ids):
		# with open(self.opt.output_name, encoding='utf-8', mode='r') as infile:
		# 	results = json.load(infile)
		# pred_labels = results['pred_labels']
		# label_ids = results['label_ids']
		test_file = self.opt.dataset_file['test']

		act_pred_label = {}  # id:{'entity': '', 'emotion': 0/1/-1, 'predictions': []}
		prev_entity = ""
		# "pred: %s, act: %s" % (pred_labels[i] - 1, label_ids[i] - 1)
		with open(test_file, encoding='utf-8', mode='r') as infile:
			lines = infile.readlines()

			doc_id = 0
			for i in range(0, len(lines), 3):
				pred_id = int(i / 3)
				doc = lines[i]
				entity = lines[i + 1].lower().strip()
				polarity_str = lines[i + 2].strip()
				assert int(polarity_str) == (label_ids[pred_id] - 1)
				# print(polarity_str, label_ids[pred_id] - 1)
				if prev_entity == "" or entity != prev_entity:
					doc_id += 1
					prev_entity = entity
					act_pred_label[doc_id] = {}
					act_pred_label[doc_id]['entity'] = entity
					act_pred_label[doc_id]['emotion'] = int(polarity_str)
					act_pred_label[doc_id]['predictions'] = [int(pred_labels[pred_id] - 1)]
				else:
					# if entity == prev_entity:
					act_pred_label[doc_id]['predictions'].append(int(pred_labels[pred_id] - 1))

		# print(act_pred_label)
		acc = 0.
		total = len(act_pred_label)
		act_labels_all = []
		pred_labels_all = []
		for idx in act_pred_label.keys():
			each_pred = act_pred_label[idx]
			predic_labels = each_pred['predictions']
			act_label = each_pred['emotion']
			num = []
			num.append(predic_labels.count(-1))
			num.append(predic_labels.count(0))
			num.append(predic_labels.count(1))
			pred = num.index(max(num))
			if pred - 1 == act_label:
				acc += 1
			act_labels_all.append(act_label+1)
			pred_labels_all.append(pred)
		# print(acc / total)
		vote_f1 = metrics.f1_score(act_labels_all, pred_labels_all, labels=[0, 1, 2], average='macro')
		logger.info('>> vote_acc: {:.4f}, vote_f1: {:.4f}'.format(acc / total, vote_f1))

	def run(self):
		# Loss and Optimizer
		criterion = nn.CrossEntropyLoss()
		_params = filter(lambda p: p.requires_grad, self.model.parameters())
		optimizer = optimization.BertAdam(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

		test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

		self._reset_params()
		if not self.opt.do_eval:
			train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
			val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

			best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)
			self.model.load_state_dict(torch.load(best_model_path))
		else:
			self.model.load_state_dict(torch.load(self.opt.best_model_path))
		self.model.eval()
		logger.info("**********ground_truth data test**********")
		test_acc, test_f1, pred_labels, label_ids = self._evaluate_acc_f1(test_data_loader)
		logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
		self.predict_vote(pred_labels, label_ids)
		if not self.opt.do_eval:
			logger.info("**********souhu dev data test**********")
			test_acc, test_f1, pred_labels, label_ids = self._evaluate_acc_f1(val_data_loader)
			logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
			self.predict_vote(pred_labels, label_ids)


def main():
	# Hyper Parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', default='bert_spc', type=str, help='bert-spc, bert-sentence-pair, bert_SA')  # bert_spc
	parser.add_argument('--dataset_train', default=None, type=str, help='Dictionary of trainning dataset')
	parser.add_argument('--dataset_test', default=None, type=str, help='Dictionary of test dataset')
	# parser.add_argument('--optimizer', default='adam', type=str)
	parser.add_argument('--output_name', default=None, type=str)
	parser.add_argument('--initializer', default='xavier_uniform_', type=str)
	parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
	parser.add_argument('--dropout', default=0.1, type=float)  # 0.5
	parser.add_argument('--l2reg', default=0.01, type=float)  # 10-5
	parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
	parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
	parser.add_argument('--log_step', default=5, type=int)
	parser.add_argument('--embed_dim', default=300, type=int)
	parser.add_argument('--hidden_dim', default=300, type=int)
	parser.add_argument('--bert_dim', default=768, type=int)
	parser.add_argument('--pretrained_bert_name', default='bert-base-chinese/', type=str)
	parser.add_argument('--max_seq_len', default=100, type=int)  # 80
	parser.add_argument('--polarities_dim', default=3, type=int)
	parser.add_argument('--hops', default=3, type=int)
	parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
	parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
	parser.add_argument('--valset_ratio', default=0.2, type=float,
	                    help='set ratio between 0 and 1 for validation support')
	parser.add_argument("--do_eval", default=False, action='store_true', help="Whether to metric the model.")
	parser.add_argument("--best_model_path", default=None, type=str, help="if do_eval is True, input the model path")
	opt = parser.parse_args()

	if opt.seed is not None:
		random.seed(opt.seed)
		numpy.random.seed(opt.seed)
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	model_classes = {
		'bert_spc': BERT_SPC,
	}
	dataset_files = {
		'train': opt.dataset_train,
		'test': opt.dataset_test

	}
	input_colses = {
		'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
		'bert-sentence-pair': ['text_bert_indices', 'bert_segments_ids'],
		'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
		'bert_SA': ['text_bert_indices', 'bert_segments_ids'],
	}
	initializers = {
		'xavier_uniform_': torch.nn.init.xavier_uniform_,
		'xavier_normal_': torch.nn.init.xavier_normal,
		'orthogonal_': torch.nn.init.orthogonal_,
	}
	optimizers = {
		'adadelta': torch.optim.Adadelta,  # default lr=1.0
		'adagrad': torch.optim.Adagrad,  # default lr=0.01
		'adam': torch.optim.Adam,  # default lr=0.001
		'adamax': torch.optim.Adamax,  # default lr=0.002
		'asgd': torch.optim.ASGD,  # default lr=0.01
		'rmsprop': torch.optim.RMSprop,  # default lr=0.01
		'sgd': torch.optim.SGD,
	}

	opt.model_class = BERT_SPC  # model_classes[opt.model_name]
	opt.dataset_file = dataset_files  # [opt.dataset_dir]
	opt.inputs_cols = input_colses[opt.model_name]
	opt.initializer = initializers[opt.initializer]
	# opt.optimizer = optimization.BertAdam(lr=opt.learning_rate)  # optimizers[opt.optimizer]
	opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
		if opt.device is None else torch.device(opt.device)

	log_file = '{}-{}.log'.format(opt.model_name, strftime("%y%m%d-%H%M", localtime()))
	logger.addHandler(logging.FileHandler(log_file))

	ins = Instructor(opt)
	ins.run()


if __name__ == '__main__':
	main()