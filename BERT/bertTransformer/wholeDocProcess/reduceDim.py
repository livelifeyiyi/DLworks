"""
This file process the long input with various length into 512 dimensional
"""
import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from bertTransformer.models.encoder import TransformerInterEncoder, Classifier, RNNEncoder
from bertTransformer.models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
	""" Build optimizer """
	saved_optimizer_state_dict = None

	if args.train_from != '':
		optim = checkpoint['optim']
		saved_optimizer_state_dict = optim.optimizer.state_dict()
	else:
		optim = Optimizer(
			args.optim, args.lr, args.max_grad_norm,
			beta1=args.beta1, beta2=args.beta2,
			decay_method=args.decay_method,
			warmup_steps=args.warmup_steps)

	optim.set_parameters(list(model.named_parameters()))

	if args.train_from != '':
		optim.optimizer.load_state_dict(saved_optimizer_state_dict)
		if args.visible_gpus != '-1':
			for state in optim.optimizer.state.values():
				for k, v in state.items():
					if torch.is_tensor(v):
						state[k] = v.cuda()

		if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
			raise RuntimeError(
				"Error: loaded Adam optimizer from existing model" +
				" but optimizer state is empty")

	return optim


class Bert(nn.Module):
	def __init__(self, pretrained_dir, load_pretrained_bert, bert_config):
		super(Bert, self).__init__()
		if load_pretrained_bert:
			self.model = BertModel.from_pretrained(pretrained_dir)
		else:
			self.model = BertModel(bert_config)

	def forward(self, x):  # , segs, mask
		encoded_layers, _ = self.model(x)  # , segs, attention_mask=mask
		top_vec = encoded_layers[-1]
		return top_vec


class DimReducer(nn.Module):
	def __init__(self, args, device, load_pretrained_bert=True, bert_config=None):
		super(DimReducer, self).__init__()
		self.args = args
		self.device = device
		self.bert = Bert(args.pretrained_dir, load_pretrained_bert, bert_config)
		self.bert_vocab_size = self.bert.model.config.vocab_size
		self.hidden_size = self.bert.model.config.hidden_size
		self.to(device)

	def load_cp(self, pt):
		self.load_state_dict(pt['model'], strict=True)

	def avg_pool(self, x, target_dim):
		"""
		by using the slide window to get the average vector with target dim
		:param x: np.array([float]), (1, seq_len, 768)
		:return: (1, max_seq_len, 768)
		"""
		# x = x.item()
		res = None
		for vec in x:  # (seq_len, 768)
			seq_len = len(vec)  # .shape[0]
			# vec_idx = [i for i in range(seq_len)]
			window_size = seq_len + 1 - target_dim  #  seq_len // target_dim + 1  #
			idx = 0
			vec_new = None
			while idx <= seq_len - window_size:
				# if idx + window_size >= seq_len:
				# 	vec_win = vec[idx:]
				# else:
				vec_win = vec[idx:idx + window_size]
				if vec_new is None:
					vec_new = np.average(vec_win, axis=0).reshape(1, -1)
				else:
					vec_new = np.append(vec_new, np.average(vec_win, axis=0).reshape(1, -1), axis=0)
				idx += 1  # window_size
			assert len(vec_new) == target_dim  # (max_seq_len, 768)
			if not res:
				res = vec_new.reshape(1, target_dim, -1) #
			else:
				res = np.append(res, vec_new.reshape(1, target_dim, -1), axis=0)  # .reshape(1, target_dim, -1)
		return res  # torch.tensor(res).to(self.device)

	def forward(self, x):
		"""
		:param x: input token_ids, (1, seq_len)
		:return: (max_seq_len, 768)
		"""
		max_seq_len = self.args.max_seq_length
		if len(x) <= max_seq_len:
			if len(x) < max_seq_len:
				x = torch.cat(x, torch.cuda.LongTensor([0] * (max_seq_len - len(x))))
			if torch.cuda.is_available():
				x_in = torch.cuda.LongTensor(x).to(self.device).reshape(1, -1)
			else:
				x_in = torch.LongTensor(x).to(self.device).reshape(1, -1)
			vec_x = self.bert(x_in)
			return vec_x.cpu().detach().numpy()  # .reshape(1, max_seq_len, -1)
		# idx = [i for i in range(len(x))]
		all_vec = None
		j = 0
		while j < len(x):
			if j + max_seq_len >= len(x):
				x_seg = x[j:]
			else:
				x_seg = x[j: j+max_seq_len]
			if torch.cuda.is_available():
				x_in = torch.cuda.LongTensor(x_seg).to(self.device).reshape(1, -1)
			else:
				x_in = torch.LongTensor(x_seg).to(self.device).reshape(1, -1)
			vec_seg = self.bert(x_in).cpu()  # (1, seq_len, 768)
			if all_vec is not None:
				all_vec = np.append(all_vec, vec_seg.detach().numpy(), axis=1)
			else:
				all_vec = vec_seg.detach().numpy()
			j += max_seq_len

		return self.avg_pool(all_vec, max_seq_len)


class Decoder(nn.Module):
	def __init__(self, args, device, hidden_size, bert_vocab_size):
		super(Decoder, self).__init__()
		self.device = device
		self.args = args
		if args.encoder == 'classifier':
			self.encoder = Classifier(hidden_size)
		elif args.encoder == 'transformer':
			self.encoder = TransformerInterEncoder(hidden_size, args.ff_size, args.heads,
												   args.dropout, args.inter_layers, args.polarities_dim)
		elif args.encoder == 'rnn':
			self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
									  input_size=hidden_size, hidden_size=args.rnn_size, tag_size=args.polarities_dim,
									  dropout=args.dropout, batch_size=args.batch_size)
		elif args.encoder == 'baseline':
			bert_config = BertConfig(bert_vocab_size, hidden_size=args.hidden_size,
									 num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
			self.bert.model = BertModel(bert_config)
			self.encoder = Classifier(hidden_size)

		if args.param_init != 0.0:
			for p in self.encoder.parameters():
				p.data.uniform_(-args.param_init, args.param_init)
		if args.param_init_glorot:
			for p in self.encoder.parameters():
				if p.dim() > 1:
					xavier_uniform_(p)

	def forward(self, top_vec):
		# (batch_size, max_seq_len, 768)
		if torch.cuda.is_available():
			mask = torch.cuda.ByteTensor((1 - (top_vec == 0))).to(self.device)
		else:
			mask = torch.ByteTensor((1 - (top_vec == 0))).to(self.device)

		x = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1)]  # top_vec[, clss]  # get vectors of [cls] tags
		# sents_vec = sents_vec * mask_cls[:, :, None].float()  # (batch_size, sentences_num, 768)

		logits = self.encoder(self.args.model_name, x, mask)
		return logits
