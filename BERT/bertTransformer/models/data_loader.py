import gc
import glob
import random

import numpy as np
import pandas as pd
import torch

from bertTransformer.others.logging import logger
#
#
# class Batch(object):
# 	def _pad(self, data, pad_id, width=-1):
# 		if (width == -1):
# 			width = max(len(d) for d in data)
# 		rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
# 		return rtn_data
#
# 	def __init__(self, data=None, device=None, is_test=False):
# 		"""Create a Batch from a list of examples."""
# 		if data is not None:
# 			self.batch_size = len(data)
# 			pre_src = [x[0] for x in data]
# 			pre_labels = [x[1] for x in data]
# 			pre_segs = [x[2] for x in data]
# 			pre_clss = [x[3] for x in data]
#
# 			src = torch.tensor(self._pad(pre_src, 0))
#
# 			labels = torch.tensor(pre_labels)  # self._pad(pre_labels, 0)
# 			segs = torch.tensor(self._pad(pre_segs, 0))
# 			mask = 1 - (src == 0)
#
# 			clss = torch.tensor(self._pad(pre_clss, -1))
# 			mask_cls = 1 - (clss == -1)
# 			clss[clss == -1] = 0
#
# 			setattr(self, 'clss', clss.to(device))
# 			setattr(self, 'mask_cls', mask_cls.to(device))
# 			setattr(self, 'src', src.to(device))
# 			setattr(self, 'labels', labels.to(device))
# 			setattr(self, 'segs', segs.to(device))
# 			setattr(self, 'mask', mask.to(device))
#
# 			if (is_test):
# 				src_str = [x[-2] for x in data]
# 				setattr(self, 'src_str', src_str)
# 				tgt_str = [x[-1] for x in data]
# 				setattr(self, 'tgt_str', tgt_str)
#
# 	def __len__(self):
# 		return self.batch_size
#
#
# def batch(data, batch_size):
# 	"""Yield elements from data in chunks of batch_size."""
# 	minibatch, size_so_far = [], 0
# 	for ex in data:
# 		minibatch.append(ex)
# 		size_so_far = simple_batch_size_fn(ex, len(minibatch))
# 		if size_so_far == batch_size:
# 			yield minibatch
# 			minibatch, size_so_far = [], 0
# 		elif size_so_far > batch_size:
# 			yield minibatch[:-1]
# 			minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
# 	if minibatch:
# 		yield minibatch
#
#
# def load_dataset(args, corpus_type, shuffle):
# 	"""
#     Dataset generator. Don't do extra stuff here, like printing,
#     because they will be postponed to the first loading time.
#     Args:
#         corpus_type: 'train' or 'valid'
#     Returns:
#         A list of dataset, the dataset(s) are lazily loaded.
#     """
# 	assert corpus_type in ["train", "valid", "test"]
#
# 	# def _lazy_dataset_loader(pt_file, corpus_type):
# 	dataset = torch.load(args.bert_data_path+corpus_type+'.data')
# 	logger.info('Loading %s dataset from %s, number of examples: %d' %
# 				(corpus_type, args.bert_data_path, len(dataset)))
# 	yield dataset
#
# 	# Sort the glob output by file name (by increasing indexes).
# 	# pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
# 	# if pts:
# 	#     if (shuffle):
# 	#         random.shuffle(pts)
# 	#     for pt in pts:
# 	#         yield _lazy_dataset_loader(pt, corpus_type)
# 	# else:
# 	#     # Only one inputters.*Dataset, simple!
# 	#     pt = args.bert_data_path  # + '.' + corpus_type + '.pt'
# 	#     yield _lazy_dataset_loader(pt, corpus_type)
#
#
# def simple_batch_size_fn(new, count):
# 	src, labels = new[0], new[1]
# 	global max_n_sents, max_n_tokens, max_size
# 	if count == 1:
# 		max_size = 0
# 		max_n_sents = 0
# 		max_n_tokens = 0
# 	max_n_sents = max(max_n_sents, len(src))
# 	max_size = max(max_size, max_n_sents)
# 	src_elements = count * max_size
# 	return src_elements
#
#
# class Dataloader(object):
# 	def __init__(self, args, datasets, batch_size,
# 				 device, shuffle, is_test):
# 		self.args = args
# 		self.datasets = datasets
# 		self.batch_size = batch_size
# 		self.device = device
# 		self.shuffle = shuffle
# 		self.is_test = is_test
# 		self.cur_iter = self._next_dataset_iterator(datasets)
#
# 		assert self.cur_iter is not None
#
# 	def __iter__(self):
# 		dataset_iter = (d for d in self.datasets)
# 		while self.cur_iter is not None:
# 			for batch in self.cur_iter:
# 				yield batch
# 			self.cur_iter = self._next_dataset_iterator(dataset_iter)
#
# 	def _next_dataset_iterator(self, dataset_iter):
# 		try:
# 			# Drop the current dataset for decreasing memory
# 			if hasattr(self, "cur_dataset"):
# 				self.cur_dataset = None
# 				gc.collect()
# 				del self.cur_dataset
# 				gc.collect()
#
# 			self.cur_dataset = next(dataset_iter)
# 		except StopIteration:
# 			return None
#
# 		return DataIterator(args=self.args,
# 							dataset=self.cur_dataset, batch_size=self.batch_size,
# 							device=self.device, shuffle=self.shuffle, is_test=self.is_test)
#
#
# class DataIterator(object):
# 	def __init__(self, args, dataset, batch_size, device=None, is_test=False,
# 				 shuffle=True):
# 		self.args = args
# 		self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
# 		self.iterations = 0
# 		self.device = device
# 		self.shuffle = shuffle
#
# 		self.sort_key = lambda x: len(x[1])
#
# 		self._iterations_this_epoch = 0
#
# 	def data(self):
# 		if self.shuffle:
# 			random.shuffle(self.dataset)
# 		xs = self.dataset
# 		return xs
#
# 	def preprocess(self, ex, is_test):
# 		src = ex['src']
# 		if ('labels' in ex):
# 			labels = ex['labels']
# 		else:
# 			labels = ex['src_sent_labels']
#
# 		segs = ex['segs']
# 		if (not self.args.use_interval):
# 			segs = [0] * len(segs)
# 		clss = ex['clss']
# 		src_txt = ex['src_txt']
# 		# tgt_txt = ex['tgt_txt']
#
# 		if (is_test):
# 			return src, labels, segs, clss, src_txt
# 		else:
# 			return src, labels, segs, clss
#
# 	def batch_buffer(self, data, batch_size):
# 		minibatch, size_so_far = [], 0
# 		for ex in data:
# 			if len(ex['src']) == 0:
# 				continue
# 			ex = self.preprocess(ex, self.is_test)
# 			if ex is None:
# 				continue
# 			minibatch.append(ex)
# 			size_so_far = len(minibatch)  #simple_batch_size_fn(ex, len(minibatch))
# 			if size_so_far == batch_size:
# 				yield minibatch
# 				minibatch, size_so_far = [], 0
# 			elif size_so_far > batch_size:
# 				yield minibatch  # [:-1]
# 				minibatch, size_so_far = minibatch, len(minibatch)   # [-1:], simple_batch_size_fn(ex, 1)
# 		if minibatch:
# 			yield minibatch
#
# 	def create_batches(self):
# 		""" Create batches """
# 		data = self.data()
# 		for buffer in self.batch_buffer(data, self.batch_size):  # * 50
#
# 			p_batch = sorted(buffer, key=lambda x: len(x[3]))
# 			# p_batch = batch(p_batch, self.batch_size)
#
# 			p_batch = list(p_batch)
# 			if (self.shuffle):
# 				random.shuffle(p_batch)
# 			for b in p_batch:
# 				yield b
#
# 	def __iter__(self):
# 		while True:
# 			self.batches = self.create_batches()
# 			for idx, minibatch in enumerate(self.batches):
# 				# fast-forward if loaded from state
# 				if self._iterations_this_epoch > idx:
# 					continue
# 				self.iterations += 1
# 				self._iterations_this_epoch += 1
# 				# batch = Batch(minibatch, self.device, self.is_test)
#
# 				yield minibatch  # batch
# 			return
#
#
# class DataLoadBatches(object):
# 	def __init__(self, args, dataset, batch_size, device=None, is_test=False,
# 				 shuffle=True):
# 		self.args = args
# 		self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
# 		self.iterations = 0
# 		self.device = device
# 		self.shuffle = shuffle
#
# 	def data(self):
# 		if self.shuffle:
# 			random.shuffle(self.dataset)
# 		xs = self.dataset
# 		return xs
#
# 	def batch_buffer(self, data, batch_size):
# 		minibatch, size_so_far = [], 0
# 		for ex in data:
# 			if len(ex['src']) == 0:
# 				continue
# 			ex = self.preprocess(ex, self.is_test)
# 			if ex is None:
# 				continue
# 			minibatch.append(ex)
# 			size_so_far = len(minibatch)  #simple_batch_size_fn(ex, len(minibatch))
# 			if size_so_far == batch_size:
# 				yield minibatch
# 				minibatch, size_so_far = [], 0
# 			elif size_so_far > batch_size:
# 				yield minibatch  # [:-1]
# 				minibatch, size_so_far = minibatch, len(minibatch)   # [-1:], simple_batch_size_fn(ex, 1)
# 		if minibatch:
# 			yield minibatch
#
# 	def get_batch(self):
# 		self.batches = self.batch_buffer(self.data(), self.batch_size)
# 		for idx, minibatch in enumerate(self.batches):
# 			# fast-forward if loaded from state
#
# 			self.iterations += 1
# 			batch = Batch(minibatch, self.device, self.is_test)
#
# 			yield batch


def _pad(data, pad_id, width=-1):
	if (width == -1):
		width = max(len(d) for d in data)
	# rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
	rtn_data = data + [pad_id] * (width-len(data))
	return rtn_data


def preprocess(ex, max_seq_length, max_cls):
	src = ex['src']
	if ('labels' in ex):
		labels = ex['labels']
	else:
		labels = ex['src_sent_labels']
	segs = ex['segs']
	# if (not self.args.use_interval):
	# 	segs = [0] * len(segs)
	clss = ex['clss']
	src_txt = ex['src_txt']
	# tgt_txt = ex['tgt_txt']

	src = _pad(src, 0, max_seq_length)  # torch.tensor().to(device)
	# labels = torch.tensor(labels).to(device)
	segs = _pad(segs, 0, max_seq_length)  # torch.tensor().to(device)
	# clss = _pad(clss, -1, max_cls)  # torch.tensor().to(device)
	# mask = [(1 - (i == 0)) for i in src]
	# mask_cls = [(1 - (i == -1)) for i in clss]
	# clss[clss == -1] = 0
	return [src, labels, segs, clss]


def get_minibatches(data_dict, minibatch_size, max_seq_length, shuffle=True):
	"""
	Iterates through the provided data one minibatch at at time. You can use this function to
	iterate through data in minibatches as follows:
		for inputs_minibatch in get_minibatches(inputs, minibatch_size):
			...
	Or with multiple data sources:
		for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
			...
	Args:
		data: there are two possible values:
			- a list or numpy array
			- a list where each element is either a list or numpy array
		minibatch_size: the maximum number of items in a minibatch
		shuffle: whether to randomize the order of returned data
	Returns:
		minibatches: the return value depends on data:
			- If data is a list/array it yields the next minibatch of data.
			- If data a list of lists/arrays it returns the next minibatch of each element in the
			  list. This can be used to iterate through multiple data sources
			  (e.g., features and labels) at the same time.
	"""
	data = []
	# max_cls = 0
	# for ex in data_dict:
	# 	cls_lenth = len(ex['clss'])
	# 	if cls_lenth > max_cls:
	# 		max_cls = cls_lenth
	for ex in data_dict:
		if len(ex['src']) == 0:
			continue
		ex = preprocess(ex, max_seq_length, max_cls=0)
		if ex is None:
			continue
		data.append(ex)

	# list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
	data_size = len(data)  # if list_data else len(data)
	indices = np.arange(data_size)
	if shuffle:
		np.random.shuffle(indices)
	for minibatch_start in np.arange(0, data_size, minibatch_size):
		minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
		yield minibatch(data, minibatch_indices)  # [minibatch(d, minibatch_indices) for d in data] if list_data \


def minibatch(data, minibatch_idx):
	return pd.DataFrame(data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx])


def minibatch_WDP(data, minibatch_idx):
	return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def get_minibatches_WDP(data, minibatch_size, shuffle=True):
	list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
	data_size = len(data[0]) if list_data else len(data)
	indices = np.arange(data_size)
	if shuffle:
		np.random.shuffle(indices)
	for minibatch_start in np.arange(0, data_size, minibatch_size):
		minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
		yield [minibatch_WDP(d, minibatch_indices) for d in data] if list_data \
			else minibatch_WDP(data, minibatch_indices)