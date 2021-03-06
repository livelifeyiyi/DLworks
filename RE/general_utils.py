# coding=utf-8
import numpy
import numpy as np


def get_minibatches(data, minibatch_size, shuffle=True):
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
	list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
	data_size = len(data[0]) if list_data else len(data)
	indices = np.arange(data_size)
	if shuffle:
		np.random.shuffle(indices)
	for minibatch_start in np.arange(0, data_size, minibatch_size):
		minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
		yield [minibatch(d, minibatch_indices) for d in data] if list_data \
			else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
	return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def get_bags(data, relation_types, max_batchsize, shuffle=True):
	# get bags by relation type
	# [train_sentences_id, train_position_lambda, train_entity_tags, train_sentences_words, train_relation_tags, train_relation_names]
	list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
	# data_size = len(data[0]) if list_data else len(data)
	# indices = np.arange(data_size)
	if shuffle:
		np.random.shuffle(np.array(list(relation_types)))
	for relation_type in relation_types:
		bag_indices = numpy.argwhere(numpy.array(data[4]) == relation_type).reshape(-1)
		# print(relation_type, bag_indices)
		if shuffle:
			np.random.shuffle(bag_indices)
		if len(bag_indices) >= max_batchsize:
			for minibatch_start in np.arange(0, len(bag_indices), max_batchsize):
				minibatch_indices = bag_indices[minibatch_start:minibatch_start + max_batchsize]
				if len(minibatch_indices) < max_batchsize:
					continue
				yield [minibatch(d, minibatch_indices) for d in data] if list_data \
					else minibatch(data, minibatch_indices)
		else:
			continue
			# minibatch_indices = bag_indices
			# yield [minibatch(d, minibatch_indices) for d in data] if list_data \
			# 	else minibatch(data, minibatch_indices)


def padding_sequence(sequences, pad_token=0, pad_length=None):
	Y_lengths = [len(sentence) for sentence in sequences]
	# create an empty matrix with padding tokens
	if not pad_length:
		longest_sent = max(Y_lengths)
	else:
		longest_sent = pad_length
	batch_size = len(sequences)
	padded_Y = np.ones((batch_size, longest_sent)) * pad_token
	# copy over the actual sequences
	for i, y_len in enumerate(Y_lengths):
		sequence = sequences[i]
		padded_Y[i, 0:y_len] = sequence[:y_len]
	return padded_Y, longest_sent

	'''
	# assuming trailing dimensions and type of all the Tensors
	# in sequences are same and fetching those from sequences[0]
	max_size = sequences[0].size()
	trailing_dims = max_size[1:]
	max_len = max([s.size(0) for s in sequences])
	if batch_first:
		out_dims = (len(sequences), max_len) + trailing_dims
	else:
		out_dims = (max_len, len(sequences)) + trailing_dims

	out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
	for i, tensor in enumerate(sequences):
		length = tensor.size(0)
		# use index notation to prevent duplicate references to the tensor
		if batch_first:
			out_tensor[i, :length, ...] = tensor
		else:
			out_tensor[:length, i, ...] = tensor

	return out_tensor, max_len'''


def padding_sequence_recurr(sequences):
	"""
	pad relation sequence, each sentence may have several relations, pad them to a same length for RE_optimizer
	:param sequences: relation sequence
	:return: padded relation sequence
	"""
	Y_lengths = [len(sentence) for sentence in sequences]
	longest_sent = max(Y_lengths)
	batch_size = len(sequences)
	padded_Y = np.zeros((batch_size, longest_sent))
	for i, y_len in enumerate(Y_lengths):
		sequence = sequences[i]
		if longest_sent == y_len or y_len == 1:
			padded_Y[i] = sequence
		# elif longest_sent % y_len == 0:
		# 	padded_Y[i] = np.pad(sequence, ((longest_sent - y_len)/2, (longest_sent - y_len)/2), "edge")
		else:
			padded_Y[i] = np.pad(sequence, (0, longest_sent - y_len), "edge")
	return padded_Y