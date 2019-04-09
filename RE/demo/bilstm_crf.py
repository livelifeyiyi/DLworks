# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import os
torch.manual_seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0")
ROOT_DIR = "./NYT/"
# model_PATH = "./model/model_LSTMCRF.pkl"
# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
'''
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cpu")
model_PATH = "C:\\(O_O)!\\thesis\\5-RE with LSTM\\model\\model_LSTMCRF.pkl"
ROOT_DIR = "C:\\(O_O)!\\thesis\\5-RE with LSTM\\testData\\"
# torch.set_default_tensor_type('torch.DoubleTensor')
'''


# ####################################################################
# Helper functions to make the code more readable.
# import torch.multiprocessing as multiprocessing
# multiprocessing.set_start_method('spawn')

def argmax(vec):
	# return the argmax as a python int
	_, idx = torch.max(vec, 1)
	return idx.item()


'''def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)
'''

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + \
		torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

	def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch):
		super(BiLSTM_CRF, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.tag_to_ix = tag_to_ix
		self.tagset_size = len(tag_to_ix)
		self.batch = batch

		self.sente_padding_idx = int(word2id["UKN"])
		self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.sente_padding_idx)
		self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2,
							num_layers=1, bidirectional=True, batch_first=True)

		# Maps the output of the LSTM into tag space.
		self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

		# Matrix of transition parameters.  Entry i,j is the score of
		# transitioning *to* i *from* j.
		self.transitions = nn.Parameter(
			torch.randn(self.tagset_size, self.tagset_size, device=device))

		# These two statements enforce the constraint that we never transfer
		# to the start tag and we never transfer from the stop tag
		self.transitions.data[tag_to_ix[START_TAG], :] = -10000
		self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

		self.hidden = self.init_hidden()

	def init_hidden(self):
		# # The axes semantics are (num_layers* num_directions, minibatch_size, hidden_dim)
		return (torch.randn(2, self.batch, self.hidden_dim // 2, device=device),  # .cuda
				torch.randn(2, self.batch, self.hidden_dim // 2, device=device))

	def _forward_alg(self, feats):
		# Do the forward algorithm to compute the partition function
		init_alphas = torch.full((1, self.tagset_size), -10000., device=device)
		# START_TAG has all of the score.
		init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

		# Wrap in a variable so that we will get automatic backprop
		forward_var = init_alphas

		# Iterate through the sentence
		for feat in feats:
			alphas_t = []  # The forward tensors at this timestep
			for next_tag in range(self.tagset_size):
				# broadcast the emission score: it is the same regardless of
				# the previous tag
				emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
				# the ith entry of trans_score is the score of transitioning to
				# next_tag from i
				trans_score = self.transitions[next_tag].view(1, -1)
				# The ith entry of next_tag_var is the value for the
				# edge (i -> next_tag) before we do log-sum-exp
				next_tag_var = forward_var + trans_score + emit_score
				# The forward variable for this tag is log-sum-exp of all the
				# scores.
				alphas_t.append(log_sum_exp(next_tag_var).view(1))
			forward_var = torch.cat(alphas_t).view(1, -1)
		terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
		alpha = log_sum_exp(terminal_var)
		return alpha

	def _get_lstm_features(self, sentence):  # , sentence_len
		self.hidden = self.init_hidden()
		# embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
		# batch_size, seq_len = sentence.size()
		'''sentence_np = sentence.numpy()
		sentence_nopad = []
		for each in sentence_np:
			each_no_pad = []
			for word_id in each:
				if word_id == self.sente_padding_idx:
					break
				each_no_pad.append(word_id)
			sentence_nopad.append(each_no_pad)
		sentence_nopad = sorted(sentence_nopad, key=lambda i: len(i), reverse=True)
		sentence_len = [len(i) for i in sentence_nopad]  # list of sequences lengths of each batch element'''
		# embeds: (batch_size, seq_len, embedding_dim)
		embeds = self.word_embeds(sentence)  # .view(len(sentence), -1, self.embedding_dim)
		# (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, lstm_units/hidden_dim)
		# pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
		# embeds = nn.utils.rnn.pack_padded_sequence(embeds1, sentence_len, batch_first=True)
		lstm_out, self.hidden = self.bilstm(embeds, self.hidden)
		# undo the packing operation
		# lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
		# (batch_size, seq_len, lstm_units) -> (batch_size * seq_len, lstm_units)
		lstm_out = lstm_out.contiguous()
		lstm_out = lstm_out.view(-1,  lstm_out.shape[2])  # len(sentence), self.hidden_dim
		lstm_feats = self.hidden2tag(lstm_out)
		return lstm_feats

	def _score_sentence(self, feats, tags):
		# print(feats.size())
		# print(tags.size())
		# Gives the score of a provided tag sequence
		score = torch.zeros(1, device=device)
		tags = torch.cat((torch.tensor(
			[[self.tag_to_ix[START_TAG]] for j in range(tags.size()[0])],
			device=device, dtype=torch.long), tags), 1)  # dtype=torch.long,
		tags = tags.view(-1, 1)
		for i, feat in enumerate(feats):
			score = score + \
					self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
		score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
		return score

	def _viterbi_decode(self, feats):
		backpointers = []

		# Initialize the viterbi variables in log space
		init_vvars = torch.full((1, self.tagset_size), -10000., device=device)
		init_vvars[0][self.tag_to_ix[START_TAG]] = 0

		# forward_var at step i holds the viterbi variables for step i-1
		forward_var = init_vvars
		for feat in feats:
			bptrs_t = []  # holds the backpointers for this step
			viterbivars_t = []  # holds the viterbi variables for this step

			for next_tag in range(self.tagset_size):
				# next_tag_var[i] holds the viterbi variable for tag i at the
				# previous step, plus the score of transitioning
				# from tag i to next_tag.
				# We don't include the emission scores here because the max
				# does not depend on them (we add them in below)
				next_tag_var = forward_var + self.transitions[next_tag]
				best_tag_id = argmax(next_tag_var)
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
			# Now add in the emission scores, and assign forward_var to the set
			# of viterbi variables we just computed
			forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
			backpointers.append(bptrs_t)

		# Transition to STOP_TAG
		terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
		best_tag_id = argmax(terminal_var)
		path_score = terminal_var[0][best_tag_id]

		# Follow the back pointers to decode the best path.
		best_path = [best_tag_id]
		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
		# Pop off the start tag (we dont want to return that to the caller)
		start = best_path.pop()
		assert start == self.tag_to_ix[START_TAG]  # Sanity check
		best_path.reverse()
		return path_score, best_path

	def neg_log_likelihood(self, sentence, tags):
		'''sentence_np = sentence.numpy()
		sentence_nopad = []
		for each in sentence_np:
			each_no_pad = []
			for word_id in each:
				if word_id == self.sente_padding_idx:
					break
				each_no_pad.append(word_id)
			sentence_nopad.append(each_no_pad)
		sentence_nopad = sorted(sentence_nopad, key=lambda i: len(i), reverse=True)
		sentence_len = [len(i) for i in sentence_nopad]

		tags_np = tags.numpy()
		tags_nopad = []
		for each in tags_np:
			each_no_pad = []
			for word_id in each:
				if word_id == tag2id["UKN"]:
					break
				each_no_pad.append(word_id)
			tags_nopad.append(each_no_pad)
		tags_nopad = sorted(tags_nopad, key=lambda i: len(i), reverse=True)'''
		if torch.cuda.is_available():
			sentence = Variable(torch.cuda.LongTensor(sentence))
		else:
			sentence = Variable(torch.LongTensor(sentence))
		if torch.cuda.is_available():
			tags = Variable(torch.cuda.LongTensor(tags))
		else:
			tags = Variable(torch.LongTensor(tags))
		feats = self._get_lstm_features(sentence)
		# feats = self._get_lstm_features(torch.from_numpy(np.array(sentence_nopad)), torch.from_numpy(np.array(sentence_len)))
		forward_score = self._forward_alg(feats)
		gold_score = self._score_sentence(feats, tags)
		return forward_score - gold_score

	def forward(self, sentence):  # dont confuse this with _forward_alg above.
		if torch.cuda.is_available():
			sentence = Variable(torch.cuda.LongTensor(sentence))
		else:
			sentence = Variable(torch.LongTensor(sentence))
		# Get the emission scores from the BiLSTM
		lstm_feats = self._get_lstm_features(sentence)

		# Find the best path, given the features.
		score, tag_seq = self._viterbi_decode(lstm_feats)
		return score, tag_seq

#####################################################################
# Run training


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4  # 300
BATCH = 10

# training data

# ROOT_DIR = "C:\\(O_O)!\\thesis\\5-RE with LSTM\\testData\\"
# VECTOR_NAME = ROOT_DIR+"vec.txt"
with open(ROOT_DIR+'RE_data_train_pad.pkl', 'rb') as inp:
	word2id = pickle.load(inp, encoding='latin1')
	id2word = pickle.load(inp, encoding='latin1')
	tag2id = pickle.load(inp, encoding='latin1')
	train_x = pickle.load(inp, encoding='latin1')  # train sentence
	train_y = pickle.load(inp, encoding='latin1')  # train sequence

with open(ROOT_DIR+'RE_data_test_pad.pkl', 'rb') as inp:
	test_x = pickle.load(inp, encoding='latin1')  # test sentence
	test_y = pickle.load(inp, encoding='latin1')  # test sequence
print("train len", len(train_x))
print("test len", len(test_x))
print("word2id len", len(word2id))

'''
training_data = [(
		"the wall street journal reported today that apple corporation made money".split(),
		"B I I I O O O B I O O".split()
	), (
		"georgia tech is a university in georgia".split(),
		"B I O O O O B".split()
	)]
'''

word_to_ix = word2id
'''for sentence, tags in training_data:
	for word in sentence:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)'''
# start_stop = pd.Series([START_TAG, STOP_TAG])
# tag2id.append(start_stop, ignore_index=True)  # {"B": 0, "I": 1, "O": 2, START_TAG: 3, START_TAG: 4}
tag_to_ix = tag2id.to_dict()
tag_to_ix[START_TAG] = len(tag_to_ix.values())
tag_to_ix[STOP_TAG] = len(tag_to_ix.values())

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, BATCH)  # .to(device)
if torch.cuda.is_available():
	model = model.cuda()
# PATH = "./model/model_tony1.pkl"
# model.load_state_dict(torch.load(PATH))
print(model)
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
'''with torch.no_grad():
	precheck_sent = torch.tensor(train_x[0], device=device, dtype=torch.long)  #  .to(device)  # .cuda()
	precheck_tags = torch.tensor(train_y[0], device=device, dtype=torch.long)  #  .to(device)  # .cuda()
	# precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
	# precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
	print(model(precheck_sent))'''

if torch.cuda.is_available():
	train_x = torch.cuda.LongTensor(train_x, device=device)  # , dtype=torch.long .to(device)  # .cuda()
	train_y = torch.cuda.LongTensor(train_y, device=device)  # , dtype=torch.long
else:
	train_x = torch.LongTensor(train_x, device=device)  # , dtype=torch.long .to(device)  # .cuda()
	train_y = torch.LongTensor(train_y, device=device)

train_dataset = Data.TensorDataset(train_x, train_y)
train_dataloader = Data.DataLoader(train_dataset, BATCH, True)  # , num_workers=2
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
	# for sentence, tags in training_data:

	# for i in range(len(train_x)):
	#     sentence_in = torch.tensor(train_x[i], device=device, dtype=torch.long)  #  .to(device)  # .cuda()
	#     targets = torch.tensor(train_y[i], device=device, dtype=torch.long)  #  .to(device)  # .cuda()
	for step, (sentence_in, targets) in enumerate(train_dataloader):
		print("epoch %s, step %s" % (epoch, step))
		# sentence_in = torch.tensor(x, device=device, dtype=torch.long)  #  .to(device)  # .cuda()
		# targets = torch.tensor(y, device=device, dtype=torch.long)  #  .to(device)  # .cuda()
		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		# for i in range(len(x)):
		# 	sentence_in = x[i]
		# 	targets = y[i]
		model.zero_grad()

		# Step 2. Get our inputs ready for the network, that is,
		# turn them into Tensors of word indices.
		# sentence_in = prepare_sequence(sentence, word_to_ix)
		# targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

		# Step 3. Run our forward pass.
		loss = model.neg_log_likelihood(sentence_in, targets)
		if torch.cuda.is_available():
			loss = loss.cuda()
		# Step 4. Compute the loss, gradients, and update the parameters by
		# calling optimizer.step()
		loss.backward()
		optimizer.step()
		print(loss)
	torch.save(model, "./model/model_BiLSTM_CRF_%s.pkl"%epoch)
	print("%s-th model has been saved" % epoch)
torch.save(model, "./model/model_BiLSTM_CRF.pkl")
print("model has been saved")

'''
# Check predictions after training
with torch.no_grad():
	precheck_sent = train_x[0]
	print(model(precheck_sent))
'''