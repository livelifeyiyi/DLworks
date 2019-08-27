# -*- coding: utf-8 -*-
import codecs
import pickle
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as D
from torch.autograd import Variable
from pytorch_transformers import BertModel
from general_utils import get_minibatches, padding_sequence
from torch.nn.utils.rnn import pad_sequence
import os
from RE import Jointly_RL

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SOS_token = 0
# EOS_token = 1


class JointModel(nn.Module):
	def __init__(self, config, embedding_pre, dim, relation_count):
		super(JointModel, self).__init__()
		self.encoder_model = config.encoder_model
		if self.encoder_model == "BiLSTM":
			self.encoder = EncoderRNN(config, embedding_pre)
		elif self.encoder_model == "BERT":
			self.encoder = EncoderBert(config)
		self.decoder = DecoderRNN(config)
		self.relation_model = RelationDecoder(config, dim, relation_count)  # Jointly_RL.RelationModel(config, dim, statedim, relation_count, noisy_count)


class EncoderRNN(nn.Module):
	def __init__(self, config, embedding_pre):
		super(EncoderRNN, self).__init__()
		self.batch = config.batchsize
		self.embedding_dim = config.embedding_dim
		self.hidden_dim = config.hidden_dim
		# self.tag_size = config['TAG_SIZE']
		self.pretrained = config.pretrain_vec
		self.dropout = config.dropout
		if self.pretrained:
			if torch.cuda.is_available():
				self.embedding = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(embedding_pre, device=device), freeze=False)
			else:
				self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre, device=device), freeze=False)
		else:
			self.embedding_size = config.embedding_size + 1
			self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
		# self.embedding = nn.Embedding(input_size, hidden_size)
		self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=self.dropout)

	def forward(self, input, pos_tensor, hidden):
		embedded = self.embedding(input)
		embedded_pos = torch.mul(embedded, pos_tensor.view(self.batch, -1, 1))
		# embedded = embedded.view(-1, self.batch, self.embedding_dim)  # .view(1, 1, -1)
		# x_t dim:(seq, batch, feature)
		# embedded = torch.transpose(embedded, 0, 1)
		output, hidden = self.bilstm(embedded_pos, hidden)
		return output, hidden

	def initHidden_bilstm(self):
		# (layers*direction, batch, hidden)each  (h_0, c_0)
		return (torch.randn(4, self.batch, self.hidden_dim // 2, device=device),
				torch.randn(4, self.batch, self.hidden_dim // 2, device=device))  # if use_cuda, .cuda


class EncoderBert(nn.Module):
	def __init__(self, config, load_pretrained_bert=True):
		super(EncoderBert, self).__init__()
		if load_pretrained_bert:
			self.model = BertModel.from_pretrained(config.pretrained_dir)
		else:
			self.model = BertModel(config.pretrained_dir+'bert_config.json')
		self.encoder_model = config.encoder_model

	def forward(self, x, mask):  # , segs
		encoded_layers, pooled_output = self.model(x, attention_mask=mask)  # , segs
		top_vec = encoded_layers[-1]
		return top_vec


class DecoderRNN(nn.Module):
	def __init__(self, config):
		super(DecoderRNN, self).__init__()
		self.batch = config.batchsize
		self.embedding_dim = config.embedding_dim
		self.hidden_dim = config.hidden_dim
		# self.tag_size = config['TAG_SIZE']
		# self.pretrained = config.pretrain_vec
		self.dropout = config.dropout
		self.tag_size = config.entity_tag_size + 1
		# if self.pretrained:
		# 	if torch.cuda.is_available():
		# 		self.embedding = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(embedding_pre, device=device), freeze=False)
		# 	else:
		# 		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre, device=device), freeze=False)
		# else:
		# 	self.embedding_size = config.embedding_size + 1
		# 	self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
		# input_size=self.embedding_dim*2
		self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True, num_layers=2, dropout=self.dropout)  # hidden_size, hidden_size)
		self.entity_embeds = nn.Embedding(self.tag_size, self.hidden_dim)
		self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)  # out
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden=None):
		# output = self.embedding(input).view(-1, self.batch, self.embedding_dim)   # ??? need embedding??
		output = F.relu(input)
		output, hidden = self.lstm(output, hidden)
		output_tag = self.softmax(self.hidden2tag(output))
		return output, output_tag, hidden

	def initHidden(self):
		# (layers*direction, batch, hidden)
		return torch.zeros(2, self.batch, self.hidden_dim, device=device)


class RelationDecoder(nn.Module):
	def __init__(self, args, dim, relation_count):
		super(RelationDecoder, self).__init__()
		self.dropout = args.dropout
		self.batch = args.batchsize
		self.dim = dim
		self.tag_size = relation_count + 1
		# self.hid2state = nn.Linear(dim * 2 + dim, dim)  # statedim
		# self.state2prob_relation = nn.Linear(dim, self.tag_size)  # + 1
		self.att_weight = nn.Parameter(torch.randn(self.batch, 1, self.dim))  # (self.batch, 1, self.hidden_dim)
		self.lstm = nn.LSTM(input_size=dim * 2, hidden_size=dim, num_layers=2,batch_first=True,
							dropout=self.dropout)  # dim * 2 + dim batch_first=True,hidden_size, hidden_size)
		self.relation_bias = nn.Parameter(torch.randn(self.batch, self.tag_size, 1))  # self.batch, self.tag_size, 1
		self.dropout_lstm = nn.Dropout(p=self.dropout)
		# self.dropout_att = nn.Dropout(p=self.dropout)
		self.relation_embeds = nn.Embedding(self.tag_size, self.dim)

	def attention(self, H):  # input: (batch/1, hidden, seq); output: (batch/1, hidden, 1)
		M = torch.tanh(H)
		a = F.softmax(torch.bmm(self.att_weight, M), 2)
		a = torch.transpose(a, 1, 2)
		return torch.bmm(H, a)

	def forward(self, hidden, encoder_output, decoder_output):
		if torch.cuda.is_available():
			encoder_output = encoder_output.cuda()  # Variable(torch.cuda.LongTensor(encoder_output, device=device)).cuda()
			decoder_output = decoder_output.cuda()  # Variable(torch.cuda.LongTensor(decoder_output, device=device)).cuda()
			# memory = memory.cuda()  # Variable(torch.cuda.LongTensor(memory, device=device)).cuda()

		seq_vec = torch.cat((encoder_output.view(self.batch, -1, self.dim), decoder_output.view(self.batch, -1, self.dim)), 2)
		lstm_out, hidden = self.lstm(seq_vec, hidden)
		lstm_out = self.dropout_lstm(lstm_out)

		sentence_vec = torch.tanh(self.attention(torch.transpose(lstm_out, 1, 2)))  # (1, dim*2, 1)
		# self.state2prob_relation(sentence_vec.squeeze())

		'''inp = sentence_vec  # torch.cat((sentence_vec.view(-1), memory), 0)  # (2100-300)
		outp = F.dropout(torch.tanh(self.hid2state(inp)), training=training)
		prob_relation = self.softmax(self.state2prob_relation(outp))'''

		# output = F.relu(inp)

		# att_out = torch.tanh(self.attention(lstm_out.view(self.batch, self.hidden_dim, -1)))
		# att_out = self.dropout_att(att_out)
		relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(self.batch, 1)  # (batch, 1)
		if torch.cuda.is_available():
			relation = relation.cuda()
		relation = self.relation_embeds(relation)
		res = torch.add(torch.bmm(relation, sentence_vec), self.relation_bias)
		res = F.softmax(res, 1)
		# prob_relation = F.softmax(self.state2prob_relation(outp.view(-1)), dim=0)  # self.softmax(self.state2prob_relation(outp.view(-1)))

		return lstm_out, res.squeeze()  # .view(-1)


# class AttnDecoderRNN(nn.Module):

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

teacher_forcing_ratio = 0.5


def train(input_tensor, pos_tensor, target_tensor, relation_target_tensor, JointModel, optimizer, criterion, BATCH, TEST=False, mask=None):  # max_length=MAX_LENGTH

	optimizer.zero_grad()
	encoder = JointModel.encoder
	decoder = JointModel.decoder
	relation_decoder = JointModel.relation_model

	# target_length = target_tensor.size(0)
	# seq_length = target_tensor.size(1)
	# target_length = target_tensor.size(0)
	# one word by one ?????
	# encoder_outputs = torch.zeros(input_length, encoder_hidden, device=device)  # max_length
	# for ei in range(input_length):
	# 	encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
	# 	encoder_outputs[ei] = encoder_output[0, 0]

	if JointModel.encoder_model == "BiLSTM":
		encoder_hidden = encoder.initHidden_bilstm()
		# input batch
		encoder_outputs, encoder_hidden = encoder(input_tensor, pos_tensor, encoder_hidden)
		# input_tensor: (batch, seq); encoder_hidden: (layer*direction, batch, hidden_dim//2)
		# encoder_outputs: (batch, seq, hidden_dim//2*2); encoder_hidden: (layer*direction, batch, hidden_dim//2)

		# decoder_input = torch.tensor([[SOS_token] for i in range(BATCH)], device=device).view(BATCH, 1, 1)
		# for one-layer
		# decoder_hidden = (torch.cat((encoder_hidden[0][0], encoder_hidden[0][1]), 1).view(1, BATCH, -1),
		# 				  torch.cat((encoder_hidden[1][0], encoder_hidden[1][1]), 1).view(1, BATCH, -1))# encoder_hidden
		# for 2 layer
		h1 = torch.cat((encoder_hidden[0][0], encoder_hidden[0][1]), 1).view(1, BATCH, -1)  # concat forward and backward hidden at layer1
		h2 = torch.cat((encoder_hidden[0][2], encoder_hidden[0][3]), 1).view(1, BATCH, -1)  # layer2
		c1 = torch.cat((encoder_hidden[1][0], encoder_hidden[1][1]), 1).view(1, BATCH, -1)
		c2 = torch.cat((encoder_hidden[1][2], encoder_hidden[1][3]), 1).view(1, BATCH, -1)
		decoder_hidden = (torch.cat((h1, h2), 0),
						  torch.cat((c1, c2), 0))  # (layer*direction, batch, hidden_dim)

		decoder_input = encoder_outputs
		decoder_output, decoder_output_tag, decoder_hidden = decoder(decoder_input, decoder_hidden)  # entity
		RE_output, RE_output_tag = relation_decoder(decoder_hidden, encoder_outputs, decoder_output)  # relation
	else:
		top_vec = encoder(input_tensor, mask)
		decoder_output, decoder_output_tag, decoder_hidden = decoder(top_vec)
		RE_output, RE_output_tag = relation_decoder(decoder_hidden, top_vec, decoder_output)

	# (batch, seq, hidden_dim)  (layer*direction, batch, hidden_dim)
	# decoder_output_T = decoder_output_tag.transpose(0, 1)  # (batch, seq, hidden_dim) -- >(seq, batch, hidden_dim)
	# target_tensor_T = target_tensor.transpose(0, 1)  # (batch, seq) --> (seq, batch)
	# for i in range(target_length):
		# Only keep active parts of the loss
	if mask is not None:
		active_loss = mask.view(-1) == 1
		NER_active_logits = decoder_output_tag.view(-1, decoder.tag_size)[active_loss]
		NER_active_labels = target_tensor.view(-1)[active_loss]
	else:
		NER_active_logits = decoder_output_tag.view(-1, decoder.tag_size)
		NER_active_labels = target_tensor.view(-1)

	if not TEST:
		loss_entity = criterion(NER_active_logits, NER_active_labels)
		# for j in range(target_length):  # each sentence  # seq_length
		# 	loss += criterion(decoder_output_tag[j], target_tensor[j])
		# loss /=  float(target_length)
		# loss_RE = criterion(RE_output_tag, relation_target_tensor)
		loss_RE = 0
		for i in range(len(relation_target_tensor[0])):
			target = torch.transpose(relation_target_tensor, 0, 1)[i]
			loss_RE += criterion(RE_output_tag, target)
		loss_RE /= len(relation_target_tensor[0])
		loss = 0.4*loss_entity + 0.6*loss_RE

		'''
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
		if use_teacher_forcing:
			# Teacher forcing: Feed the target as the next input
			for di in range(target_length):
				decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
				loss += criterion(decoder_output, target_tensor[di])
				decoder_input = target_tensor[di]  # Teacher forcing
		else:
			# Without teacher forcing: use its own predictions as the next input
			for di in range(target_length):
				decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
				topv, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze().detach()  # detach from history as input
	
				loss += criterion(decoder_output, target_tensor[di])
				if decoder_input.item() == EOS_token:
					break
		'''

		loss.backward()  # retain_graph=True

		optimizer.step()

		return loss.item(), NER_active_logits, NER_active_labels, RE_output_tag, relation_target_tensor  # encoder_outputs, decoder_output, decoder_output_tag, decoder_hidden
	else:
		return NER_active_logits, NER_active_labels, RE_output_tag, relation_target_tensor  # encoder_outputs, decoder_output, decoder_output_tag, decoder_hidden


def trainEpoches(encoder, decoder, criterion, print_every=10, learning_rate=0.001, l2=0.0001):
	start = time.time()
	out_losses = []
	print_loss_total = 0  # Reset every print_every
	# plot_loss_total = 0  # Reset every plot_every

	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)
	# training_pairs = [tensorsFromPair(random.choice(pairs))
	# 				  for i in range(n_iters)]

	# for iter in range(1, n_iters + 1):
	# training_pair = training_pairs[iter - 1]
	# for epoch in range(epoches):
	# i = 0
	mini_batches = get_minibatches(train_datasets, BATCH)
	batches_size = len(train_datasets[0]) // BATCH  # len(list(mini_batches))
	for i, data in enumerate(mini_batches):
		if i == batches_size:
			break
		# for i, data in enumerate(train_dataloader, 1):
		sentences, tags = data
		input_tensor, input_length = padding_sequence(sentences, pad_token=EMBEDDING_SIZE)
		target_tensor, target_length = padding_sequence(tags, pad_token=TAG_SIZE)
		if torch.cuda.is_available():
			input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
			target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
		else:
			input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
			target_tensor = Variable(torch.LongTensor(target_tensor, device=device))

		loss = train(input_tensor, target_tensor, encoder,
					 decoder, encoder_optimizer, decoder_optimizer, criterion)  # , input_length, target_length
		out_losses.append(loss)
		print_loss_total += loss
		# plot_loss_total += loss

		if i % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print(' (%d %d%%) %.4f' % (i, float(i) / batches_size * 100, print_loss_avg))
			# print('%s (%d %d%%) %.4f' % (timeSince(start, float(i) / batches_size),
			# i, float(i) / batches_size * 100, print_loss_avg))

		# plot_loss_avg = plot_loss_total / plot_every
		# plot_losses.append(plot_loss_avg)
		# plot_loss_total = 0
		# i += 1
	np.save("loss", out_losses)
	if epoch % 10 == 0:
		model_name = "./model/model_encoder_epoch" + str(epoch) + ".pkl"
		torch.save(encoder, model_name)
		model_name = "./model/model_decoder_epoch" + str(epoch) + ".pkl"
		torch.save(decoder, model_name)
		print("Model has been saved")
	# showPlot(plot_losses)


def run_this_model_only():
	# ROOT_DIR = "E:\\newFolder\\data\\entity&relation_dataset\\NYT10\\"
	ROOT_DIR = "NYT10/"
	# ROOT_DIR = "C:\\(O_O)!\\thesis\\5-RE with LSTM\\code\\testData\\"
	with open(ROOT_DIR+'RE_data_train.pkl', 'rb') as inp:
		# with codecs.open(ROOT_DIR+'RE_data_train.pkl', 'rb', encoding="utf-8") as inp:
		id2word = pickle.load(inp)  # , encoding='latin1'
		# tag2id = pickle.load(inp)
		train_x = pickle.load(inp)  # train sentence
		train_y = pickle.load(inp)


	EMBEDDING_SIZE = len(id2word)
	EMBEDDING_DIM = 300
	HIDDEN_DIM = 600  # 300
	TAG_SIZE = 7  # len(tag2id)
	BATCH = 128  # 100
	EPOCHS = 100  # 100
	# MAX_LENGTH = 188  # max length of the sentences
	VECTOR_NAME = "vector.txt"
	DROPOUT = 0.5
	LR = 0.001  # learning rate
	L2 = 0.0001

	config = {}
	config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
	config['EMBEDDING_DIM'] = EMBEDDING_DIM
	config['HIDDEN_DIM'] = HIDDEN_DIM
	config['TAG_SIZE'] = TAG_SIZE
	config['BATCH'] = BATCH
	config["pretrained"] = False
	config["dropout"] = DROPOUT

	embedding_pre = []
	if len(sys.argv) == 2 and sys.argv[1] == "pretrained":
		print("use pretrained embedding")
		config["pretrained"] = True
		word2vec = {}
		with codecs.open(VECTOR_NAME, 'r', 'utf-8') as input_data:
			for line in input_data.readlines():
				word2vec[line.split()[0]] = map(eval, line.split()[1:])

		unknow_pre = []
		unknow_pre.extend([1] * 100)
		embedding_pre.append(unknow_pre)  # wordvec id 0
		for word in word2id:
			if word2vec.has_key(word):
				embedding_pre.append(word2vec[word])
			else:
				embedding_pre.append(unknow_pre)

		embedding_pre = np.asarray(embedding_pre)
		print(embedding_pre.shape)


	# if torch.cuda.is_available():
	# 	train_x = torch.cuda.LongTensor(train_x, device=device)  # .cuda()  # train_x[:len(train_x) - len(train_x) % BATCH]
	# 	train_y = torch.cuda.LongTensor(train_y, device=device)  # .cuda()  # train_y[:len(train_x) - len(train_x) % BATCH]
	# else:
	# 	train_x = torch.LongTensor(train_x, device=device)  # .cuda()
	# 	train_y = torch.LongTensor(train_y, device=device)  # .cuda()

	# train_dataloader = D.DataLoader(train_datasets, BATCH, True)


	encoder1 = EncoderRNN(config, embedding_pre).to(device)
	decoder1 = DecoderRNN(config, embedding_pre).to(device)
	criterion = nn.NLLLoss()  # CrossEntropyLoss()
	# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
	if torch.cuda.is_available():
		encoder1 = encoder1.cuda()
		attn_decoder1 = decoder1.cuda()
		criterion = criterion.cuda()

	for epoch in range(EPOCHS):
		print("Epoch-" + str(epoch) + "."*10)
		train_datasets = [train_x, train_y]  # D.TensorDataset(train_x, train_y)

		trainEpoches(encoder1, decoder1, criterion, learning_rate=LR, l2=L2)
