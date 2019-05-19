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
from torch.nn.utils.rnn import pad_sequence
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SOS_token = 0
# EOS_token = 1


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
				self.embedding = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(embedding_pre, device=device),
															  freeze=False)
			else:
				self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre, device=device),
															  freeze=False)

		else:
			self.embedding_size = config.embedding_size + 1
			self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
		# self.embedding = nn.Embedding(input_size, hidden_size)
		self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2, num_layers=2,
							  bidirectional=True, batch_first=True, dropout=self.dropout)

	def forward(self, input, hidden):
		embedded = self.embedding(input)
		# embedded_pos = torch.mul(embedded, pos_tensor.view(self.batch, -1, 1))
		# embedded = embedded.view(-1, self.batch, self.embedding_dim)  # .view(1, 1, -1)
		# x_t dim:(seq, batch, feature)
		# embedded = torch.transpose(embedded, 0, 1)
		output, hidden = self.bilstm(embedded, hidden)
		return output, hidden

	def initHidden_bilstm(self):
		# (layers*direction, batch, hidden)each  (h_0, c_0)
		return (torch.randn(4, self.batch, self.hidden_dim // 2, device=device),
				torch.randn(4, self.batch, self.hidden_dim // 2, device=device))  # if use_cuda, .cuda


class DecoderRNN(nn.Module):
	def __init__(self, config, embedding_pre, tags_num):
		super(DecoderRNN, self).__init__()
		self.batch = config.batchsize
		self.embedding_dim = config.embedding_dim
		self.hidden_dim = config.hidden_dim
		# self.tag_size = config['TAG_SIZE']
		self.pretrained = config.pretrain_vec
		self.dropout = config.dropout
		self.tag_size = tags_num + 1
		if self.pretrained:
			if torch.cuda.is_available():
				self.embedding = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(embedding_pre, device=device),
															  freeze=False)
			else:
				self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre, device=device),
															  freeze=False)
		else:
			self.embedding_size = config.embedding_size + 1
			self.embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
		self.lstm = nn.LSTM(input_size=self.embedding_dim * 2, hidden_size=self.hidden_dim, batch_first=True,
							num_layers=2, dropout=self.dropout)  # hidden_size, hidden_size)
		self.entity_embeds = nn.Embedding(self.tag_size, self.hidden_dim)
		self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)  # out
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		# output = self.embedding(input).view(-1, self.batch, self.embedding_dim)   # ??? need embedding??
		output = F.relu(input)
		output, hidden = self.lstm(output, hidden)
		output_tag = self.softmax(self.hidden2tag(output))
		return output, output_tag, hidden

	def initHidden(self):
		# (layers*direction, batch, hidden)
		return torch.zeros(2, self.batch, self.hidden_dim, device=device)


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


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
		  BATCH, TEST=False):  # max_length=MAX_LENGTH
	encoder_hidden = encoder.initHidden_bilstm()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	target_length = target_tensor.size(0)
	seq_length = target_tensor.size(1)
	# target_length = target_tensor.size(0)
	loss = 0

	# one word by one ?????
	# encoder_outputs = torch.zeros(input_length, encoder_hidden, device=device)  # max_length
	# for ei in range(input_length):
	# 	encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
	# 	encoder_outputs[ei] = encoder_output[0, 0]

	# input batch
	encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
	# input_tensor: (batch, seq); encoder_hidden: (layer*direction, batch, hidden_dim//2)
	# encoder_outputs: (batch, seq, hidden_dim//2*2); encoder_hidden: (layer*direction, batch, hidden_dim//2)

	# decoder_input = torch.tensor([[SOS_token] for i in range(BATCH)], device=device).view(BATCH, 1, 1)
	# for one-layer
	# decoder_hidden = (torch.cat((encoder_hidden[0][0], encoder_hidden[0][1]), 1).view(1, BATCH, -1),
	# 				  torch.cat((encoder_hidden[1][0], encoder_hidden[1][1]), 1).view(1, BATCH, -1))# encoder_hidden
	# for 2 layer
	h1 = torch.cat((encoder_hidden[0][0], encoder_hidden[0][1]), 1).view(1, BATCH,
																		 -1)  # concat forward and backward hidden at layer1
	h2 = torch.cat((encoder_hidden[0][2], encoder_hidden[0][3]), 1).view(1, BATCH, -1)  # layer2
	c1 = torch.cat((encoder_hidden[1][0], encoder_hidden[1][1]), 1).view(1, BATCH, -1)
	c2 = torch.cat((encoder_hidden[1][2], encoder_hidden[1][3]), 1).view(1, BATCH, -1)
	decoder_hidden = (torch.cat((h1, h2), 0),
					  torch.cat((c1, c2), 0))  # (layer*direction, batch, hidden_dim)

	decoder_input = encoder_outputs

	decoder_output, decoder_output_tag, decoder_hidden = decoder(decoder_input, decoder_hidden)
	# (batch, seq, hidden_dim)  (layer*direction, batch, hidden_dim)
	# decoder_output_T = decoder_output_tag.transpose(0, 1)  # (batch, seq, hidden_dim) -- >(seq, batch, hidden_dim)
	# target_tensor_T = target_tensor.transpose(0, 1)  # (batch, seq) --> (seq, batch)
	# for i in range(target_length):
	if not TEST:
		for j in range(target_length):  # each sentence  # seq_length
			loss += criterion(decoder_output_tag[j], target_tensor[j])

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
		loss.backward(retain_graph=True)

		encoder_optimizer.step()
		decoder_optimizer.step()

		return loss.item() / float(target_length), decoder_output_tag
	else:
		return decoder_output_tag