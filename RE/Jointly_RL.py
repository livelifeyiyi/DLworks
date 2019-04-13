# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from gensim.models import KeyedVectors

from TFgirl.RE import Optimize

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RelationModel(nn.Module):
	def __init__(self, dim, statedim, relation_count, noisy_count):
		super(RelationModel, self).__init__()
		self.dim = dim
		self.hid2state = nn.Linear(dim * 3 + statedim, statedim)
		self.state2prob_relation = nn.Linear(statedim, relation_count + 1)
		self.state2prob_noisy = nn.Linear(statedim, noisy_count + 1)
		self.att_weight = nn.Parameter(torch.randn(1, 1, self.dim))  # (self.batch, 1, self.hidden_dim)

	def attention(self, H):
		M = F.tanh(H)
		a = F.softmax(torch.bmm(self.att_weight, M), 2)
		a = torch.transpose(a, 1, 2)
		return torch.bmm(H, a)

	def forward(self, encoder_output, decoder_output, noisy_vec_mean, memory, training):

		inp = torch.cat([encoder_output, decoder_output, noisy_vec_mean, memory])
		att_out = F.tanh(self.attention(inp))   # attention
		outp = F.dropout(torch.tanh(self.hid2state(att_out)), training=training)
		prob_relation = F.softmax(self.state2prob_relation(outp), dim=0)
		prob_noisy = F.softmax(self.state2prob_noisy(outp), dim=0)
		return outp, prob_relation, prob_noisy


class RLModel(nn.Module):
	def __init__(self, sentences, encoder_output, decoder_output, dim, statedim, wv, relation_count, lr):
		self.RelationModel = RelationModel(dim, statedim, relation_count)
		self.relationvector = nn.Embedding(relation_count + 1, dim)
		self.optimizer = torch.optim.Adam(self.RelationModel.parameters(), lr=lr)
		self.sentences = sentences
		self.encoder_output = encoder_output
		self.decoder_output = decoder_output
		self.vec_model = KeyedVectors.load_word2vec_format(wv, binary=False)
		self.noisy_sentences_vec = np.array([])  # torch.from_numpy(np.array([]))
		self.decoder_hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)  # out
		self.decoder_softmax = nn.LogSoftmax(dim=1)

		# self.criterion = nn.CrossEntropyLoss()

	# get the tag with the max probability
	def sample(self, prob, training, position, preoptions=None):
		if not training:
			return torch.max(prob, 0)[1]
		elif preoptions is not None:
			return Variable(torch.cuda.LongTensor(1, ).fill_(preoptions[position]))
		else:
			return torch.multinomial(prob, 1)

	def forward(self, round_num, relation_model):
		# calculate the probability of each entity tag
		decoder_output_prob = self.decoder_softmax(self.decoder_hidden2tag(self.decoder_output))

		training = True
		if torch.cuda.is_available():
			mem = Variable(torch.cuda.FloatTensor(self.statedim, ).fill_(0))
			# action = Variable(torch.cuda.LongTensor(1, ).fill_(0))
			# rel_action = Variable(torch.cuda.LongTensor(1, ).fill_(0))
		else:
			mem = Variable(torch.FloatTensor(self.statedim, ).fill_(0))
			# action = Variable(torch.LongTensor(1, ).fill_(0))
			# rel_action = Variable(torch.LongTensor(1, ).fill_(0))
		noisy_vec_mean = np.array([])
		for sentence_id in range(len(self.sentences)):
			relation_actions, relation_actprobs, noisy_actions, noisy_actprobs = [], [], [], []
			# sentence_relations = self.sentences[sentence_id]
			for realtion in self.sentences[sentence_id]["reations"]:
				for i in range(round_num):

					mem, prob_relation, prob_noisy = relation_model(self.encoder_output[sentence_id], self.decoder_output[sentence_id],
																	torch.from_numpy(noisy_vec_mean), mem, training)
					# get relation tag
					action_realtion = self.sample(prob_relation, training, realtion)
					# get relation tag's corresponding probability
					actprob_relation = prob_relation[action_realtion]

					action_noisy = self.sample(prob_noisy, training, realtion)
					actprob_noisy = prob_noisy[action_noisy]

					relation_actions.append(action_realtion)
					relation_actprobs.append(actprob_relation)
					noisy_actions.append(action_noisy)
					noisy_actprobs.append(actprob_noisy)

				# cal reward of noisy classification ## by using the context based word vec similarity
				reward_noisy = self.calculate_similarity(realtion["rtex"], self.sentences[sentence_id]["sentext"])
				realtion["reward_noisy"] = reward_noisy
				print("Reward of noisy: " + str(reward_noisy))

				if reward_noisy < 0.3:
					self.noisy_sentences_vec = np.concatenate(self.noisy_sentences_vec, self.encoder_output)
				# vector of removed/noisy sentences
				noisy_vec_mean = np.mean(self.noisy_sentences_vec, axis=1)
				# cal reward of relation classification
				# cal total reward of relation classification and entity extraction (decoder_output)
				entity_tags, entity_probs = [], []
				for i in range(len(decoder_output_prob)):  # each word
					entity_tag = self.sample(decoder_output_prob[i], training, realtion)
					entity_prob = decoder_output_prob[i][entity_tag]
					entity_tags.append(entity_tag)
					entity_probs.append(entity_prob)
				loss = Optimize.optimize(relation_actions, relation_actprobs, realtion, entity_tags, entity_probs)
				print("Reward of jointly RE adn RL: " + str(loss))

				# optimize
				# loss = self.criterion(q_eval, q_target)  # mse 作为 loss 函数
				# 更新网络
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

	def calculate_similarity(self, relation_name, sentence):
		relation_words = relation_name.split("/")
		for i in relation_words:
			if i in sentence:
				return 1.0
		# for i in relation_name.split("/"):
		# 	if i != "":
		# 		relation_words += i.split("_")
		# print(realation_words)
		similarity_value = []
		for relation_word in relation_words:
			for word in sentence.split(" "):
				try:
					similarity_value.append(self.vec_model.similarity(relation_word, word))
				except Exception as e:
					pass
					print(e)
		if similarity_value:
			return np.mean(np.array(similarity_value))
		else:
			return 0.0

