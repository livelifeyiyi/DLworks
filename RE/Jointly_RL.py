# -*- coding: utf-8 -*-
import codecs

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
		self.state2prob_noisy = nn.Linear(statedim, noisy_count)  # + 1
		self.att_weight = nn.Parameter(torch.randn(1, 1, self.dim*2))  # (self.batch, 1, self.hidden_dim)

	def attention(self, H):  # input: (batch/1, hidden, seq); output: (batch/1, hidden, 1)
		M = torch.tanh(H)
		a = F.softmax(torch.bmm(self.att_weight, M), 2)
		a = torch.transpose(a, 1, 2)
		return torch.bmm(H, a)

	def forward(self, encoder_output, decoder_output, noisy_vec_mean, memory, training):
		seq_vec = torch.cat((encoder_output.view(1, -1, self.dim), decoder_output.view(1, -1, self.dim)), 2)
		sentence_vec = torch.tanh(self.attention(torch.transpose(seq_vec, 1, 2)))  # (1, dim*2, 1)

		inp = torch.cat((sentence_vec.view(-1), noisy_vec_mean.view(-1), memory), 0)  # (2100)

		outp = F.dropout(torch.tanh(self.hid2state(inp)), training=training)
		prob_relation = F.softmax(self.state2prob_relation(outp), dim=0)
		prob_noisy = F.softmax(self.state2prob_noisy(outp), dim=0)
		return outp, prob_relation, prob_noisy


class RLModel(nn.Module):
	def __init__(self, sentences, encoder_output, decoder_output, decoder_output_tag, dim, statedim, relation_count, lr, relation_model, wv_file):
		super(RLModel, self).__init__()
		self.statedim = statedim
		self.dim = dim
		self.RelationModel = relation_model  # RelationModel(dim, statedim, relation_count, noisy_count)
		self.relationvector = nn.Embedding(relation_count + 1, dim)
		self.optimizer = torch.optim.Adam(self.RelationModel.parameters(), lr=lr)
		self.sentences = sentences
		self.encoder_output = encoder_output
		self.decoder_output = decoder_output
		self.decoder_output_prob = decoder_output_tag
		self.vec_model = KeyedVectors.load_word2vec_format(wv_file+'vector2.txt', binary=False)  # C:\\(O_O)!\\thesis\\5-RE with LSTM\\code\\HRL-RE-use\\data\\NYT10_demo\\
		# self.vec_model = KeyedVectors.load_word2vec_format('/home/xiaoya/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
		# self.decoder_hidden2tag = nn.Linear(dim, entity_tag_size+1)  # decoder output2entity_tag
		# self.decoder_softmax = nn.LogSoftmax(dim=1)
		if torch.cuda.is_available():
			self.noisy_sentences_vec = Variable(torch.cuda.FloatTensor(1, self.dim).fill_(0))  # torch.from_numpy(np.array([]))
		else:
			self.noisy_sentences_vec = Variable(torch.FloatTensor(1, self.dim).fill_(0))  # torch.from_numpy(np.array([]))
		self.sentence_reward_noisy = [0 for i in range(len(self.sentences))]

		# self.criterion = nn.CrossEntropyLoss()
		self.att_weight = nn.Parameter(torch.randn(1, 1, self.dim))  # (self.batch, 1, self.hidden_dim)

	def attention(self, H):  # input: (batch/1, hidden, seq); output: (batch/1, hidden, 1)
		M = torch.tanh(H)
		a = F.softmax(torch.bmm(self.att_weight, M), 2)
		a = torch.transpose(a, 1, 2)
		return torch.bmm(H, a)

	# get the tag with the max probability
	def sample(self, prob, training, position=None, preoptions=None):
		if not training:
			return torch.max(prob, 1)[1]  # prob, 0
		elif preoptions is not None:
			return Variable(torch.cuda.LongTensor(1, ).fill_(preoptions[position]))
		else:
			return torch.multinomial(prob, 1)

	def forward(self, round_num, train_entity_tags, train_sentences_words, train_relation_tags, train_relation_names, seq_loss, TEST=False):
		# calculate the probability of each entity tag
		# decoder_output_prob = self.decoder_softmax(self.decoder_hidden2tag(self.decoder_output))  # (batch, seq, tag_size)

		training = True
		if torch.cuda.is_available():
			mem = Variable(torch.cuda.FloatTensor(self.statedim, ).fill_(0))
			# action = Variable(torch.cuda.LongTensor(1, ).fill_(0))
			# rel_action = Variable(torch.cuda.LongTensor(1, ).fill_(0))
		else:
			mem = Variable(torch.FloatTensor(self.statedim, ).fill_(0))
			# action = Variable(torch.LongTensor(1, ).fill_(0))
			# rel_action = Variable(torch.LongTensor(1, ).fill_(0))
		# if torch.sum(self.noisy_sentences_vec):
		noisy_vec_mean = torch.mean(self.noisy_sentences_vec, 0, True)  #
		# else:
		# 	if torch.cuda.is_available():
		# 		noisy_vec_mean = Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
		# 	else:
		# 		noisy_vec_mean = Variable(torch.FloatTensor(self.dim, ).fill_(0))
		RL_RE_loss = []
		relation_actions_batch, noisy_actions_batch, entity_actions_batch = [], [], []
		noisy_similarity_batch = []
		for sentence_id in range(len(self.sentences)):
			relation_actions, relation_actprobs, noisy_actions, noisy_actprobs = [], [], [], []
			# sentence_relations = self.sentences[sentence_id]
			# for realtion in self.sentences[sentence_id]["reations"]:
			for i in range(round_num):

				mem, prob_relation, prob_noisy = self.RelationModel(self.encoder_output[sentence_id], self.decoder_output[sentence_id],
																noisy_vec_mean, mem, training)
				# get relation tag
				action_realtion = self.sample(prob_relation, training)
				# get relation tag's corresponding probability
				actprob_relation = prob_relation[action_realtion]

				action_noisy = self.sample(prob_noisy, training)
				actprob_noisy = prob_noisy[action_noisy]

				relation_actions.append(action_realtion)
				relation_actprobs.append(actprob_relation)
				noisy_actions.append(action_noisy)
				noisy_actprobs.append(actprob_noisy)

			# cal reward of noisy classification ## by using the context based word vec similarity
			reward_noisy = self.calculate_similarity(train_relation_names[sentence_id], train_sentences_words[sentence_id])
			self.sentence_reward_noisy[sentence_id] = reward_noisy  # realtion["reward_noisy"] = reward_noisy
			print("Reward of noisy: " + str(reward_noisy))

			if reward_noisy < 0.3:
				if torch.sum(self.noisy_sentences_vec):
					self.noisy_sentences_vec = torch.cat((self.noisy_sentences_vec, self.encoder_output), 1)  # np.concatenate
				else:
					self.noisy_sentences_vec = self.attention(torch.transpose(self.encoder_output[sentence_id].view(1, -1, self.dim), 1, 2))
				# vector of removed/noisy sentences
				noisy_vec_mean = torch.mean(self.noisy_sentences_vec, 0, True)
			# cal reward of relation classification
			# cal total reward of relation classification and entity extraction (decoder_output)
			entity_actions, entity_probs = [], []
			entity_actions = self.sample(self.decoder_output_prob[sentence_id], False)
			idx = 0
			for tag in entity_actions:
				entity_probs.append(self.decoder_output_prob[sentence_id][idx][tag])
				idx += 1

			# for i in range(len(decoder_output_prob[sentence_id])):  # each word
			# 	entity_tag = self.sample(decoder_output_prob[sentence_id][i], False)
			# 	entity_prob = decoder_output_prob[sentence_id][i][entity_tag]
			# 	entity_actions.append(entity_tag)
			# 	entity_probs.append(entity_prob)
			if not TEST:
				loss = Optimize.optimize(relation_actions, relation_actprobs, train_relation_tags[sentence_id],
								train_entity_tags[sentence_id], entity_actions, entity_probs, seq_loss)
				print("Reward of jointly RE and RL: " + str(loss))
				RL_RE_loss.append(loss)
				# optimize
				# 更新网络
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			if TEST:
				relation_actions_batch.append(relation_actions)
				noisy_actions_batch.append(noisy_actions)
				entity_actions_batch.append(entity_actions)
				noisy_similarity_batch.append(reward_noisy)
		# a batch of sentences
		if TEST:
			self.cal_F_score(relation_actions_batch, train_relation_tags, train_entity_tags, entity_actions_batch,
							noisy_actions_batch, train_sentences_words, noisy_similarity_batch)
		return RL_RE_loss

	def cal_F_score(self, relation_actions_batch, train_relation_tags, train_entity_tags, entity_actions_batch, noisy_actions_batch, train_sentences_words, noisy_similarity_batch):
		batch_size = len(self.sentences)
		round_num = len(relation_actions_batch[0])
		# cal the P,R and F of relation extraction for a batch of sentences
		acc_R, cnt_R, tot_R = 0., 0., len(train_relation_tags)
		rec_R = 0.
		acc_E, cnt_E, tot_E = 0., 0., 0.  # len(train_entity_tags)

		for sentence_id in range(batch_size):
			# relation extraction
			for i in range(round_num):
				if relation_actions_batch[sentence_id][i] == train_relation_tags[sentence_id]:
					acc_R += 1
				if relation_actions_batch[sentence_id][i] > 0:
					cnt_R += 1
			if train_relation_tags[sentence_id] in relation_actions_batch[sentence_id]:
				rec_R += 1
			# entity extraction
			for word_id in range(len(entity_actions_batch[sentence_id])):
				if entity_actions_batch[sentence_id][word_id] == train_entity_tags[sentence_id][word_id]:
					acc_E += 1
				if entity_actions_batch[sentence_id][word_id] > 0:
					cnt_E += 1
				tot_E += 1
			# store the noisy action and sentence in txt file
			sentence_word = train_sentences_words[sentence_id]
			noisy_tag = noisy_actions_batch[sentence_id]
			noisy_reward = noisy_similarity_batch[sentence_id]
			line = str(noisy_tag) + ",	" + str(noisy_reward) + ",	" + sentence_word + "\n"
			with codecs.open("TEST_sentence_noisy_tag.out", mode='a+', encoding='utf-8') as f:
				f.write(line)

		precision_R = acc_R/cnt_R
		# recall = acc/round_num/tot
		recall_R = rec_R/tot_R
		beta = 1.
		F_RE = (1 + beta * beta) * precision_R * recall_R / (beta * beta * precision_R + recall_R)
		print("******TEST******: Relation precision: " + str(precision_R) + ", recall: " + str(acc_R / round_num / tot_R) + ", "
				+ str(recall_R) + ", F-score: " + str(F_RE))

		# cal the P,R and F of entity extraction for each sentence
		precision_E = acc_E / cnt_E
		recall_E = acc_E / tot_E
		F_NER = (1 + beta * beta) * precision_E * recall_E / (beta * beta * precision_E + recall_E)
		print("******TEST******: Entity precision: " + str(precision_E) + ", recall: " + str(recall_E) + ", F-score: " + str(F_NER))

		line_RE = str(precision_R) + ",	" + str(acc_R / round_num / tot_R) + ",	" + str(recall_R) + ",	" + str(F_RE) + '\n'
		with codecs.open("TEST_RE.out", mode='a+', encoding='utf-8') as f1:
			f1.write(line_RE)
		line_NER = str(precision_E) + ",	" + str(recall_E) + ",	" + str(F_NER) + '\n'
		with codecs.open("TEST_NER.out", mode='a+', encoding='utf-8') as f2:
			f2.write(line_NER)

	def calculate_similarity(self, relation_name, sentence):
		relation_words = []  # relation_name.split("/")

		for i in relation_name.split("/"):
			if i != "":
				relation_words += i.split("_")
		for i in relation_words:
			if i and i in sentence:
				return 1.0
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

