# -*- coding: utf-8 -*-
import codecs

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np


import Optimize

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
		if torch.cuda.is_available():
			encoder_output = encoder_output.cuda()  # Variable(torch.cuda.LongTensor(encoder_output, device=device)).cuda()
			decoder_output = decoder_output.cuda()  # Variable(torch.cuda.LongTensor(decoder_output, device=device)).cuda()
			memory = memory.cuda()  # Variable(torch.cuda.LongTensor(memory, device=device)).cuda()

		seq_vec = torch.cat((encoder_output.view(1, -1, self.dim), decoder_output.view(1, -1, self.dim)), 2)
		sentence_vec = torch.tanh(self.attention(torch.transpose(seq_vec, 1, 2)))  # (1, dim*2, 1)

		inp = torch.cat((sentence_vec.view(-1), noisy_vec_mean.view(-1), memory), 0)  # (2100)

		outp = F.dropout(torch.tanh(self.hid2state(inp)), training=training)
		prob_relation = F.softmax(self.state2prob_relation(outp), dim=0)
		prob_noisy = F.softmax(self.state2prob_noisy(outp), dim=0)
		return outp, prob_relation, prob_noisy


class RLModel(nn.Module):
	def __init__(self, sentences, encoder_output, decoder_output, decoder_output_prob, batchsize, dim, statedim, relation_count, lr, relation_model, vec_model):
		super(RLModel, self).__init__()
		self.statedim = statedim
		self.dim = dim
		self.batch_size = batchsize
		self.RelationModel = relation_model  # RelationModel(dim, statedim, relation_count, noisy_count)
		self.relationvector = nn.Embedding(relation_count + 1, dim)
		self.optimizer = torch.optim.Adam(self.RelationModel.parameters(), lr=lr)
		self.sentences = sentences
		self.encoder_output = encoder_output
		self.decoder_output = decoder_output
		self.decoder_output_prob = decoder_output_prob
		self.vec_model = vec_model  # KeyedVectors.load_word2vec_format(wv_file+'vector2.txt', binary=False)
		# self.decoder_hidden2tag = nn.Linear(dim, entity_tag_size+1)  # decoder output2entity_tag
		# self.decoder_softmax = nn.LogSoftmax(dim=1)
		if torch.cuda.is_available():
			self.noisy_sentences_vec = Variable(torch.cuda.FloatTensor(1, self.dim).fill_(0))  # torch.from_numpy(np.array([]))
		else:
			self.noisy_sentences_vec = Variable(torch.FloatTensor(1, self.dim).fill_(0))  # torch.from_numpy(np.array([]))
		self.sentence_reward_noisy = [0 for i in range(self.batch_size)]

		# self.criterion = nn.CrossEntropyLoss()
		self.att_weight = nn.Parameter(torch.randn(1, 1, self.dim, device=device))  # (self.batch, 1, self.hidden_dim)

	def attention(self, H):  # input: (batch/1, hidden, seq); output: (batch/1, hidden, 1)
		M = torch.tanh(H)
		a = F.softmax(torch.bmm(self.att_weight, M), 2)
		a = torch.transpose(a, 1, 2)
		return torch.bmm(H, a)

	# get the action tag with the max probability
	def sample(self, prob, training, position=None, preoptions=None):
		if not training:
			return torch.max(prob, 1)[1]  # prob, 0
		elif preoptions is not None:
			return Variable(torch.cuda.LongTensor(1, ).fill_(preoptions[position]))
		else:
			return torch.multinomial(prob, 1)

	# , sentences, encoder_output, decoder_output, decoder_output_prob
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
		print_loss_total = 0.
		criterion = nn.NLLLoss()  # CrossEntropyLoss()
		loss = 0.
		for sentence_id in range(len(self.sentences)):
			relation_actions, relation_actprobs, noisy_actions, noisy_actprobs = [], [], [], []

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
				# actprob_noisy = prob_noisy[action_noisy]

				relation_actions.append(action_realtion.item())  # .item()
				relation_actprobs.append(actprob_relation.item())
				noisy_actions.append(action_noisy.item())
				# noisy_actprobs.append(actprob_noisy)


				# cal reward of relation classification
				# cal total reward of relation classification and entity extraction (decoder_output)

				if not TEST:
					loss = Optimize.optimize([action_realtion.item()], [actprob_relation.item()], train_relation_tags[sentence_id],
											 train_entity_tags[sentence_id], entity_actions, entity_probs, seq_loss)
					# print("Reward of jointly RE and RL: " + str(loss))
					'''print_every = 50
					print_loss_total += loss.item()
					if sentence_id % print_every == 0:
						print_loss_avg = print_loss_total / float(print_every)  # *sentence_id//print_every)
						print_loss_total = 0.
						print('Jointly RE and RL: (%d %.2f%%), loss reward: %.4f' % (sentence_id, float(sentence_id) / len(self.sentences) * 100, print_loss_avg))
					'''

					# optimize
					# for j in range(len(relation_actions)):  # each sentence  # seq_length
					# 	loss += criterion(relation_actions[j], train_relation_tags[sentence_id])
					# 更新网络
					self.optimizer.zero_grad()
					loss.backward(retain_graph=True)
					self.optimizer.step()
			RL_RE_loss.append(loss.item())
			# cal reward of noisy classification ## by using the context based word vec similarity
			reward_noisy = self.calculate_similarity(train_relation_names[sentence_id], train_sentences_words[sentence_id])
			self.sentence_reward_noisy[sentence_id] = reward_noisy  # realtion["reward_noisy"] = reward_noisy

			if reward_noisy < 0.2:
				# print("Reward of noisy: " + str(reward_noisy))
				sentence_vec = self.attention(torch.transpose(self.encoder_output[sentence_id].view(1, -1, self.dim), 1, 2))
				if torch.sum(self.noisy_sentences_vec):
					self.noisy_sentences_vec = torch.cat((self.noisy_sentences_vec, sentence_vec), 0)  # np.concatenate
				else:
					self.noisy_sentences_vec = sentence_vec
				# vector of removed/noisy sentences
				noisy_vec_mean = torch.mean(self.noisy_sentences_vec, 0, True)


			# if TEST:
			relation_actions_batch.append(relation_actions)
			noisy_actions_batch.append(noisy_actions)
			entity_actions_batch.append(entity_actions)
			noisy_similarity_batch.append(reward_noisy)
			torch.cuda.empty_cache()
		print("Reward of jointly RE and RL: " + str(np.average(np.array(RL_RE_loss))))
		# a batch of sentences
		if not TEST:
			self.cal_F_score(relation_actions_batch, train_relation_tags, train_relation_names, train_entity_tags, entity_actions_batch,
							noisy_actions_batch, train_sentences_words, noisy_similarity_batch, "TRAIN")
		if TEST:
			self.cal_F_score(relation_actions_batch, train_relation_tags, train_relation_names, train_entity_tags, entity_actions_batch,
							noisy_actions_batch, train_sentences_words, noisy_similarity_batch, "TEST")

		return RL_RE_loss

	def cal_F_score(self, relation_actions_batch, train_relation_tags, train_relation_names,  train_entity_tags, entity_actions_batch, noisy_actions_batch, train_sentences_words, noisy_similarity_batch, flag):
		batch_size = self.batch_size  # len(relation_actions_batch)
		round_num = len(relation_actions_batch[0])
		# cal the P,R and F of relation extraction for a batch of sentences
		acc_R, cnt_R, tot_R = 0., 0., len(train_relation_tags)
		acc_R_last, cnt_R_last = 0., 0.
		rec_R = 0.
		acc_E, cnt_E, tot_E = 0., 0., 0.  # len(train_entity_tags)
		acc_E_no0 = 0.
		for sentence_id in range(batch_size):
			# relation extraction
			if int(relation_actions_batch[sentence_id][-1]) == train_relation_tags[sentence_id]:
				acc_R_last += 1
			if int(relation_actions_batch[sentence_id][-1]) > 0:
				cnt_R_last += 1
			for i in range(round_num):
				if int(relation_actions_batch[sentence_id][i]) == train_relation_tags[sentence_id]:
					acc_R += 1
				if int(relation_actions_batch[sentence_id][i]) > 0:
					cnt_R += 1
			if train_relation_tags[sentence_id] in relation_actions_batch[sentence_id]:
				rec_R += 1
			# entity extraction
			for word_id in range(len(train_entity_tags[sentence_id])):
				if int(entity_actions_batch[sentence_id][word_id]) == train_entity_tags[sentence_id][word_id]:
					acc_E += 1
				if train_entity_tags[sentence_id][word_id] > 0:
					cnt_E += 1
					if int(entity_actions_batch[sentence_id][word_id]) == train_entity_tags[sentence_id][word_id]:
						acc_E_no0 += 1
				tot_E += 1
			# store the noisy action and sentence in txt file
			sentence_word = train_sentences_words[sentence_id]
			noisy_tag = noisy_actions_batch[sentence_id]
			noisy_reward = noisy_similarity_batch[sentence_id]
			line = str(noisy_tag) + ",	" + str(noisy_reward) + ",	" + str(train_relation_names[sentence_id]) + ',	' + \
					str(train_entity_tags[sentence_id]) + ",	" + sentence_word + "\n"
			with codecs.open(flag+"_sentence_noisy_tag.out", mode='a+', encoding='utf-8') as f:
				f.write(line)

		precision_R = acc_R/cnt_R
		# recall = acc/round_num/tot
		recall_R = rec_R/tot_R
		beta = 1.
		try:
			F_RE = (1 + beta * beta) * precision_R * recall_R / (beta * beta * precision_R + recall_R)
		except Exception as e:
			print(e)
			F_RE = 0.
		print("********: Relation precision: " + str(acc_R_last/cnt_R_last) + ", " + str(precision_R) +
				", recall: " + str(acc_R_last/tot_R) + ", " + str(acc_R / round_num / tot_R) + ", "
				+ str(recall_R) + ", F-score: " + str(F_RE))

		line_RE = str(acc_R_last/cnt_R_last) + ", " + str(precision_R) + ",	" + str(acc_R_last/tot_R) + ", " + str(acc_R / round_num / tot_R) + ",	" + str(recall_R) + ",	" + str(
			F_RE) + '\n'
		with codecs.open(flag+"_RE.out", mode='a+', encoding='utf-8') as f1:
			f1.write(line_RE)

		# cal the P,R and F of entity extraction for each sentence
		precision_E = acc_E / tot_E
		recall_E = acc_E_no0 / cnt_E  # acc_E / tot_E
		try:
			F_NER = (1 + beta * beta) * precision_E * recall_E / (beta * beta * precision_E + recall_E)
		except Exception as e:
			print(e)
			F_NER = 0.
		print("********: Entity precision: " + str(precision_E) + ", recall: " + str(recall_E) + ", F-score: " + str(F_NER))

		line_NER = str(precision_E) + ",	" + str(recall_E) + ",	" + str(F_NER) + '\n'
		with codecs.open(flag+"_NER.out", mode='a+', encoding='utf-8') as f2:
			f2.write(line_NER)

	def calculate_similarity(self, relation_name, sentence):
		relation_words = []  # relation_name.split("/")

		for i in relation_name.split("/"):
			if i != "":
				relation_words += i.split("_")
		for i in relation_words:
			if i != "of" and i in sentence:
				return 1.0
		# print(realation_words)
		similarity_value = []
		for relation_word in relation_words:
			for word in sentence.split(" "):
				if word != '' and word != ',':
					try:
						similarity_value.append(self.vec_model.similarity(relation_word, word))
					except Exception as e:
						pass
						# print(e)
		if similarity_value:
			return np.mean(np.array(similarity_value))
		else:
			return 0.0

