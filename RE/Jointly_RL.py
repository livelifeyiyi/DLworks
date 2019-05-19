# -*- coding: utf-8 -*-
import codecs

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np


from Optimize import Optimize
Optimize = Optimize()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RelationModel(nn.Module):
	def __init__(self, args, dim, statedim, relation_count, noisy_count):
		super(RelationModel, self).__init__()
		self.dropout = args.dropout
		self.batch = args.batchsize
		self.dim = dim
		self.statedim = statedim
		self.tag_size = relation_count + 1
		self.hid2state = nn.Linear(dim * 2 + dim, dim)  # statedim
		self.hid2state_noisy = nn.Linear(dim * 3 + dim, dim)
		self.state2prob_relation = nn.Linear(dim, self.tag_size)  # + 1
		self.state2prob_noisy = nn.Linear(dim, noisy_count)  # + 1
		self.att_weight = nn.Parameter(torch.randn(1, 1, self.dim*2))  # (self.batch, 1, self.hidden_dim)
		self.lstm = nn.LSTM(input_size=dim * 2 + dim, hidden_size=dim, num_layers=2, dropout=self.dropout)  # batch_first=True,hidden_size, hidden_size)

		self.relation_bias = nn.Parameter(torch.randn(1, self.tag_size, 1))  # self.batch, self.tag_size, 1

		self.dropout_lstm = nn.Dropout(p=0.5)
		self.dropout_att = nn.Dropout(p=0.5)
		self.relation_embeds = nn.Embedding(self.tag_size, self.dim)

		# if torch.cuda.is_available():
		# 	self.noisy_sentences_vec = Variable(
		# 		torch.cuda.FloatTensor(1, self.dim).fill_(0))  # torch.from_numpy(np.array([]))
		# else:
		# 	self.noisy_sentences_vec = Variable(
		# 		torch.FloatTensor(1, self.dim).fill_(0))  # torch.from_numpy(np.array([]))

	def attention(self, H):  # input: (batch/1, hidden, seq); output: (batch/1, hidden, 1)
		M = torch.tanh(H)
		a = F.softmax(torch.bmm(self.att_weight, M), 2)
		a = torch.transpose(a, 1, 2)
		return torch.bmm(H, a)

	def initHidden(self):
		# (layers*direction, batch, hidden)
		return (torch.randn(2, 1, self.dim, device=device),
				torch.randn(2, 1, self.dim, device=device))

	def forward(self, hidden, encoder_output, decoder_output, noisy_vec_mean, memory, training):
		if torch.cuda.is_available():
			encoder_output = encoder_output.cuda()  # Variable(torch.cuda.LongTensor(encoder_output, device=device)).cuda()
			decoder_output = decoder_output.cuda()  # Variable(torch.cuda.LongTensor(decoder_output, device=device)).cuda()
			memory = memory.cuda()  # Variable(torch.cuda.LongTensor(memory, device=device)).cuda()
			noisy_vec_mean = noisy_vec_mean.cuda()

		seq_vec = torch.cat((encoder_output.view(1, -1, self.dim), decoder_output.view(1, -1, self.dim)), 2)
		sentence_vec = torch.tanh(self.attention(torch.transpose(seq_vec, 1, 2)))  # (1, dim*2, 1)

		inp_noisy = torch.cat((sentence_vec.view(-1), noisy_vec_mean.view(-1), memory), 0)  # (2400)
		outp_noisy = F.dropout(torch.tanh(self.hid2state_noisy(inp_noisy)), training=training)
		prob_noisy = F.softmax(self.state2prob_noisy(outp_noisy), dim=0)

		inp = torch.cat((sentence_vec.view(-1), memory), 0)  # (2100-300)
		'''outp = F.dropout(torch.tanh(self.hid2state(inp)), training=training)
		prob_relation = self.softmax(self.state2prob_relation(outp))'''

		output = F.relu(inp)
		lstm_out, hidden = self.lstm(output.view(1,1,-1), hidden)
		lstm_out = self.dropout_lstm(lstm_out)
		# att_out = torch.tanh(self.attention(lstm_out.view(self.batch, self.hidden_dim, -1)))
		# att_out = self.dropout_att(att_out)
		relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(1, 1)  # (batch, 1)
		if torch.cuda.is_available():
			relation = relation.cuda()
		relation = self.relation_embeds(relation)
		res = torch.add(torch.bmm(relation, torch.transpose(lstm_out, 1, 2)), self.relation_bias)
		res = F.softmax(res, 1)
		# prob_relation = F.softmax(self.state2prob_relation(outp.view(-1)), dim=0)  # self.softmax(self.state2prob_relation(outp.view(-1)))

		return lstm_out.view(-1), res.view(-1), prob_noisy


class RLModel(nn.Module):
	def __init__(self, batchsize, dim, statedim, relation_count, vec_model):
		super(RLModel, self).__init__()
		self.statedim = statedim
		self.dim = dim
		self.batch_size = batchsize
		self.relationvector = nn.Embedding(relation_count + 1, dim)
		self.vec_model = vec_model  # KeyedVectors.load_word2vec_format(wv_file+'vector2.txt', binary=False)
		# self.RelationModel = relation_model  # RelationModel(dim, statedim, relation_count, noisy_count)
		# self.optimizer = torch.optim.Adam(self.RelationModel.parameters(), lr=lr)
		# self.sentences = sentences
		# self.encoder_output = encoder_output
		# self.decoder_output = decoder_output
		# self.decoder_output_prob = decoder_output_prob
		# self.decoder_hidden2tag = nn.Linear(dim, entity_tag_size+1)  # decoder output2entity_tag
		# self.decoder_softmax = nn.LogSoftmax(dim=1)
		# self.sentence_reward_noisy = [0 for i in range(self.batch_size)]
		# self.criterion = nn.CrossEntropyLoss()
		self.att_weight = nn.Parameter(torch.randn(1, 1, self.dim, device=device))  # (self.batch, 1, self.hidden_dim)
		self.acc = 0.
		self.cnt = 0.
		self.tot = 0.

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
			return torch.multinomial(prob, 1)  # torch.max(prob, 0)[1]  #

	# sentences, encoder_output, decoder_output, decoder_output_prob  # for RE optimize: relation_target_tensor, criterion
	def forward(self, sentences, encoder_output, decoder_output, decoder_output_prob, decoder_hidden, sentence_reward_noisy, noisy_sentences_vec,
				RE_optimizer, RL_optimizer, relation_model, round_num, train_entity_tags, train_sentences_words, train_relation_tags,
				train_relation_names, relation_target_tensor, criterion, seq_loss, flag="TRAIN"):
		# calculate the probability of each entity tag
		# decoder_output_prob = self.decoder_softmax(self.decoder_hidden2tag(self.decoder_output))  # (batch, seq, tag_size)

		training = True
		if torch.cuda.is_available():
			mem = Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
			# action = Variable(torch.cuda.LongTensor(1, ).fill_(0))
			# rel_action = Variable(torch.cuda.LongTensor(1, ).fill_(0))
		else:
			mem = Variable(torch.FloatTensor(self.dim, ).fill_(0))
			# action = Variable(torch.LongTensor(1, ).fill_(0))
			# rel_action = Variable(torch.LongTensor(1, ).fill_(0))
		# if torch.sum(self.noisy_sentences_vec):
		noisy_vec_mean = torch.mean(noisy_sentences_vec, 0, True)  #
		# else:
		# 	if torch.cuda.is_available():
		# 		noisy_vec_mean = Variable(torch.cuda.FloatTensor(self.dim, ).fill_(0))
		# 	else:
		# 		noisy_vec_mean = Variable(torch.FloatTensor(self.dim, ).fill_(0))
		RL_RE_loss = []
		RE_rewards = []
		TOTAL_rewards = []
		relation_actions_batch, noisy_actions_batch, entity_actions_batch = [], [], []
		noisy_similarity_batch = []
		# hidden = relation_model.initHidden()
		print_loss_total = 0.
		# criterion = nn.NLLLoss()  # CrossEntropyLoss()
		# if torch.cuda.is_available():
		# 	criterion = criterion.cuda()
		loss_RL = 0.
		loss_RE = 0.
		pred_prob_relations_batch = []
		hidden_transp_1 = torch.transpose(decoder_hidden[0], 0, 1)  # (batch, 2, dim)
		hidden_transp_2 = torch.transpose(decoder_hidden[1], 0, 1)  # (batch, 2, dim)
		for sentence_id in range(len(sentences)):
			inp_hidden = (torch.transpose(hidden_transp_1[sentence_id].view(1,2,-1), 0, 1).contiguous(),
						torch.transpose(hidden_transp_2[sentence_id].view(1,2,-1), 0, 1).contiguous())
			relation_actions, relation_actprobs, noisy_actions, noisy_actprobs = [], [], [], []
			pred_prob_relations = []
			# entity_actions = []
			entity_actions = self.sample(decoder_output_prob[sentence_id], False)
			#  ######
			#  entity_probs = []
			# idx = 0
			# for tag in entity_actions:
			# 	entity_probs.append(self.decoder_output_prob[sentence_id][idx][tag])
			# 	idx += 1
			#  ######

			# for i in range(len(decoder_output_prob[sentence_id])):  # each word
			# 	entity_tag = self.sample(decoder_output_prob[sentence_id][i], False)
			# 	entity_prob = decoder_output_prob[sentence_id][i][entity_tag]
			# 	entity_actions.append(entity_tag)
			# 	entity_probs.append(entity_prob)

			# sentence_relations = self.sentences[sentence_id]
			# for realtion in self.sentences[sentence_id]["reations"]:
			for i in range(round_num):

				mem, prob_relation, prob_noisy = relation_model(inp_hidden,
																encoder_output[sentence_id], decoder_output[sentence_id],
																noisy_vec_mean, mem, training)
				# get relation tag
				action_realtion = torch.multinomial(prob_relation, 1)  # self.sample(prob_relation, training)
				# get relation tag's corresponding probability
				actprob_relation = prob_relation[action_realtion]

				action_noisy = self.sample(prob_noisy, training)
				# actprob_noisy = prob_noisy[action_noisy]

				relation_actions.append(action_realtion.item())  # .item()
				relation_actprobs.append(actprob_relation.item())
				noisy_actions.append(action_noisy.item())
				pred_prob_relations.append(prob_relation.cpu().detach().numpy())
				# noisy_actprobs.append(actprob_noisy)
				if (action_realtion.item() in train_relation_tags[sentence_id] and len(set(train_relation_tags[sentence_id])) == 1) \
					or (set(relation_actions) > set(train_relation_tags[sentence_id])):
					# if set(relation_actions) > set(train_relation_tags[sentence_id]):
					break

			if flag == "TRAIN":
				RL_optimizer.zero_grad()
				# loss_RL = Optimize.optimize(relation_actions, relation_actprobs, train_relation_tags[sentence_id], seq_loss)
				loss_RL, relation_reward, total_reward = Optimize.optimize_each(relation_actions, relation_actprobs, train_relation_tags[sentence_id], train_entity_tags[sentence_id], entity_actions)
				# if loss_RL == 0.:
				# 	loss_RL = loss_RL_tmp
				# else:
				# 	loss_RL += loss_RL_tmp
				#
				loss_RL.backward(retain_graph=True)
				RL_optimizer.step()
			if loss_RL != 0.:
				RL_RE_loss.append(loss_RL.item())
				RE_rewards.append(relation_reward)
				TOTAL_rewards.append(total_reward)
			# cal reward of relation classification
			# cal total reward of relation classification and entity extraction (decoder_output)

			'''if flag == "TRAIN":
				loss_RL = Optimize.optimize(relation_actions, relation_actprobs, train_relation_tags[sentence_id], seq_loss)
				# train_entity_tags[sentence_id], entity_actions, entity_probs,   [action_realtion.item()], [actprob_relation.item()]
				# print("Reward of jointly RE and RL: " + str(loss))
				# print_every = 50
				# print_loss_total += loss.item()
				# if sentence_id % print_every == 0:
				# 	print_loss_avg = print_loss_total / float(print_every)  # *sentence_id//print_every)
				# 	print_loss_total = 0.
				# 	print('Jointly RE and RL: (%d %.2f%%), loss reward: %.4f' % (sentence_id, float(sentence_id) / len(self.sentences) * 100, print_loss_avg))
				# 

				# optimize

				# loss = criterion(action_realtion.view(1, -1), torch.tensor(train_relation_tags[sentence_id]).view(1, -1))
				# 更新网络
				self.optimizer.zero_grad()
				loss_RL.backward(retain_graph=True)
				self.optimizer.step()
			if loss_RL != 0.:
				RL_RE_loss.append(loss_RL.item())'''

			# cal reward of noisy classification ## by using the context based word vec similarity
			reward_noisy = self.calculate_similarity(train_relation_names[sentence_id], train_sentences_words[sentence_id])
			sentence_reward_noisy[sentence_id] = reward_noisy  # realtion["reward_noisy"] = reward_noisy

			if reward_noisy < 0.2:
				# print("Reward of noisy: " + str(reward_noisy))
				sentence_vec = self.attention(torch.transpose(encoder_output[sentence_id].view(1, -1, self.dim), 1, 2))
				if torch.sum(noisy_sentences_vec):
					noisy_sentences_vec = torch.cat((noisy_sentences_vec, sentence_vec), 0)  # np.concatenate
				else:
					noisy_sentences_vec = sentence_vec
				# vector of removed/noisy sentences
				noisy_vec_mean = torch.mean(noisy_sentences_vec, 0, True)

			# if TEST:
			relation_actions_batch.append(relation_actions)
			noisy_actions_batch.append(noisy_actions)
			entity_actions_batch.append(entity_actions)
			noisy_similarity_batch.append(reward_noisy)
			pred_prob_relations_batch.append(np.mean(np.array(pred_prob_relations), 0))
			torch.cuda.empty_cache()
		if flag == "TRAIN":
			pred_target = torch.from_numpy(np.array(pred_prob_relations_batch))
			if torch.cuda.is_available():
				pred_target = pred_target.cuda()
				pred_target = torch.autograd.Variable(pred_target, requires_grad=True)
			else:
				pred_target = torch.autograd.Variable(pred_target, requires_grad=True)
			RE_optimizer.zero_grad()
			for i in range(len(relation_target_tensor[0])):
				target = torch.transpose(relation_target_tensor, 0, 1)[i]
				loss_RE += criterion(pred_target, target)

			loss_RE.backward(retain_graph=True)
			RE_optimizer.step()
		# 	self.optimizer.zero_grad()
		# 	loss_RL.backward(retain_graph=True)
		# 	self.optimizer.step()
		if flag == "TRAIN":
			print("Reward of jointly RE and RL: " + str(np.average(np.array(RL_RE_loss))) + ", " + str(loss_RE.item()))
		# a batch of sentences

		self.cal_F_score(relation_actions_batch, train_relation_tags, train_relation_names, train_entity_tags, entity_actions_batch,
							noisy_actions_batch, train_sentences_words, noisy_similarity_batch, flag)

		return RL_RE_loss, RE_rewards, TOTAL_rewards

	def calc_acc_total(self, relation_action, entity_action, relation_labels, entity_labels):
		# acc, cnt, tot = 0, 0, len(relation_labels)
		self.tot += len(relation_labels)
		cnt = 0.
		# used = [0 for i in range(len(relation_action))]
		# tp, tags = label, label['tags']
		j, ok = 0, 0
		# for i in range(len(relation_action)):  # each round
		if isinstance(relation_action, np.int64):
			for label in relation_labels:
				if label == relation_action and ok == 0 and label > 0:  # relation_action[i] == label and used[i] == 0
					match = 1
					for k in range(len(entity_labels)):
						if entity_labels[k] == 4 and entity_action[k] != 4:
							match = 0
						if entity_labels[k] != 4 and entity_action[k] == 4:
							match = 0
						if entity_labels[k] == 5 and entity_action[k] != 5:
							match = 0
						if entity_labels[k] != 5 and entity_action[k] == 5:
							match = 0
					if match == 1:
						ok = 1
				self.acc += ok
			if relation_action > 0:
				cnt += 1
			self.cnt += cnt
		else:
			for label in relation_labels:
				if label in relation_action and ok == 0 and label > 0:  # relation_action[i] == label and used[i] == 0
					match = 1
					for k in range(len(entity_labels)):
						if entity_labels[k] == 4 and entity_action[k] != 4:
							match = 0
						if entity_labels[k] != 4 and entity_action[k] == 4:
							match = 0
						if entity_labels[k] == 5 and entity_action[k] != 5:
							match = 0
						if entity_labels[k] != 5 and entity_action[k] == 5:
							match = 0
					if match == 1:
						ok = 1
				self.acc += ok
				# used[i] = 1
			for i in range(len(relation_action)):
				if relation_action[i] > 0:
					# j += 1
					cnt += 1

			self.cnt += cnt // len(relation_labels)
		return self.acc, self.tot, self.cnt

	def cal_F_score(self, relation_actions_batch, train_relation_tags, train_relation_names,  train_entity_tags, entity_actions_batch, noisy_actions_batch, train_sentences_words, noisy_similarity_batch, flag):
		batch_size = self.batch_size  # len(relation_actions_batch)
		# cal the P,R and F of relation extraction for a batch of sentences
		# acc_total, tot_total, cnt_total = 0., 0., 0.
		tot_R_relation_num = 0.
		acc_R, cnt_R, tot_R = 0., 0., 0.  # len(train_relation_tags)
		# acc_R_last, cnt_R_last, tot_R_last = 0., 0., 0.
		cnt_R_last = 0.
		rec_R = 0.
		acc_E, cnt_E, tot_E = 0., 0., 0.  # len(train_entity_tags)
		acc_E_no0 = 0.
		for sentence_id in range(batch_size):
			relation_tag = list(set(train_relation_tags[sentence_id]))
			if isinstance(relation_actions_batch[sentence_id], np.int64):
				round_num = 1
			else:
				round_num = len(relation_actions_batch[sentence_id])
			acc_total, tot_total, cnt_total = self.calc_acc_total(relation_actions_batch[sentence_id],
																  entity_actions_batch[sentence_id],
																  relation_tag,
																  train_entity_tags[sentence_id])

			if isinstance(relation_actions_batch[sentence_id], np.int64):
				if relation_actions_batch[sentence_id] in relation_tag:
					acc_R += 1
				if relation_actions_batch[sentence_id] > 0:
					cnt_R += 1
				cnt_R_last += round_num
			else:
				for i in range(round_num):
					if int(relation_actions_batch[sentence_id][i]) in relation_tag:
						acc_R += 1
					if int(relation_actions_batch[sentence_id][i]) > 0:
						cnt_R += 1
					# tot_R += 1
				cnt_R_last += round_num // len(relation_tag)
			tot_R_relation_num += len(relation_tag)
			for each_relation in relation_tag:
				if each_relation > 0:
					# tot_R += 1
					if isinstance(relation_actions_batch[sentence_id], np.int64):
						if each_relation == relation_actions_batch[sentence_id]:
							rec_R += 1
					else:
						if each_relation == relation_actions_batch[sentence_id] or each_relation in relation_actions_batch[sentence_id]:
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
			with codecs.open("./"+flag+"_sentence_noisy_tag.out", mode='a+', encoding='utf-8') as f:
				f.write(line)

		precision_total = acc_total / cnt_total
		recall_total = acc_total / tot_total
		beta = 1.
		try:
			F_total = (1 + beta * beta) * precision_total * recall_total / (beta * beta * precision_total + recall_total)
		except Exception as e:
			print(e)
			F_total = 0.
		print("********: TOTAL precision: " + str(precision_total) + ", recall: " + str(recall_total) + ", F-score: " + str(F_total))
		line_total = str(precision_total) + ", " + str(recall_total) + ", " + str(F_total) + "\n"
		with codecs.open("./"+flag+"_TOTAL.out", mode='a+', encoding='utf-8') as f1:
			f1.write(line_total)

		if cnt_R != 0 and tot_R_relation_num != 0:
			precision_R = acc_R / cnt_R
			# recall = acc/round_num/tot
			recall_R = rec_R / tot_R_relation_num
		else:
			precision_R = 0
			recall_R = 0
		beta = 1.
		try:
			F_RE = (1 + beta * beta) * precision_R * recall_R / (beta * beta * precision_R + recall_R)
		except Exception as e:
			print(e)
			F_RE = 0.
		print("********: Relation precision: " + str(acc_R / tot_R_relation_num) + ", " + str(
			acc_R / cnt_R_last) + ", " + str(precision_R) +
			  ", recall: " + str(recall_R) + ", F-score: " + str(F_RE))

		line_RE = str(acc_R / tot_R_relation_num) + ", " + str(acc_R / cnt_R_last) + ", " + str(
			precision_R) + ",	" + str(recall_R) + ",	" + str(F_RE) + '\n'
		with codecs.open("./"+flag+"_RE.out", mode='a+', encoding='utf-8') as f1:
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
		with codecs.open("./"+flag+"_NER.out", mode='a+', encoding='utf-8') as f2:
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

