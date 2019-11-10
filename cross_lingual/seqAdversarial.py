"""
build_sequence_adversarial_model
class SeqAdversarialTrainer
"""
import os
import numpy as np
import scipy
import scipy.linalg

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from layers import Embedding, CharCnnWordEmb, EncodeLstm, CnnDiscriminator, LinearProj, CRFLoss
from loader import load_pretrained


def build_sequence_adversarial_model(params, mappings):
	# construct word embedding layers
	target_word_embedding = Embedding(params['word_vocab_size'], params['word_dim'])
	related_word_embedding = Embedding(params['bi_word_vocab_size'], params['word_dim'])
	load_pretrained(target_word_embedding.emb, mappings['id_to_word'], params['target_emb'])
	load_pretrained(related_word_embedding.emb, mappings['bi_id_to_word'], params['related_emb'])

	# char embedding layer
	char_embedding = Embedding(params['char_vocab_size'], params['char_dim'])

	# CNN and concatenate with word for target language
	target_char_cnn_word = CharCnnWordEmb(params['word_dim'], params['char_dim'], params['char_conv'],
										  params['max_word_length'], params['filter_withs'])

	# CNN and concatenate with word for related language
	related_char_cnn_word = CharCnnWordEmb(params['word_dim'], params['char_dim'], params['char_conv'],
										   params['max_word_length'], params['filter_withs'])

	# sequence encoder
	adv_lstm = EncodeLstm(params['char_cnn_word_dim'], params['char_cnn_word_dim'], bidrection=True,
						  dropout=params['dropout'])

	# sequence discriminator
	seq_discriminator = CnnDiscriminator(params['char_cnn_word_dim']*2, params['word_lstm_dim'], [2, 3], 1)

	# context encoder
	context_lstm = EncodeLstm(params['char_cnn_word_dim']*2, params['word_lstm_dim'],
							  dropout=params['dropout'])
	# linear projection
	linear_proj = LinearProj(params['word_lstm_dim'] * 2, params['word_lstm_dim'], params['label_size'])

	tagger_criterion = CRFLoss(params['label_size'])

	dis_criterion = nn.NLLLoss()

	if params['gpu']:
		target_word_embedding = target_word_embedding.cuda()
		related_word_embedding = related_word_embedding.cuda()
		char_embedding = char_embedding.cuda()
		target_char_cnn_word = target_char_cnn_word.cuda()
		related_char_cnn_word = related_char_cnn_word.cuda()
		adv_lstm = adv_lstm.cuda()
		seq_discriminator = seq_discriminator.cuda()
		context_lstm = context_lstm.cuda()
		linear_proj = linear_proj.cuda()
		tagger_criterion = tagger_criterion.cuda()
		dis_criterion = dis_criterion.cuda()

	return target_word_embedding, related_word_embedding, char_embedding, target_char_cnn_word, \
		   related_char_cnn_word, adv_lstm, seq_discriminator, context_lstm, linear_proj, \
		   tagger_criterion, dis_criterion


class SeqAdversarialTrainer(object):
	def __init__(self, target_word_embedding, related_word_embedding, embedding_mapping, char_embedding,
				 target_char_cnn_word, related_char_cnn_word, adv_lstm, seq_discriminator, context_lstm,
				 linear_proj, tagger_criterion, dis_criterion, params):
		"""
		Initialize trainer script.
		"""
		self.target_word_embedding = target_word_embedding
		self.related_word_embedding = related_word_embedding
		self.embedding_mapping = embedding_mapping
		self.char_embedding = char_embedding
		self.target_char_cnn_word = target_char_cnn_word
		self.related_char_cnn_word = related_char_cnn_word
		self.adv_lstm = adv_lstm
		self.seq_discriminator = seq_discriminator
		self.context_lstm = context_lstm
		self.linear_proj = linear_proj
		self.tagger_criterion = tagger_criterion
		self.dis_criterion = dis_criterion
		self.params = params

		feature_parameters = []
		feature_parameters += char_embedding.parameters()
		feature_parameters += target_char_cnn_word.parameters()
		feature_parameters += related_char_cnn_word.parameters()
		feature_parameters += adv_lstm.parameters()
		feature_optim = optim.SGD
		if params['tagger_optimizer'] == 'adadelta':
			feature_optim = optim.Adadelta
		elif params['tagger_optimizer'] == 'adagrad':
			feature_optim = optim.Adagrad
		elif params['tagger_optimizer'] == 'adam':
			feature_optim = optim.Adam
		elif params['tagger_optimizer'] == 'adamax':
			feature_optim = optim.Adamax
		elif params['tagger_optimizer'] == 'asgd':
			feature_optim = optim.ASGD
		elif params['tagger_optimizer'] == 'rmsprop':
			feature_optim = optim.RMSprop
		elif params['tagger_optimizer'] == 'rprop':
			feature_optim = optim.Rprop
		elif params['tagger_optimizer'] == 'sgd':
			feature_optim = optim.SGD
		self.feature_parameters = feature_parameters
		self.feature_optimizer = feature_optim(feature_parameters, lr=params['tagger_learning_rate'], momentum=0.9)

		tagger_parameters = []
		tagger_parameters += context_lstm.parameters()
		tagger_parameters += linear_proj.parameters()
		tagger_parameters += tagger_criterion.parameters()
		# optimizers
		tagger_optim = optim.SGD
		if params['tagger_optimizer'] == 'adadelta':
			tagger_optim = optim.Adadelta
		elif params['tagger_optimizer'] == 'adagrad':
			tagger_optim = optim.Adagrad
		elif params['tagger_optimizer'] == 'adam':
			tagger_optim = optim.Adam
		elif params['tagger_optimizer'] == 'adamax':
			tagger_optim = optim.Adamax
		elif params['tagger_optimizer'] == 'asgd':
			tagger_optim = optim.ASGD
		elif params['tagger_optimizer'] == 'rmsprop':
			tagger_optim = optim.RMSprop
		elif params['tagger_optimizer'] == 'rprop':
			tagger_optim = optim.Rprop
		elif params['tagger_optimizer'] == 'sgd':
			tagger_optim = optim.SGD
		self.tagger_parameters = tagger_parameters
		self.tagger_optimizer = tagger_optim(tagger_parameters, lr=params['tagger_learning_rate'], momentum=0.9)

		discriminator_parameters = []
		discriminator_parameters += seq_discriminator.parameters()
		# optimizers
		discriminator_optim = optim.SGD
		self.discriminator_parameters = discriminator_parameters
		self.discriminator_optimizer = discriminator_optim(discriminator_parameters, lr=params['tagger_learning_rate'],
														   momentum=0.9)

	def pretrain_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference):

		self.target_word_embedding.train()
		self.related_word_embedding.eval()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.train()
		self.related_char_cnn_word.eval()
		self.adv_lstm.train()
		self.seq_discriminator.eval()
		self.context_lstm.train()
		self.linear_proj.train()
		self.tagger_criterion.train()
		self.dis_criterion.train()

		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		target_combined_word_input, target_combined_word_input_dim = \
			self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)

		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		target_context_lstm_output = self.context_lstm(target_adv_lstm_output, target_seq_len,
															   len(target_seq_len), dropout=self.params['dropout'])
		target_pred_probs = self.linear_proj(target_context_lstm_output)

		target_tagger_loss = self.tagger_criterion(target_pred_probs, target_reference, target_seq_len,
														   decode=False)
		target_tagger_loss /= len(target_seq_len)

		loss_tagger = target_tagger_loss

		# check NaN
		if (loss_tagger != loss_tagger).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		self.feature_optimizer.zero_grad()
		self.tagger_optimizer.zero_grad()
		loss_tagger.backward()
		torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
		torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
		self.feature_optimizer.step()
		self.tagger_optimizer.step()
		return loss_tagger

	# only seq_discriminator and dis_criterion are trainable
	# only optimize dis loss
	def dis_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, related_word_ids,
				 related_char_ids, related_char_len, related_seq_len):

		self.target_word_embedding.train()
		self.related_word_embedding.train()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.train()
		self.related_char_cnn_word.train()
		self.adv_lstm.train()
		self.seq_discriminator.train()
		self.context_lstm.eval()
		self.linear_proj.eval()
		self.tagger_criterion.eval()
		# for target language
		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		# batchsize * seq_length * (word_emb + char_emb*3)
		target_combined_word_input, target_combined_word_input_dim = self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)
		# batchsize * seq_length * (word_emb + char_emb*3)*2
		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		# discriminator
		batch_size = target_word_ids.size(0)
		y = torch.FloatTensor(batch_size).zero_()
		y[:] = 1 - self.params['seq_dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		# batchsize
		target_discriminator_output = self.seq_discriminator(target_adv_lstm_output)
		loss_dis_target = F.binary_cross_entropy(target_discriminator_output, y)
		loss_dis_target /= len(target_seq_len)

		# for related languages
		input_related_words_old = self.related_word_embedding(related_word_ids)
		input_related_words = self.embedding_mapping(input_related_words_old)
		input_related_chars = self.char_embedding(related_char_ids)
		related_combined_word_input, related_combined_word_input_dim = \
			self.related_char_cnn_word(input_related_chars, related_char_len, input_related_words)

		related_adv_lstm_output = self.adv_lstm(related_combined_word_input, related_seq_len, len(related_seq_len), dropout=self.params['dropout'])

		batch_size = related_word_ids.size(0)
		y = torch.FloatTensor(batch_size).zero_()
		y[:] = self.params['seq_dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		related_discriminator_output = self.seq_discriminator(related_adv_lstm_output)

		loss_dis_related = F.binary_cross_entropy(related_discriminator_output, y)
		loss_dis_related /= len(related_seq_len)

		loss_dis = loss_dis_target + loss_dis_related

		# check NaN
		if (loss_dis != loss_dis).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		self.feature_optimizer.zero_grad()
		self.discriminator_optimizer.zero_grad()
		loss_dis.backward()
		torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
		torch.nn.utils.clip_grad_norm(self.discriminator_parameters, 5)
		self.feature_optimizer.step()
		self.discriminator_optimizer.step()
		# clip_parameters(self.dis_seq_optimizer, 0)  # self.params['dis_clip_weights']

		return loss_dis

	def dis_step1(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, related_word_ids,
				 related_char_ids, related_char_len, related_seq_len):

		self.target_word_embedding.train()
		self.related_word_embedding.train()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.train()
		self.related_char_cnn_word.train()
		self.adv_lstm.train()
		self.seq_discriminator.train()
		self.context_lstm.eval()
		self.linear_proj.eval()
		self.tagger_criterion.eval()

		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		target_combined_word_input, target_combined_word_input_dim = \
			self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)
		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		# discriminator
		batch_size = target_word_ids.size(0)
		s = np.random.normal(0.55, 0.1, batch_size)
		y = torch.from_numpy(s).float()
		#y = torch.FloatTensor(batch_size).zero_()
		#y[:] = 1 - self.params['seq_dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		target_discriminator_output = self.seq_discriminator(target_adv_lstm_output)

		loss_dis_target = F.binary_cross_entropy(target_discriminator_output, y)
		loss_dis_target /= len(target_seq_len)

		# for related languages
		input_related_words_old = self.related_word_embedding(related_word_ids)
		input_related_words = self.embedding_mapping(input_related_words_old)
		input_related_chars = self.char_embedding(related_char_ids)
		related_combined_word_input, related_combined_word_input_dim = \
			self.related_char_cnn_word(input_related_chars, related_char_len, input_related_words)

		related_adv_lstm_output = self.adv_lstm(related_combined_word_input, related_seq_len,
														len(related_seq_len), dropout=self.params['dropout'])

		batch_size = related_word_ids.size(0)
		s = 1 - np.random.normal(0.55, 0.1, batch_size)
		y = torch.from_numpy(s).float()
		# y = torch.FloatTensor(batch_size).zero_()
		# y[:] = self.params['seq_dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		related_discriminator_output = self.seq_discriminator(related_adv_lstm_output)

		loss_dis_related = F.binary_cross_entropy(related_discriminator_output, y)
		loss_dis_related /= len(related_seq_len)

		loss_dis = loss_dis_target + loss_dis_related

		# check NaN
		if (loss_dis != loss_dis).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		self.feature_optimizer.zero_grad()
		self.discriminator_optimizer.zero_grad()
		loss_dis.backward()
		torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
		torch.nn.utils.clip_grad_norm(self.discriminator_parameters, 5)
		self.feature_optimizer.step()
		self.discriminator_optimizer.step()
		# clip_parameters(self.dis_seq_optimizer, 0)  # self.params['dis_clip_weights']

		return loss_dis

	def tagger_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference,
					related_word_ids, related_char_ids, related_char_len, related_seq_len, related_reference):

		self.target_word_embedding.train()
		self.related_word_embedding.train()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.train()
		self.related_char_cnn_word.train()
		self.adv_lstm.train()
		self.seq_discriminator.eval()
		self.context_lstm.train()
		self.linear_proj.train()
		self.tagger_criterion.train()
		self.dis_criterion.train()

		# for target language
		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		target_combined_word_input, target_combined_word_input_dim = \
			self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)

		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		target_context_lstm_output = self.context_lstm(target_adv_lstm_output, target_seq_len,
															   len(target_seq_len), dropout=self.params['dropout'])
		target_pred_probs = self.linear_proj(target_context_lstm_output)

		target_tagger_loss = self.tagger_criterion(target_pred_probs, target_reference, target_seq_len,
														   decode=False)
		target_tagger_loss /= len(target_seq_len)

		# discriminator
		batch_size = target_word_ids.size(0)
		y = torch.FloatTensor(batch_size).zero_()
		y[:] = 1 #- self.params['seq_dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		target_discriminator_output = self.seq_discriminator(target_adv_lstm_output)

		loss_dis_target = F.binary_cross_entropy(target_discriminator_output, 1 - y)
		loss_dis_target /= len(target_seq_len)

		# for related languages
		input_related_words_old = self.related_word_embedding(related_word_ids)
		input_related_words = self.embedding_mapping(input_related_words_old)
		input_related_chars = self.char_embedding(related_char_ids)
		related_combined_word_input, related_combined_word_input_dim = \
			self.related_char_cnn_word(input_related_chars, related_char_len, input_related_words)

		related_adv_lstm_output = self.adv_lstm(related_combined_word_input, related_seq_len,
														len(related_seq_len), dropout=self.params['dropout'])

		related_context_lstm_output = self.context_lstm(related_adv_lstm_output, related_seq_len,
																len(related_seq_len), dropout=self.params['dropout'])
		related_pred_probs = self.linear_proj(related_context_lstm_output)

		batch_size = related_word_ids.size(0)
		y = torch.FloatTensor(batch_size).zero_()
		#y[:] = self.params['dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		related_discriminator_output = self.seq_discriminator(related_adv_lstm_output)

		loss_dis_related = F.binary_cross_entropy(related_discriminator_output, 1 - y)
		loss_dis_related /= len(related_seq_len)

		related_discriminator_output_target = 1 - related_discriminator_output
		related_discriminator_output_all = torch.stack([related_discriminator_output_target,
														related_discriminator_output], 1)
		related_max, related_discriminator_output_label = torch.max(related_discriminator_output_all, 1)
		related_discriminator_output_label = related_discriminator_output_label.float()

		related_tagger_loss = self.tagger_criterion(related_pred_probs, related_reference, related_seq_len,
															decode=False)
		related_tagger_loss /= len(related_seq_len)

		# loss_dis = loss_dis_target + loss_dis_related
		loss_tagger = target_tagger_loss + related_tagger_loss + loss_dis_target + loss_dis_related  #loss_dis

		# check NaN
		if (loss_tagger != loss_tagger).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		self.feature_optimizer.zero_grad()
		self.tagger_optimizer.zero_grad()
		loss_tagger.backward()
		torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
		torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
		self.feature_optimizer.step()
		self.tagger_optimizer.step()
		# clip_parameters(self.tagger_optimizer, 0)  # self.params['dis_clip_weights']

		return loss_tagger

	def tagger_step1(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference,
					related_word_ids, related_char_ids, related_char_len, related_seq_len, related_reference):

		self.target_word_embedding.train()
		self.related_word_embedding.train()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.train()
		self.related_char_cnn_word.train()
		self.adv_lstm.train()
		self.seq_discriminator.eval()
		self.context_lstm.train()
		self.linear_proj.train()
		self.tagger_criterion.train()
		self.dis_criterion.train()

		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		target_combined_word_input, target_combined_word_input_dim = \
			self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)

		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		target_context_lstm_output = self.context_lstm(target_adv_lstm_output, target_seq_len,
															   len(target_seq_len), dropout=self.params['dropout'])
		target_pred_probs = self.linear_proj(target_context_lstm_output)

		target_tagger_loss = self.tagger_criterion(target_pred_probs, target_reference, target_seq_len,
														   decode=False)
		target_tagger_loss /= len(target_seq_len)

		# discriminator
		batch_size = target_word_ids.size(0)
		y = torch.FloatTensor(batch_size).zero_()
		y[:] = 1 - self.params['seq_dis_smooth']-0.3
		y = Variable(y.cuda() if self.params['gpu'] else y)
		target_discriminator_output = self.seq_discriminator(target_adv_lstm_output)

		loss_dis_target = F.binary_cross_entropy(target_discriminator_output, y)
		loss_dis_target /= len(target_seq_len)

		# for related languages
		input_related_words_old = self.related_word_embedding(related_word_ids)
		input_related_words = self.embedding_mapping(input_related_words_old)
		input_related_chars = self.char_embedding(related_char_ids)
		related_combined_word_input, related_combined_word_input_dim = \
			self.related_char_cnn_word(input_related_chars, related_char_len, input_related_words)

		related_adv_lstm_output = self.adv_lstm(related_combined_word_input, related_seq_len,
														len(related_seq_len), dropout=self.params['dropout'])

		related_context_lstm_output = self.context_lstm(related_adv_lstm_output, related_seq_len,
																len(related_seq_len), dropout=self.params['dropout'])
		related_pred_probs = self.linear_proj(related_context_lstm_output)

		batch_size = related_word_ids.size(0)
		y = torch.FloatTensor(batch_size).zero_()
		y[:] = self.params['seq_dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		related_discriminator_output = self.seq_discriminator(related_adv_lstm_output)

		loss_dis_related = F.binary_cross_entropy(related_discriminator_output, 1 - y)
		loss_dis_related /= len(related_seq_len)

		related_discriminator_output_target = 1 - related_discriminator_output
		related_discriminator_output_all = torch.stack([related_discriminator_output_target,
														related_discriminator_output], 1)
		related_max, related_discriminator_output_label = torch.max(related_discriminator_output_all, 1)
		related_discriminator_output_label = related_discriminator_output_label.float()

		related_tagger_loss = self.tagger_criterion(related_pred_probs, related_reference, related_seq_len,
															decode=False, batch_mask=related_discriminator_output_label)
		related_tagger_loss /= len(related_seq_len)

		# loss_dis = loss_dis_target + loss_dis_related
		loss_tagger = target_tagger_loss + related_tagger_loss + loss_dis_target + loss_dis_related #loss_dis

		# check NaN
		if (loss_tagger != loss_tagger).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		self.feature_optimizer.zero_grad()
		self.tagger_optimizer.zero_grad()
		loss_tagger.backward()
		torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
		torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
		self.feature_optimizer.step()
		self.tagger_optimizer.step()
		# clip_parameters(self.tagger_optimizer, 0)  # self.params['dis_clip_weights']

		return loss_tagger

	def tagging_train_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference,
						   related_word_ids, related_char_ids, related_char_len, related_seq_len, related_reference):

		self.target_word_embedding.train()
		self.related_word_embedding.train()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.train()
		self.related_char_cnn_word.train()
		self.adv_lstm.train()
		self.seq_discriminator.eval()
		self.context_lstm.train()
		self.linear_proj.train()
		self.tagger_criterion.train()
		self.dis_criterion.train()

		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		target_combined_word_input, target_combined_word_input_dim = \
			self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)

		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		target_context_lstm_output = self.context_lstm(target_adv_lstm_output, target_seq_len,
															   len(target_seq_len), dropout=self.params['dropout'])
		target_pred_probs = self.linear_proj(target_context_lstm_output)

		target_tagger_loss = self.tagger_criterion(target_pred_probs, target_reference, target_seq_len,
														   decode=False)
		target_tagger_loss /= len(target_seq_len)

		# discriminator
		batch_size = target_word_ids.size(0)
		y = torch.LongTensor(batch_size).zero_()
		y[:] = self.params['dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		target_discriminator_output = self.seq_discriminator(target_adv_lstm_output)
		# loss_dis_target = F.binary_cross_entropy(target_discriminator_output, y)
		# loss_dis_target = self.dis_criterion(target_discriminator_output, y)
		loss_dis_target = F.cross_entropy(target_discriminator_output, 1 - y)
		loss_dis_target /= len(target_seq_len)

		# for related languages
		input_related_words_old = self.related_word_embedding(related_word_ids)
		input_related_words = self.embedding_mapping(input_related_words_old)
		input_related_chars = self.char_embedding(related_char_ids)
		related_combined_word_input, related_combined_word_input_dim = \
			self.related_char_cnn_word(input_related_chars, related_char_len, input_related_words)

		related_adv_lstm_output = self.adv_lstm(related_combined_word_input, related_seq_len,
														len(related_seq_len), dropout=self.params['dropout'])

		related_context_lstm_output = self.context_lstm(related_adv_lstm_output, related_seq_len,
																len(related_seq_len), dropout=self.params['dropout'])
		related_pred_probs = self.linear_proj(related_context_lstm_output)
		related_tagger_loss = self.tagger_criterion(related_pred_probs, related_reference, related_seq_len,
															decode=False)
		related_tagger_loss /= len(related_seq_len)

		batch_size = related_word_ids.size(0)
		y = torch.LongTensor(batch_size).zero_()
		y[:] = 1 - self.params['dis_smooth']
		y = Variable(y.cuda() if self.params['gpu'] else y)
		related_discriminator_output = self.seq_discriminator(related_adv_lstm_output)
		# loss_dis_related = self.dis_criterion(related_discriminator_output, y)
		# loss_dis_related = F.binary_cross_entropy(related_discriminator_output, y)
		loss_dis_related = F.cross_entropy(related_discriminator_output, 1 - y)
		loss_dis_related /= len(related_seq_len)

		loss_dis = loss_dis_target + loss_dis_related
		loss_tagger = target_tagger_loss + related_tagger_loss + loss_dis

		# check NaN
		if (loss_dis != loss_dis).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		self.tagger_optimizer.zero_grad()
		loss_tagger.backward()
		torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
		self.tagger_optimizer.step()
		# clip_parameters(self.tagger_optimizer, 0)  # self.params['dis_clip_weights']

		return loss_tagger

	def target_tagging_train_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len,
								  target_reference):

		self.target_word_embedding.train()
		self.related_word_embedding.eval()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.train()
		self.related_char_cnn_word.eval()
		self.adv_lstm.train()
		self.seq_discriminator.eval()
		self.context_lstm.train()
		self.linear_proj.train()
		self.tagger_criterion.train()
		self.dis_criterion.train()

		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		target_combined_word_input, target_combined_word_input_dim = \
			self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)

		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		target_context_lstm_output = self.context_lstm(target_adv_lstm_output, target_seq_len,
															   len(target_seq_len), dropout=self.params['dropout'])
		target_pred_probs = self.linear_proj(target_context_lstm_output)

		target_tagger_loss = self.tagger_criterion(target_pred_probs, target_reference, target_seq_len,
														   decode=False)
		target_tagger_loss /= len(target_seq_len)

		loss_tagger = target_tagger_loss

		# check NaN
		if (loss_tagger != loss_tagger).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		self.mono_tagger_optimizer.zero_grad()
		loss_tagger.backward()
		torch.nn.utils.clip_grad_norm(self.mono_tagger_parameters, 5)
		self.mono_tagger_optimizer.step()

		return loss_tagger

	def related_tagging_train_step(self, related_word_ids, related_char_ids, related_char_len, related_seq_len,
								   related_reference):

		self.target_word_embedding.eval()
		self.related_word_embedding.train()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.eval()
		self.related_char_cnn_word.train()
		self.adv_lstm.train()
		self.seq_discriminator.eval()
		self.context_lstm.train()
		self.linear_proj.train()
		self.tagger_criterion.train()
		self.dis_criterion.train()

		# for related languages
		input_related_words_old = self.related_word_embedding(related_word_ids)
		input_related_words = self.embedding_mapping(input_related_words_old)
		input_related_chars = self.char_embedding(related_char_ids)
		related_combined_word_input, related_combined_word_input_dim = \
			self.related_char_cnn_word(input_related_chars, related_char_len, input_related_words)

		related_adv_lstm_output = self.adv_lstm(related_combined_word_input, related_seq_len,
														len(related_seq_len), dropout=self.params['dropout'])

		related_context_lstm_output = self.context_lstm(related_adv_lstm_output, related_seq_len,
																len(related_seq_len), dropout=self.params['dropout'])
		related_pred_probs = self.linear_proj(related_context_lstm_output)
		related_tagger_loss = self.tagger_criterion(related_pred_probs, related_reference, related_seq_len,
															decode=False)
		related_tagger_loss /= len(related_seq_len)
		loss_tagger = related_tagger_loss

		# check NaN
		if (loss_tagger != loss_tagger).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		self.related_tagger_optimizer.zero_grad()
		loss_tagger.backward()
		torch.nn.utils.clip_grad_norm(self.related_tagger_parameters, 5)
		self.related_tagger_optimizer.step()
		# clip_parameters(self.tagger_optimizer, 0)  # self.params['dis_clip_weights']

		return loss_tagger

	def combine_tagging_train_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len,
								   target_reference, related_word_ids, related_char_ids, related_char_len,
								   related_seq_len, related_reference):

		self.target_word_embedding.train()
		self.related_word_embedding.train()
		self.embedding_mapping.eval()
		self.char_embedding.train()
		self.target_char_cnn_word.train()
		self.related_char_cnn_word.train()
		self.adv_lstm.train()
		self.seq_discriminator.eval()
		self.context_lstm.train()
		self.linear_proj.train()
		self.tagger_criterion.train()
		self.dis_criterion.train()

		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		target_combined_word_input, target_combined_word_input_dim = \
			self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)

		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		target_context_lstm_output = self.context_lstm(target_adv_lstm_output, target_seq_len,
															   len(target_seq_len), dropout=self.params['dropout'])
		target_pred_probs = self.linear_proj(target_context_lstm_output)

		target_tagger_loss = self.tagger_criterion(target_pred_probs, target_reference, target_seq_len,
														   decode=False)
		target_tagger_loss /= len(target_seq_len)

		# for related languages
		input_related_words_old = self.related_word_embedding(related_word_ids)
		input_related_words = self.embedding_mapping(input_related_words_old)
		input_related_chars = self.char_embedding(related_char_ids)
		related_combined_word_input, related_combined_word_input_dim = \
			self.related_char_cnn_word(input_related_chars, related_char_len, input_related_words)

		related_adv_lstm_output = self.adv_lstm(related_combined_word_input, related_seq_len,
														len(related_seq_len), dropout=self.params['dropout'])

		related_context_lstm_output = self.context_lstm(related_adv_lstm_output, related_seq_len,
																len(related_seq_len), dropout=self.params['dropout'])
		related_pred_probs = self.linear_proj(related_context_lstm_output)

		related_discriminator_output = self.seq_discriminator(related_adv_lstm_output)
		# related_max, related_discriminator_output_label = torch.max(related_discriminator_output, 1)
		# related_discriminator_output_label = related_discriminator_output_label.float()

		related_discriminator_output_neg = 1 - related_discriminator_output
		related_discriminator_output_all = torch.stack([related_discriminator_output, related_discriminator_output_neg],
													   1)
		related_max, related_discriminator_output_label = torch.max(related_discriminator_output_all, 1)
		related_discriminator_output_label = related_discriminator_output_label.float()

		related_tagger_loss = self.tagger_criterion(related_pred_probs, related_reference, related_seq_len,
															decode=False) # , batch_mask=related_discriminator_output_label
		related_tagger_loss /= len(related_seq_len)

		loss_tagger = target_tagger_loss + related_tagger_loss

		# check NaN
		if (loss_tagger != loss_tagger).data.any():
			print("NaN detected (discriminator)")
			exit()

		# optim
		# self.feature_optimizer.zero_grad()
		# self.tagger_optimizer.zero_grad()
		# loss_tagger.backward()
		# torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
		# torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
		# self.feature_optimizer.step()
		# self.tagger_optimizer.step()

		#self.feature_optimizer.zero_grad()
		self.tagger_optimizer.zero_grad()
		loss_tagger.backward()
		#torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
		torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
		#self.feature_optimizer.step()
		self.tagger_optimizer.step()

		return loss_tagger

	def tagging_dev_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference):

		self.target_word_embedding.eval()
		self.related_word_embedding.eval()
		self.embedding_mapping.eval()
		self.char_embedding.eval()
		self.target_char_cnn_word.eval()
		self.related_char_cnn_word.eval()
		self.adv_lstm.eval()
		self.seq_discriminator.eval()
		self.context_lstm.eval()
		self.linear_proj.eval()
		self.tagger_criterion.eval()
		self.dis_criterion.eval()

		# batchsize * seq_length * word_emb
		input_target_words = self.target_word_embedding(target_word_ids)
		# batchsize * seq_length * max_char_length * char_emb
		input_target_chars = self.char_embedding(target_char_ids)
		target_combined_word_input, target_combined_word_input_dim = \
			self.target_char_cnn_word(input_target_chars, target_char_len, input_target_words)

		target_adv_lstm_output = self.adv_lstm(target_combined_word_input, target_seq_len, len(target_seq_len),
													   dropout=self.params['dropout'])

		target_context_lstm_output = self.context_lstm(target_adv_lstm_output, target_seq_len,
															   len(target_seq_len), dropout=self.params['dropout'])
		target_pred_probs = self.linear_proj(target_context_lstm_output)

		dev_pred_seq = self.tagger_criterion(target_pred_probs, target_reference, target_seq_len, decode=True)

		return dev_pred_seq

	def related_tagging_dev_step(self, related_word_ids, related_char_ids, related_char_len, related_seq_len):

		self.target_word_embedding.eval()
		self.related_word_embedding.eval()
		self.embedding_mapping.eval()
		self.char_embedding.eval()
		self.target_char_cnn_word.eval()
		self.related_char_cnn_word.eval()
		self.adv_lstm.eval()
		self.seq_discriminator.eval()
		self.context_lstm.eval()
		self.linear_proj.eval()
		self.tagger_criterion.eval()
		self.dis_criterion.eval()

		# for related languages
		# for related languages
		input_related_words_old = self.related_word_embedding(related_word_ids)
		input_related_words = self.embedding_mapping(input_related_words_old)
		input_related_chars = self.char_embedding(related_char_ids)
		related_combined_word_input, related_combined_word_input_dim = \
			self.related_char_cnn_word(input_related_chars, related_char_len, input_related_words)

		related_adv_lstm_output = self.adv_lstm(related_combined_word_input, related_seq_len,
														len(related_seq_len), dropout=self.params['dropout'])

		related_context_lstm_output = self.context_lstm(related_adv_lstm_output, related_seq_len,
																len(related_seq_len), dropout=self.params['dropout'])
		related_pred_probs = self.linear_proj(related_context_lstm_output)

		related_discriminator_output = self.seq_discriminator(related_adv_lstm_output)
		related_discriminator_output_target = 1 - related_discriminator_output
		related_discriminator_output_all = torch.stack([related_discriminator_output_target,
														related_discriminator_output], 1)
		related_max, related_discriminator_output_label = torch.max(related_discriminator_output_all, 1)
		related_discriminator_output_label = related_discriminator_output_label.float()

		return related_max, related_discriminator_output_label, related_discriminator_output
