import json
import os
import pickle
import random
import sys

import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch import optim
from torch.autograd import Variable
from gensim.models import KeyedVectors

import BiLSTM_LSTM
import Jointly_RL
from Parser import Parser
# from TFgirl.RE.PreProcess.data_manager import DataManager
from general_utils import padding_sequence, get_minibatches, get_bags
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers import optimization

import Noisy_RL

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(datasets, mode):  # optimizer, criterion, args,
	# JointModel.train()
	if args.use_RL:
		mini_batches = get_bags(datasets, relations, args.batchsize)
		noisy_sentences_vec = Variable(torch.FloatTensor(1, args.hidden_dim).fill_(0))
		noisy_vec_mean = torch.mean(noisy_sentences_vec, 0, True)
	else:
		mini_batches = get_minibatches(datasets, args.batchsize)
	batchcnt = len(datasets[0]) // args.batchsize  # len(list(mini_batches))
	logger.info("********************%s data*********************" % mode)
	logger.info("number of batches: %s" % batchcnt)
	NER_correct, NER_total = 0., 0.
	RE_correct, RE_total = 0., 0.
	if mode != 'train':
		# NER_target_all, NER_output_all = None, None
		# RE_target_all, RE_output_all = None, None
		NER_target_all2, NER_output_all2 = [], []
		RE_target_all2, RE_output_all2 = [], []
		NER_output_logits, RE_output_logits = [], []

	for b, data in enumerate(mini_batches):
		if b >= batchcnt:
			break
		sentences, pos_lambda, tags, sentences_words, relation_tags, relation_names = data
		input_tensor, input_length = padding_sequence(sentences, pad_token=0)
		pos_tensor, input_length = padding_sequence(pos_lambda, pad_token=0)
		target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)   # entity tags
		relation_target_tensor = relation_tags  # padding_sequence_recurr(relation_tags)  		# relation tag
		if torch.cuda.is_available():
			input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
			target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
			if args.encoder_model == "BiLSTM":
				mask = torch.cuda.ByteTensor((1 - (target_tensor == args.entity_tag_size))).to(device)
			else:
				mask = torch.cuda.ByteTensor((1 - (input_tensor == 0))).to(device)
			pos_tensor = Variable(torch.cuda.FloatTensor(pos_tensor, device=device)).cuda()
			relation_target_tensor = Variable(torch.cuda.LongTensor(relation_target_tensor, device=device)).cuda()
		else:
			input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
			target_tensor = Variable(torch.LongTensor(target_tensor, device=device))
			if args.encoder_model == "BiLSTM":
				mask = torch.ByteTensor((1 - (target_tensor == args.entity_tag_size))).to(device)
			else:
				mask = torch.ByteTensor((1 - (input_tensor == 0))).to(device)
			pos_tensor = Variable(torch.Tensor(pos_tensor, device=device))
			relation_target_tensor = Variable(torch.LongTensor(relation_target_tensor, device=device))

		if mode == 'train':
			optimizer.zero_grad()
			NER_active_logits, NER_active_labels, RE_output_tag, NER_output_tag, NER_output, BERT_pooled_output = JointModel(
									input_tensor, pos_tensor, target_tensor,
									args.batchsize,
									mask)  # , input_length, target_length
			if args.use_RL:
				mask_entity = [list(map(lambda x: 1 if x in [1, 2, 4, 5] else 0, i)) for i in target_tensor]
				if torch.cuda.is_available():
					mask_entity = torch.cuda.ByteTensor(mask_entity).to(device)
				else:
					mask_entity = torch.ByteTensor(mask_entity).to(device)
				NER_embedding = None
				for i in range(len(mask_entity)):
					NER_embedding = torch.mean(NER_output[i][mask_entity[i]], 0).view(1, -1) if NER_embedding is None \
						else torch.cat((NER_embedding, torch.mean(NER_output[i][mask_entity[i]], 0).view(1, -1)), 0)

				RE_rewards, loss_RL, noisy_sentences_vec, noisy_vec_mean = RL_model(BERT_pooled_output, NER_embedding,
								JointModel.noysy_model, RE_output_tag, relation_target_tensor, noisy_sentences_vec, noisy_vec_mean)

			if not args.use_RL:
				loss_entity = criterion(NER_active_logits, NER_active_labels)
				loss_RE = criterion(RE_output_tag, relation_target_tensor)
				loss = loss_entity + loss_RE
				if args.merge_loss:
					loss.backward()
				else:
					loss_entity.backward(retain_graph=True)  # retain_graph=True
					loss_RE.backward(retain_graph=True)
			if args.use_RL:
				loss = loss_RL
				loss_RL.backward()
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

			optimizer.step()
		else:
			NER_active_logits, NER_active_labels, RE_output_tag, NER_output_tag, _, _ = JointModel(
									input_tensor, pos_tensor, target_tensor,
									args.batchsize,
									mask, True)  # , input_length, target_length
		NER_correct += (torch.argmax(NER_active_logits, -1) == NER_active_labels).sum().item()
		NER_total += len(NER_active_logits)
		# temp = 0.
		# for i in range(len(relation_target_tensor[0])):
		# 	target = torch.transpose(relation_target_tensor, 0, 1)[i]
		# 	temp += (torch.argmax(RE_output_tag, -1) == target).sum().item()
		RE_correct += (torch.argmax(RE_output_tag, -1) == relation_target_tensor).sum().item()
		RE_total += len(RE_output_tag)
		if mode != 'train':
			NER_target_all2.append(target_tensor.cpu().tolist())  # target_tensor, NER_active_labels .numpy()
			NER_output_all2.append(torch.argmax(NER_output_tag, -1).cpu().tolist())  # NER_output_tag, NER_active_logits
			NER_output_logits.append(NER_output_tag.detach().cpu().tolist())
			RE_output_all2.append(torch.argmax(RE_output_tag, -1).cpu().tolist())
			RE_target_all2.append(relation_target_tensor.detach().cpu().tolist())
			RE_output_logits.append(RE_output_tag.cpu().tolist())
			if b % args.print_batch == 0:
				logger.info('seq-seq model: (%d %.2f%%), NER acc: %.4f, RE acc: %.4f' %
						(b, float(b) / batchcnt * 100, NER_correct / NER_total, RE_correct / RE_total))
			'''if not args.do_train:
				if NER_target_all is None:
					NER_target_all = NER_active_labels.to('cpu')
					NER_output_all = NER_active_logits.to('cpu')
				else:
					NER_target_all = torch.cat((NER_target_all.to('cpu'), NER_active_labels.to('cpu')), dim=0)
					NER_output_all = torch.cat((NER_output_all.to('cpu'), NER_active_logits.to('cpu')), dim=0)
				if RE_target_all is None:
					RE_target_all = relation_target_tensor.to('cpu')
					RE_output_all = RE_output_tag.to('cpu')
				else:
					RE_target_all = torch.cat((RE_target_all.to('cpu'), relation_target_tensor.to('cpu')), dim=0)
					RE_output_all = torch.cat((RE_output_all.to('cpu'), RE_output_tag.to('cpu')), dim=0)'''
		if mode == 'train':
			out_losses.append(loss.item())
			if b % args.print_batch == 0:
				logger.info('seq-seq model: (%d %.2f%%), loss_NER: %.4f, loss_RE: %.4f, NER acc: %.4f, RE acc: %.4f' %
				  (b, float(b) / batchcnt * 100, loss_entity.item(), loss_RE.item(), NER_correct / NER_total, RE_correct / RE_total))

	if mode != 'train':
		cal_F_score(RE_output_all2, RE_target_all2, NER_target_all2, NER_output_all2, args.batchsize)
		if args.do_train:
			if mode == 'test' or (mode == 'dev'and e == args.epochRL-1):
				with open(args.output_dir+'predict_%s_epoch_%s.json' % (mode, e), "a+") as fw:
					json.dump({"RE_predict": RE_output_all2, "RE_actual": RE_target_all2, "RE_output_logits": RE_output_logits,
							   "NER_predict": NER_output_all2, "NER_actual": NER_target_all2, "NER_output_logits": NER_output_logits}, fw)
		else:
			with open(args.output_dir + 'predict_%s.json' % mode, "a+") as fw:
				json.dump({"RE_predict": RE_output_all2, "RE_actual": RE_target_all2, "RE_output_logits": RE_output_logits,
						   "NER_predict": NER_output_all2, "NER_actual": NER_target_all2, "NER_output_logits": NER_output_logits}, fw)
			# np.save('pred_res/RE_predict', RE_output_all2)  # RE_output_all.to('cpu').detach().numpy()
			# np.save('pred_res/RE_actual', RE_target_all2)
			# np.save('pred_res/NER_predict', NER_output_all2)
			# np.save('pred_res/NER_actual', NER_target_all2)

			'''NER_pred_res = metrics.classification_report(NER_target_all2, NER_output_all2)
			logger.info('NER Prediction results: \n{}'.format(NER_pred_res))
			RE_pred_res = metrics.classification_report(RE_target_all2, RE_output_all2)
			logger.info('RE Prediction results: \n{}'.format(RE_pred_res))'''
	else:
		np.save(args.output_dir+"loss_train", out_losses)
	# np.save("RL_RE_loss_train", RL_RE_losses)
	# np.save("RE_rewards_train", RE_rewardsall)
	# np.save("TOTAL_rewards_train", TOTAL_rewardsall)


def calc_acc_total(all_acc, all_tot, all_cnt, relation_action, entity_action, relation_labels, entity_labels):
	# acc, cnt, tot = 0, 0, len(relation_labels)
	all_tot += len(relation_labels)
	cnt = 0.
	# used = [0 for i in range(len(relation_action))]
	# tp, tags = label, label['tags']
	j, ok = 0, 0
	# for i in range(len(relation_action)):  # each round
	if isinstance(relation_action, np.int64) or isinstance(relation_action, int):
		for label in relation_labels:
			if label == relation_action and ok == 0:  # and label > 0 relation_action[i] == label and used[i] == 0
				match = 1
				for k in range(len(entity_labels)):
					if entity_labels[k] in [1, 2, 4, 5] and entity_action[k] != entity_labels[k]:
						match = 0

				if match == 1:
					ok = 1
			all_acc += ok
		# if relation_action > 0:
		cnt += 1
		all_cnt += cnt
	else:
		for label in relation_labels:
			if label in relation_action and ok == 0 and label > 0:  # relation_action[i] == label and used[i] == 0
				match = 1
				for k in range(len(entity_labels)):
					if entity_labels[k] in [1, 2, 4, 5] and entity_action[k] != entity_labels[k]:
						match = 0
				if match == 1:
					ok = 1
			all_acc += ok
			# used[i] = 1
		for i in range(len(relation_action)):
			if relation_action[i] > 0:
				# j += 1
				cnt += 1

		all_cnt += cnt // len(relation_labels)
	return all_acc, all_tot, all_cnt


def cal_F_score(relation_actions_batch, train_relation_tags,  train_entity_tags, entity_actions_batch, batch_size):
		batch_num = len(relation_actions_batch)  # batch_size = self.batch_size  # len(relation_actions_batch)
		# cal the P,R and F of relation extraction for a batch of sentences
		# acc_total, tot_total, cnt_total = 0., 0., 0.
		tot_R_relation_num = 0.
		acc_R, cnt_R, tot_R = 0., 0., 0.  # len(train_relation_tags)
		# acc_R_last, cnt_R_last, tot_R_last = 0., 0., 0.
		cnt_R_last = 0.
		rec_R = 0.
		acc_E, cnt_E, tot_E = 0., 0., 0.  # len(train_entity_tags)
		acc_E_no0 = 0.
		acc_total, tot_total, cnt_total = 0., 0., 0.
		for i in range(batch_num):
			for sentence_id in range(batch_size):
				relation_tag = [train_relation_tags[i][sentence_id]]  # set()
				if isinstance(relation_actions_batch[i][sentence_id], np.int64) or isinstance(relation_actions_batch[i][sentence_id], int):
					round_num = 1
				else:
					round_num = len(relation_actions_batch[i][sentence_id])
				acc_total, tot_total, cnt_total = calc_acc_total(acc_total, tot_total, cnt_total, relation_actions_batch[i][sentence_id],
																	  entity_actions_batch[i][sentence_id],
																	  relation_tag,
																	  train_entity_tags[i][sentence_id])

				if isinstance(relation_actions_batch[i][sentence_id], np.int64) or isinstance(relation_actions_batch[i][sentence_id], int):
					if relation_actions_batch[i][sentence_id] in relation_tag:
						acc_R += 1
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
					# if each_relation > 0:
					# tot_R += 1
					if isinstance(relation_actions_batch[i][sentence_id], np.int64) or isinstance(relation_actions_batch[i][sentence_id], int):
						if each_relation == relation_actions_batch[i][sentence_id]:
							rec_R += 1
					else:
							if each_relation == relation_actions_batch[i][sentence_id] or each_relation in relation_actions_batch[sentence_id]:
								rec_R += 1
				# entity extraction
				for word_id in range(len(train_entity_tags[i][sentence_id])):
					if train_entity_tags[i][sentence_id][word_id] == 7:
						break
					if int(entity_actions_batch[i][sentence_id][word_id]) == train_entity_tags[i][sentence_id][word_id]:
						acc_E += 1
					if train_entity_tags[i][sentence_id][word_id] > 0:
						cnt_E += 1
						if int(entity_actions_batch[i][sentence_id][word_id]) == train_entity_tags[i][sentence_id][word_id]:
							acc_E_no0 += 1
					tot_E += 1

		precision_total = acc_total / cnt_total
		recall_total = acc_total / tot_total
		beta = 1.
		try:
			F_total = (1 + beta * beta) * precision_total * recall_total / (beta * beta * precision_total + recall_total)
		except Exception as e:
			logger.warning(e)
			F_total = 0.
		logger.info("********: TOTAL precision: " + str(precision_total) + ", recall: " + str(recall_total) + ", F-score: " + str(F_total))

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
			logger.warning(e)
			F_RE = 0.
		logger.info("********: Relation precision: " + str(acc_R / tot_R_relation_num) + ", " + str(
			acc_R / cnt_R_last) + ", " + str(precision_R) +
			  ", recall: " + str(recall_R) + ", F-score: " + str(F_RE))

		# cal the P,R and F of entity extraction for each sentence
		precision_E = acc_E / tot_E
		recall_E = acc_E_no0 / cnt_E  # acc_E / tot_E
		try:
			F_NER = (1 + beta * beta) * precision_E * recall_E / (beta * beta * precision_E + recall_E)
		except Exception as e:
			logger.warning(e)
			F_NER = 0.
		logger.info("********: Entity precision: " + str(precision_E) + ", recall: " + str(recall_E) + ", F-score: " + str(F_NER))


if __name__ == "__main__":
	argv = sys.argv[1:]
	parser = Parser().getParser()
	args, _ = parser.parse_known_args(argv)
	logger.info("Load data start...")
	# dm = DataManager(args.datapath, args.testfile)
	wv = None
	if args.encoder_model == "BiLSTM":
		wv = np.loadtxt(args.datapath+'wv.txt')
	# load data from pkl file
	with open(args.datapath+'data_train.pkl', 'rb') as inp:
		train_sentences_words = pickle.load(inp)  # sentence with words
		train_sentences_id = pickle.load(inp)  # sentence with word_ids
		train_position_lambda = pickle.load(inp)  # position lambda
		train_entity_tags = pickle.load(inp)  # entity tags
		train_relation_tags = pickle.load(inp)  # relation type_id
		train_relation_names = pickle.load(inp)  # relation name

	with open(args.datapath+'data_test.pkl', 'rb') as inp:
		test_sentences_words = pickle.load(inp)  # sentence with words
		test_sentences_id = pickle.load(inp)  # sentence with word_ids
		test_position_lambda = pickle.load(inp)  # position lambda
		test_entity_tags = pickle.load(inp)  # entity tags
		test_relation_tags = pickle.load(inp)  # relation type_id
		test_relation_names = pickle.load(inp)  # relation name

	with open(args.datapath + 'data_dev.pkl', 'rb') as inp:
		dev_sentences_words = pickle.load(inp)  # sentence with words
		dev_sentences_id = pickle.load(inp)  # sentence with word_ids
		dev_position_lambda = pickle.load(inp)  # position lambda
		dev_entity_tags = pickle.load(inp)  # entity tags
		dev_relation_tags = pickle.load(inp)  # relation type_id
		dev_relation_names = pickle.load(inp)  # relation name

	if args.encoder_model == 'BERT':
		tokenizer = BertTokenizer.from_pretrained(args.pretrained_dir, do_basic_tokenize=False)
		train_sentences_id = []
		for sentence_id in range(len(train_sentences_words)):
			sentence = train_sentences_words[sentence_id]
			tokened_sent = tokenizer.tokenize(sentence)
			for index in range(len(tokened_sent)):
				if '##' in tokened_sent[index]:
					train_position_lambda[sentence_id].insert(index, train_position_lambda[sentence_id][index - 1])
					train_entity_tags[sentence_id].insert(index, train_entity_tags[sentence_id][index - 1])
			sentence2id = tokenizer.convert_tokens_to_ids(tokened_sent)
			if len(sentence2id) >= args.max_seq_length:
				sentence2id = sentence2id[:args.max_seq_length]
				train_position_lambda[sentence_id] = train_position_lambda[sentence_id][:args.max_seq_length]
				train_entity_tags[sentence_id] = train_entity_tags[sentence_id][:args.max_seq_length]
			train_sentences_id.append(sentence2id)

		test_sentences_id = []
		for sentence_id in range(len(test_sentences_words)):
			sentence = test_sentences_words[sentence_id]
			tokened_sent = tokenizer.tokenize(sentence)
			for index in range(len(tokened_sent)):
				if '##' in tokened_sent[index]:
					test_position_lambda[sentence_id].insert(index, test_position_lambda[sentence_id][index - 1])
					test_entity_tags[sentence_id].insert(index, test_entity_tags[sentence_id][index - 1])
			sentence2id = tokenizer.convert_tokens_to_ids(tokened_sent)
			if len(sentence2id) >= args.max_seq_length:
				sentence2id = sentence2id[:args.max_seq_length]
				test_position_lambda[sentence_id] = test_position_lambda[sentence_id][:args.max_seq_length]
				test_entity_tags[sentence_id] = test_entity_tags[sentence_id][:args.max_seq_length]
			test_sentences_id.append(sentence2id)

		dev_sentences_id = []
		for sentence_id in range(len(dev_sentences_words)):
			sentence = dev_sentences_words[sentence_id]
			tokened_sent = tokenizer.tokenize(sentence)
			for index in range(len(tokened_sent)):
				if '##' in tokened_sent[index]:
					dev_position_lambda[sentence_id].insert(index, dev_position_lambda[sentence_id][index - 1])
					dev_entity_tags[sentence_id].insert(index, dev_entity_tags[sentence_id][index - 1])
			sentence2id = tokenizer.convert_tokens_to_ids(tokened_sent)
			if len(sentence2id) >= args.max_seq_length:
				sentence2id = sentence2id[:args.max_seq_length]
				dev_position_lambda[sentence_id] = dev_position_lambda[sentence_id][:args.max_seq_length]
				dev_entity_tags[sentence_id] = dev_entity_tags[sentence_id][:args.max_seq_length]
			dev_sentences_id.append(sentence2id)

	# train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
	logger.info("train_data count: %s " % (len(train_sentences_id)))
	logger.info("test_data  count:  %s " % (len(test_sentences_id)))
	logger.info("dev_data  count:  %s " % (len(dev_sentences_id)))

	train_datasets = [train_sentences_id, train_position_lambda, train_entity_tags, train_sentences_words, train_relation_tags, train_relation_names]
	test_datasets = [test_sentences_id, test_position_lambda, test_entity_tags, test_sentences_words, test_relation_tags, test_relation_names]
	dev_datasets = [dev_sentences_id, dev_position_lambda, dev_entity_tags, dev_sentences_words, dev_relation_tags, dev_relation_names]
	# print("dev_data   count: ", len(dev_data))

	# if use the pre-trained word vector
	embedding_pre = args.pretrain_vec  # data['pretrain_vec']
	dim = args.hidden_dim  # data['hidden_dim']
	statedim = args.state_dim  # data['state_dim']
	# tmp = []
	# for j in train_relation_tags:
	# 	tmp += j
	relations = set(train_relation_tags)
	logger.info(relations)
	relation_count = len(relations)  # args.relation_tag_size  # data['relation_tag_size']
	noisy_count = args.noisy_tag_size  # ata['noisy_tag_size']
	learning_rate = args.lr  # data['lr']
	l2 = args.l2  # data['l2']
	logger.info("relation count: %s " % relation_count)
	# print("Reading vector file......")
	# vec_model = KeyedVectors.load_word2vec_format(args.datapath + 'vector_demo', binary=False)  # vector_demo vector2.txt
	# vec_model = KeyedVectors.load_word2vec_format('/home/xiaoya/data/GoogleNews-vectors-negative300.bin.gz', binary=True)

	# load models
	# if args.encoder_model == "BiLSTM":
	# 	encoder = BiLSTM_LSTM.EncoderRNN(args, wv).to(device)
	# else:
	# 	encoder = BiLSTM_LSTM.EncoderBert(args).to(device)
	# decoder = BiLSTM_LSTM.DecoderRNN(args).to(device)
	JointModel = BiLSTM_LSTM.JointModel(args, wv, dim, relation_count)
	# relation_model = Jointly_RL.RelationModel(args, dim, statedim, relation_count, noisy_count)
	if args.use_RL:
		RL_model = Noisy_RL.RLModel(args.batchsize, dim)
	# RL_model = Jointly_RL.RLModel(args.batchsize, dim, statedim, relation_count, vec_model)

	criterion = nn.CrossEntropyLoss()  # ()NLLLoss
	# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
	if torch.cuda.is_available():
		# encoder = encoder.cuda()
		JointModel = JointModel.cuda()
		criterion = criterion.cuda()
		# relation_model = relation_model.cuda()
		if args.use_RL:
			RL_model = RL_model.cuda()
		# RL_model.cuda()
	out_losses = []
	# RL_RE_losses = []
	# RE_rewardsall = []
	# TOTAL_rewardsall = []
	print_loss_total = 0  # Reset every print_every
	# plot_loss_total = 0  # Reset every plot_every
	_params = filter(lambda p: p.requires_grad, JointModel.parameters())
	if args.encoder_model == 'BERT':
		optimizer = optimization.AdamW(_params, lr=learning_rate, weight_decay=l2)
	else:
		optimizer = optim.Adam(_params, lr=learning_rate, weight_decay=l2)
	logger.info(JointModel)
	logger.info("hidden_dim: %s, dropout_NER: %s, dropout_RE: %s, lr: %s, epoch: %s, batch_size: %s" % (args.hidden_dim, args.dropout_NER, args.dropout_RE, args.lr, args.epochRL, args.batchsize))
	# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)
	# RE_optimizer = optim.Adam(relation_model.parameters(), lr=learning_rate, weight_decay=l2)
	# RL_optimizer = optim.Adam(RL_model.parameters(), lr=args.lr_RL, weight_decay=l2)
	# sentence_reward_noisy = [0 for i in range(args.batchsize)]
	# noisy_sentences_vec = Variable(torch.FloatTensor(1, dim).fill_(0))
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)
	if args.do_train:
		for e in range(args.epochRL):
			# random.shuffle(train_data)
			# batchcnt = (len(train_data) - 1) // args.batchsize + 1
			# for b in range(batchcnt):
			# 	# start = time.time()
			# 	datas = train_data[b * args.batchsize: (b + 1) * args.batchsize]
			logger.info("training epoch: %s" % e)
			if args.use_RL:
				JointModel.load_state_dict(torch.load(args.best_model_path))

			train(train_datasets, mode='train')
			if e % args.save_epoch == 0 or e == args.epochRL - 1:
				try:
					model_name = args.output_dir+"model_epoch%s.pkl" % e
					torch.save(JointModel.state_dict(), model_name)
					# model_name = "./model/model_decoder_epoch%s.pkl" % e
					# torch.save(decoder, model_name)
					# model_name = "./model/relation_model_epoch%s.pkl" % e
					# torch.save(relation_model, model_name)
					# model_name = "./model/RL_model_epoch%s.pkl" % e
					# torch.save(RL_model, model_name)
					logger.info("Model has been saved")
				except Exception as e:
					logger.error(e)

			# ********************dev data*********************
			if args.do_test:
				train(dev_datasets, mode='dev')
			# ********************test data*********************
			if args.do_test:
				train(test_datasets, mode='test')
	else:
		if args.do_test:
			# JointModel = torch.load(args.best_model_path)
			# JointModel.config = args
			JointModel.load_state_dict(torch.load(args.best_model_path))
			JointModel.eval()
			train(dev_datasets, mode='dev')  # dev_datasets
			train(test_datasets, mode='test')
