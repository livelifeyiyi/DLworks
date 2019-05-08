import codecs
import os
import pickle
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from gensim.models import KeyedVectors


from Parser import Parser
# from TFgirl.RE.PreProcess.data_manager import DataManager
from general_utils import padding_sequence, get_minibatches
import model


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class evaluate():
	def __init__(self, batchsize):
		self.batch_size = batchsize
		self.acc = 0.
		self.cnt = 0.
		self.tot = 0.

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
				if label == relation_action  and ok == 0 and label > 0:  # relation_action[i] == label and used[i] == 0
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
			if relation_action > 0:
				cnt += 1
				self.acc += ok
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
					# used[i] = 1

			for i in range(len(relation_action)):
				if relation_action[i] > 0:
					# j += 1
					cnt += 1
					self.acc += ok
		self.cnt += cnt // len(relation_labels)
		return self.acc, self.tot, self.cnt

	def cal_F_score(self, relation_actions_batch, train_relation_tags, train_entity_tags, entity_actions_batch, flag):
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
			if isinstance(relation_actions_batch[sentence_id], np.int64):
				round_num = 1
			else:
				round_num = len(relation_actions_batch[sentence_id])
			acc_total, tot_total, cnt_total = self.calc_acc_total(relation_actions_batch[sentence_id],
																  entity_actions_batch[sentence_id],
																  train_relation_tags[sentence_id],
																  train_entity_tags[sentence_id])

			if round_num > 1:
				for i in range(round_num):
					if int(relation_actions_batch[sentence_id][i]) in train_relation_tags[sentence_id]:
						acc_R += 1
					if int(relation_actions_batch[sentence_id][i]) > 0:
						cnt_R += 1
					tot_R += 1
			else:
				if relation_actions_batch[sentence_id] in train_relation_tags[sentence_id]:
					acc_R += 1
				if relation_actions_batch[sentence_id] > 0:
					cnt_R += 1

			tot_R_relation_num += len(train_relation_tags[sentence_id])
			cnt_R_last += round_num // len(train_relation_tags[sentence_id])
			for each_relation in train_relation_tags[sentence_id]:
				if each_relation > 0:
					tot_R += 1
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

		precision_total = acc_total / cnt_total
		recall_total = acc_total / tot_total
		beta = 1.
		try:
			F_total = (1 + beta * beta) * precision_total * recall_total / (beta * beta * precision_total + recall_total)
		except Exception as e:
			print(e)
			F_total = 0.
		print("********: TOTAL precision: " + str(precision_total) + ", recall: " + str(recall_total) + ", F-score: " + str(
			F_total))
		line_total = str(precision_total) + ", " + str(recall_total) + ", " + str(F_total) + "\n"
		with codecs.open("./" + flag + "_TOTAL.out", mode='a+', encoding='utf-8') as f1:
			f1.write(line_total)

		precision_R = acc_R / cnt_R
		# recall = acc/round_num/tot
		recall_R = rec_R / tot_R
		beta = 1.
		try:
			F_RE = (1 + beta * beta) * precision_R * recall_R / (beta * beta * precision_R + recall_R)
		except Exception as e:
			print(e)
			F_RE = 0.
		print("********: Relation precision: " + str(acc_R / tot_R_relation_num) + ", " + str(
			acc_R / cnt_R_last) + ", " + str(precision_R) +
			  ", recall: " + str(rec_R / tot_R_relation_num) + ", " + str(recall_R) + ", F-score: " + str(F_RE))

		line_RE = str(acc_R / tot_R_relation_num) + ", " + str(acc_R / cnt_R_last) + ", " + str(
			precision_R) + ",	" + str(rec_R / tot_R_relation_num) + ",	" + str(recall_R) \
				  + ",	" + str(F_RE) + '\n'
		with codecs.open("./" + flag + "_RE.out", mode='a+', encoding='utf-8') as f1:
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
		with codecs.open("./" + flag + "_NER.out", mode='a+', encoding='utf-8') as f2:
			f2.write(line_NER)

	def sample(self, prob, training, position=None, preoptions=None):
		if not training:
			return torch.max(prob, 1)[1]  # prob, 0
		elif preoptions is not None:
			return Variable(torch.cuda.LongTensor(1, ).fill_(preoptions[position]))
		else:
			return torch.multinomial(prob, 1)  # torch.max(prob, 0)[1]  #


if __name__ == "__main__":
	argv = sys.argv[1:]
	parser = Parser().getParser()
	args, _ = parser.parse_known_args(argv)
	print("Load data start...")
	# dm = DataManager(args.datapath, args.testfile)
	wv = np.loadtxt(args.datapath+'wv.txt')
	evaluate = evaluate(args.batchsize)
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

	# train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
	print("train_data count: ", len(train_sentences_id))
	print("test_data  count: ", len(test_sentences_id))
	print("dev_data  count: ", len(dev_sentences_id))

	train_datasets = [train_sentences_id, train_position_lambda, train_entity_tags, train_sentences_words, train_relation_tags, train_relation_names]
	test_datasets = [test_sentences_id, test_position_lambda, test_entity_tags, test_sentences_words, test_relation_tags, test_relation_names]
	dev_datasets = [dev_sentences_id, dev_position_lambda, dev_entity_tags, dev_sentences_words, dev_relation_tags, dev_relation_names]
	# print("dev_data   count: ", len(dev_data))

	embedding_pre = args.pretrain_vec  # data['pretrain_vec']
	dim = args.hidden_dim  # data['hidden_dim']
	statedim = args.state_dim  # data['state_dim']
	tmp = []
	for j in train_relation_tags:
		tmp += j
	relations = set(tmp)
	print(relations)
	relation_count = len(relations)  # args.relation_tag_size  # data['relation_tag_size']
	noisy_count = args.noisy_tag_size  # ata['noisy_tag_size']
	learning_rate = args.lr  # data['lr']
	l2 = args.l2  # data['l2']
	print("relation count: ", relation_count)
	print("Reading vector file......")
	vec_model = KeyedVectors.load_word2vec_format(args.datapath + 'vector2.txt', binary=False)
	# vec_model = KeyedVectors.load_word2vec_format('/home/xiaoya/data/GoogleNews-vectors-negative300.bin.gz', binary=True)

	# load models
	encoder = model.EncoderRNN(args, wv).to(device)
	decoder = model.DecoderRNN(args, wv).to(device)
	RE_model = model.RE_RNN(args, wv, relation_count).to(device)

	criterion = nn.NLLLoss()  # CrossEntropyLoss()
	# criterion_RE = nn.BCELoss()
	# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
	if torch.cuda.is_available():
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		RE_model = RE_model.cuda()
		criterion = criterion.cuda()
		# criterion_RE = criterion_RE.cuda()

	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)
	RE_optimizer = optim.Adam(RE_model.parameters(), lr=learning_rate, weight_decay=l2)

	for e in range(args.epochRL):
		print("training epoch ", e)

		mini_batches = get_minibatches(train_datasets, args.batchsize)
		batchcnt = len(train_datasets[0]) // args.batchsize  # len(list(mini_batches))
		for b, data in enumerate(mini_batches):
			if b >= batchcnt:
				break
			sentences, pos_lambda, tags, sentences_words, relation_tags, relation_names = data

			input_tensor, input_length = padding_sequence(sentences, pad_token=args.embedding_size)
			pos_tensor, input_length = padding_sequence(pos_lambda, pad_token=0)
			target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
			relation_target_tensor, relation_length = padding_sequence(relation_tags, pad_token=0)
			if torch.cuda.is_available():
				input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
				target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
				pos_tensor = Variable(torch.cuda.FloatTensor(pos_tensor, device=device)).cuda()
				relation_target_tensor = Variable(torch.cuda.LongTensor(relation_target_tensor, device=device)).cuda()
			else:
				input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
				target_tensor = Variable(torch.LongTensor(target_tensor, device=device))
				pos_tensor = Variable(torch.Tensor(pos_tensor, device=device))
				relation_target_tensor = Variable(torch.LongTensor(relation_target_tensor, device=device))

			loss_NER, loss_RE, NER_output, RE_output = model.train(input_tensor, pos_tensor, target_tensor, relation_target_tensor, encoder, decoder, RE_model,
					encoder_optimizer, decoder_optimizer, RE_optimizer,
					criterion, args.batchsize, TEST=False)

			print('seq-seq NER model: (%d %.2f%%), loss: %.4f' % (b, float(b) / batchcnt * 100, loss_NER))
			print('seq-seq RE model: (%d %.2f%%), loss: %.4f' % (b, float(b) / batchcnt * 100, loss_RE))

			# out_losses.append(loss_NER)
			entity_actions = torch.max(NER_output, 2)[1]  # evaluate.sample(NER_output, False)
			action_realtion = torch.max(RE_output, 1)[1]  # evaluate.sample(RE_output, False)

			evaluate.cal_F_score(np.array(action_realtion), relation_tags, tags, np.array(entity_actions), flag="TRAIN")


