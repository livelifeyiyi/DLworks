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


from TFgirl.RE.Parser import Parser  # TFgirl.RE.
# from TFgirl.RE.PreProcess.data_manager import DataManager
from TFgirl.RE.general_utils import padding_sequence, get_minibatches, padding_sequence_recurr
import Tagging_model


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class evaluate():
	def __init__(self, batchsize, tags_dict):
		self.batch_size = batchsize
		self.acc = 0.
		self.cnt = 0.
		self.tot = 0.
		self.tags_dict = tags_dict

	def calc_acc_total(self, relation_action, labels):
		# acc, cnt, tot = 0, 0, len(relation_labels)
		self.tot += len(labels)
		cnt = 0.
		# used = [0 for i in range(len(relation_action))]
		# tp, tags = label, label['tags']
		j, ok = 0, 0
		# for i in range(len(relation_action)):  # each round
		if isinstance(relation_action, np.int64):
			for label in labels:
				if label == relation_action and ok == 0 and label > 0:  # relation_action[i] == label and used[i] == 0
					match = 1
					if match == 1:
						ok = 1
				self.acc += ok
			if relation_action > 0:
				cnt += 1
			self.cnt += cnt
		else:
			match = []
			if len(labels) < len(relation_action):
				for idx, label in enumerate(labels):
					if label != self.tags_dict["O"]:  # not "O" tag
						if label == relation_action[idx]:  # relation_action[i] == label and used[i] == 0
							match.append(1)
						else:
							match.append(0)
			else:
				for idx, label in enumerate(relation_action):
					if labels[idx] != self.tags_dict["O"]:  # not "O" tag
						if label == labels[idx]:  # relation_action[i] == label and used[i] == 0
							match.append(1)
						else:
							match.append(0)
			if sum(match) == len(match):
				ok = 1
			self.acc += ok

		self.cnt += len(labels)
		return self.acc, self.tot, self.cnt

	def cal_F_score(self, relation_actions_batch, train_tags, flag):
		batch_size = self.batch_size  # len(relation_actions_batch)
		# cal the P,R and F of relation extraction for a batch of sentences
		for sentence_id in range(batch_size):
			acc_total, tot_total, cnt_total = self.calc_acc_total(relation_actions_batch[sentence_id],
																  train_tags[sentence_id])

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
		with codecs.open("./Tagging_model/"+flag+"_TOTAL.out", mode='a+', encoding='utf-8') as f1:
			f1.write(line_total)

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

	# load data from pkl file
	with open(args.datapath+'BSL_Tagging/data_train.pkl', 'rb') as inp:
		train_sentences_id = pickle.load(inp)  # sentence with word_ids
		train_tags = pickle.load(inp)  # tags
		w2id = pickle.load(inp)  # word2id
		relation2id = pickle.load(inp)  # relation2id
		tags_dict = pickle.load(inp)  # tags dictionary

	with open(args.datapath+'BSL_Tagging/data_test.pkl', 'rb') as inp:
		test_sentences_id = pickle.load(inp)  # sentence with word_ids
		test_tags = pickle.load(inp)  # tags

	with open(args.datapath + 'BSL_Tagging/data_dev.pkl', 'rb') as inp:
		dev_sentences_id = pickle.load(inp)  # sentence with words
		dev_tags = pickle.load(inp)  # tags
	evaluate = evaluate(args.batchsize, tags_dict)
	# train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
	print("train_data count: ", len(train_sentences_id))
	print("test_data  count: ", len(test_sentences_id))
	print("dev_data  count: ", len(dev_sentences_id))

	train_datasets = [train_sentences_id, train_tags]
	test_datasets = [test_sentences_id, test_tags]
	dev_datasets = [dev_sentences_id, dev_tags]
	# print("dev_data   count: ", len(dev_data))

	embedding_pre = args.pretrain_vec  # data['pretrain_vec']
	dim = args.hidden_dim  # data['hidden_dim']
	statedim = args.state_dim  # data['state_dim']

	tag_count = len(tags_dict)  # args.relation_tag_size  # data['relation_tag_size']
	noisy_count = args.noisy_tag_size  # ata['noisy_tag_size']
	learning_rate = args.lr  # data['lr']
	l2 = args.l2  # data['l2']
	print("Tags count: ", tag_count)
	# print("Reading vector file......")
	# vec_model = KeyedVectors.load_word2vec_format(args.datapath + 'vector2.txt', binary=False)
	# vec_model = KeyedVectors.load_word2vec_format('/home/xiaoya/data/GoogleNews-vectors-negative300.bin.gz', binary=True)

	# load models
	encoder = Tagging_model.EncoderRNN(args, wv).to(device)
	decoder = Tagging_model.DecoderRNN(args, wv, tag_count).to(device)

	criterion = nn.NLLLoss()  # CrossEntropyLoss()
	# criterion_RE = nn.BCELoss()
	# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
	if torch.cuda.is_available():
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		criterion = criterion.cuda()
		# criterion_RE = criterion_RE.cuda()

	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)

	for e in range(args.epochRL):
		print("training epoch ", e)

		mini_batches = get_minibatches(train_datasets, args.batchsize)
		batchcnt = len(train_datasets[0]) // args.batchsize  # len(list(mini_batches))
		for b, data in enumerate(mini_batches):
			if b >= batchcnt:
				break
			sentences, tags = data
			input_tensor, input_length = padding_sequence(sentences, pad_token=args.embedding_size)
			target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
			if torch.cuda.is_available():
				input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
				target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
			else:
				input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
				target_tensor = Variable(torch.LongTensor(target_tensor, device=device))

			loss_RE, RE_output = Tagging_model.train(input_tensor, target_tensor,
					encoder, decoder,
					encoder_optimizer, decoder_optimizer,
					criterion, args.batchsize, TEST=False)

			print('seq-seq RE model: (%d %.2f%%), loss: %.4f' % (b, float(b) / batchcnt * 100, loss_RE))

			# out_losses.append(loss_NER)
			# entity_actions = torch.max(NER_output, 2)[1]  # evaluate.sample(NER_output, False)
			action_realtion = torch.max(RE_output, 2)[1]  # torch.multinomial(RE_output, 1)  # torch.max(RE_output, 1)[1]  # evaluate.sample(RE_output, False)

			evaluate.cal_F_score(np.array(action_realtion.cpu()), tags, flag="TRAIN")

		if e % 10 == 0 or e == args.epochRL-1:
			try:
				model_name = "./Tagging_model/model_encoder_epoch%s.pkl" % e
				torch.save(encoder, model_name)
				model_name = "./Tagging_model/model_decoder_epoch%s.pkl" % e
				torch.save(decoder, model_name)
				print("Model has been saved")
			except Exception as e:
				print(e)
		# ********************dev data*********************
		if args.test:
			mini_batches = get_minibatches(dev_datasets, args.batchsize)
			batchcnt = len(dev_datasets[0]) // args.batchsize  # len(list(mini_batches))
			for b, data in enumerate(mini_batches):
				if b >= batchcnt:
					break
				sentences, tags = data
				input_tensor, input_length = padding_sequence(sentences, pad_token=args.embedding_size)
				target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
				if torch.cuda.is_available():
					input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
					target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
				else:
					input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
					target_tensor = Variable(torch.LongTensor(target_tensor, device=device))

				RE_output = Tagging_model.train(input_tensor, target_tensor,
														 encoder, decoder,
														 encoder_optimizer, decoder_optimizer,
														 criterion, args.batchsize, TEST=True)
				print('*****DEV*****seq-seq model: (%d %.2f%%)' % (b, float(b) / batchcnt * 100))

				# out_losses.append(loss_NER)
				# entity_actions = torch.max(NER_output, 2)[1]  # evaluate.sample(NER_output, False)
				action_realtion = torch.max(RE_output, 1)[1]  # evaluate.sample(RE_output, False)

				evaluate.cal_F_score(np.array(action_realtion.cpu()), tags, flag="DEV")

		# ********************test data*********************
		if args.test:
			mini_batches = get_minibatches(test_datasets, args.batchsize)
			batchcnt = len(test_datasets[0]) // args.batchsize  # len(list(mini_batches))
			for b, data in enumerate(mini_batches):
				if b >= batchcnt:
					break
				sentences, tags = data
				input_tensor, input_length = padding_sequence(sentences, pad_token=args.embedding_size)
				target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
				if torch.cuda.is_available():
					input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
					target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
				else:
					input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
					target_tensor = Variable(torch.LongTensor(target_tensor, device=device))

				RE_output = Tagging_model.train(input_tensor, target_tensor,
												encoder, decoder,
												encoder_optimizer, decoder_optimizer,
												criterion, args.batchsize, TEST=True)

				print('*****TEST*****seq-seq model: (%d %.2f%%)' % (b, float(b) / batchcnt * 100))

				# out_losses.append(loss_NER)
				# entity_actions = torch.max(NER_output, 2)[1]  # evaluate.sample(NER_output, False)
				action_realtion = torch.max(RE_output, 1)[1]  # evaluate.sample(RE_output, False)

				evaluate.cal_F_score(np.array(action_realtion.cpu()), tags, flag="TEST")
