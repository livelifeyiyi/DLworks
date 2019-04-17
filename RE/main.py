import os
import pickle
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import BiLSTM_LSTM
import Jointly_RL
from Parser import Parser
# from TFgirl.RE.PreProcess.data_manager import DataManager
from general_utils import padding_sequence, get_minibatches

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
	argv = sys.argv[1:]
	parser = Parser().getParser()
	args, _ = parser.parse_known_args(argv)
	print("Load data start...")
	# dm = DataManager(args.datapath, args.testfile)
	wv = np.loadtxt(args.datapath+'wv.txt')
	# load data from pkl file
	with open(args.datapath+'data_train.pkl', 'rb') as inp:
		train_sentences_words = pickle.load(inp)  # sentence with words
		train_sentences_id = pickle.load(inp)  # sentence with word_ids
		train_entity_tags = pickle.load(inp)  # entity tags
		train_relation_tags = pickle.load(inp)  # relation type_id
		train_relation_names = pickle.load(inp)  # relation name
	with open(args.datapath+'data_test.pkl', 'rb') as inp:
		test_sentences_words = pickle.load(inp)  # sentence with words
		test_sentences_id = pickle.load(inp)  # sentence with word_ids
		test_entity_tags = pickle.load(inp)  # entity tags
		test_relation_tags = pickle.load(inp)  # relation type_id
		test_relation_names = pickle.load(inp)  # relation name
	# train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
	print("train_data count: ", len(train_sentences_id))
	print("test_data  count: ", len(test_sentences_id))

	train_datasets = [train_sentences_id, train_entity_tags, train_sentences_words, train_relation_tags, train_relation_names]
	test_datasets = [test_sentences_id, test_entity_tags, test_sentences_words, test_relation_tags, test_relation_names]
	# print("dev_data   count: ", len(dev_data))

	# if use the pre-trained word vector
	embedding_pre = args.pretrain_vec  # data['pretrain_vec']
	dim = args.hidden_dim  # data['hidden_dim']
	statedim = args.state_dim  # data['state_dim']
	relation_count = len(set(train_relation_tags))  # args.relation_tag_size  # data['relation_tag_size']
	noisy_count = args.noisy_tag_size  # ata['noisy_tag_size']
	print("relation count: ", relation_count)

	# load models
	encoder = BiLSTM_LSTM.EncoderRNN(args, wv).to(device)
	decoder = BiLSTM_LSTM.DecoderRNN(args, wv).to(device)
	relation_model = Jointly_RL.RelationModel(dim, statedim, relation_count, noisy_count)

	criterion = nn.NLLLoss()  # CrossEntropyLoss()
	# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
	if torch.cuda.is_available():
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		criterion = criterion.cuda()
		relation_model = relation_model.cuda()
	out_losses = []
	RL_RE_losses = []
	print_loss_total = 0  # Reset every print_every
	# plot_loss_total = 0  # Reset every plot_every
	learning_rate = args.lr  # data['lr']
	l2 = args.l2  # data['l2']
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)

	for e in range(args.epochRL):
		print("training epoch ", e)

		# random.shuffle(train_data)
		# batchcnt = (len(train_data) - 1) // args.batchsize + 1
		# for b in range(batchcnt):
		# 	# start = time.time()
		# 	datas = train_data[b * args.batchsize: (b + 1) * args.batchsize]

		mini_batches = get_minibatches(train_datasets, args.batchsize)
		batchcnt = len(train_datasets[0]) // args.batchsize  # len(list(mini_batches))
		for b, data in enumerate(mini_batches):
			if b == batchcnt:
				break
			sentences, tags, sentences_words, relation_tags, relation_names = data

			input_tensor, input_length = padding_sequence(sentences, pad_token=args.embedding_size)
			target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
			if torch.cuda.is_available():
				input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
				target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
			else:
				input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
				target_tensor = Variable(torch.LongTensor(target_tensor, device=device))

			seq_loss, encoder_outputs, decoder_output, decoder_output_tag = BiLSTM_LSTM.train(input_tensor, target_tensor, encoder,
						decoder, encoder_optimizer, decoder_optimizer, criterion, args.batchsize)  # , input_length, target_length
			out_losses.append(seq_loss)
			# print_loss_total += seq_loss
			# # plot_loss_total += loss
			# print_every = 10
			# if (b+1) % print_every == 0:
			# 	print_loss_avg = print_loss_total / (print_every*b//print_every)
			# 	print_loss_total = 0
			print('seq-seq model: (%d %.2f%%), loss: %.4f' % (b, float(b) / batchcnt * 100, seq_loss))

			RL_model = Jointly_RL.RLModel(input_tensor, encoder_outputs, decoder_output, decoder_output_tag, dim, statedim, relation_count,
										learning_rate, relation_model, args.datapath)
			if torch.cuda.is_available():
				RL_model.cuda()

			print("Training RL based RE......")
			RL_RE_loss = RL_model(args.sampleround, tags, sentences_words, relation_tags, relation_names, seq_loss)
			RL_RE_losses.append(RL_RE_loss)
		np.save("seq2seq_loss_train", out_losses)
		np.save("RL_RE_loss_train", RL_RE_losses)

		# test data
		if args.test:
			mini_batches = get_minibatches(test_datasets, args.batchsize)
			batchcnt = len(test_datasets[0]) // args.batchsize  # len(list(mini_batches))
			for b, data in enumerate(mini_batches):
				if b == batchcnt:
					break
				sentences, tags, sentences_words, relation_tags, relation_names = data

				input_tensor, input_length = padding_sequence(sentences, pad_token=args.embedding_size)
				target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
				if torch.cuda.is_available():
					input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
					target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
				else:
					input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
					target_tensor = Variable(torch.LongTensor(target_tensor, device=device))

				seq_loss, encoder_outputs, decoder_output, decoder_output_tag = BiLSTM_LSTM.train(input_tensor,
																								  target_tensor, encoder,
																								  decoder,
																								  encoder_optimizer,
																								  decoder_optimizer,
																								  criterion,
																								  args.batchsize, TEST=True)  # , input_length, target_length
				# out_losses.append(seq_loss)
				# print_loss_total += seq_loss
				# plot_loss_total += loss
				# print_every = 10
				# if (b + 1) % print_every == 0:
				# 	print_loss_avg = print_loss_total / (print_every * b // print_every)
				# 	print_loss_total = 0
				# 	print('TEST***seq-seq model: (%d %d%%) %.4f' % (b, float(b) / batchcnt * 100, print_loss_avg))

				RL_model = Jointly_RL.RLModel(input_tensor, encoder_outputs, decoder_output, decoder_output_tag, dim,
												statedim, relation_count, learning_rate, relation_model, args.datapath)
				if torch.cuda.is_available():
					RL_model.cuda()

				# print("Testing RL based RE......")
				RL_RE_loss = RL_model(args.sampleround, tags, sentences_words, relation_tags, relation_names, seq_loss, TEST=True)
				# RL_RE_losses.append(RL_RE_loss)
			# np.save("seq2seq_loss_test", out_losses)
			# np.save("RL_RE_loss_test", RL_RE_losses)