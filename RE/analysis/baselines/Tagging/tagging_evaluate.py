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


from Parser import Parser
# from TFgirl.RE.PreProcess.data_manager import DataManager
from general_utils import padding_sequence, get_minibatches, padding_sequence_recurr
# import model
from tagging_main import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(encoder, decoder, input_tensor, BATCH):
	encoder_hidden = encoder.initHidden_bilstm()
	# input batch
	encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
	# for 2 layer
	h1 = torch.cat((encoder_hidden[0][0], encoder_hidden[0][1]), 1).view(1, BATCH, -1)  # concat forward and backward hidden at layer1
	h2 = torch.cat((encoder_hidden[0][2], encoder_hidden[0][3]), 1).view(1, BATCH, -1)  # layer2
	c1 = torch.cat((encoder_hidden[1][0], encoder_hidden[1][1]), 1).view(1, BATCH, -1)
	c2 = torch.cat((encoder_hidden[1][2], encoder_hidden[1][3]), 1).view(1, BATCH, -1)
	decoder_hidden = (torch.cat((h1, h2), 0), torch.cat((c1, c2), 0))  # (layer*direction, batch, hidden_dim)
	decoder_input = encoder_outputs
	decoder_output, decoder_output_tag, decoder_hidden = decoder(decoder_input, decoder_hidden)
	# RE_output = RE_model(decoder_input, decoder_hidden)
	return decoder_output_tag


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
	print("train_data count: ", len(train_sentences_id))
	print("test_data  count: ", len(test_sentences_id))
	print("dev_data  count: ", len(dev_sentences_id))

	train_datasets = [train_sentences_id, train_tags]
	test_datasets = [test_sentences_id, test_tags]
	dev_datasets = [dev_sentences_id, dev_tags]

	embedding_pre = args.pretrain_vec  # data['pretrain_vec']
	dim = args.hidden_dim  # data['hidden_dim']
	statedim = args.state_dim  # data['state_dim']

	tag_count = len(tags_dict)  # args.relation_tag_size  # data['relation_tag_size']
	noisy_count = args.noisy_tag_size  # ata['noisy_tag_size']
	learning_rate = args.lr  # data['lr']
	l2 = args.l2  # data['l2']
	print("Tags count: ", tag_count)
	# load models
	# encoder = model.EncoderRNN(args, wv).to(device)
	# decoder = model.DecoderRNN(args, wv).to(device)
	# RE_model = model.RE_RNN(args, wv, relation_count).to(device)

	encoder = torch.load(args.modelPath+"model_encoder_epoch24.pkl", map_location=device)
	decoder = torch.load(args.modelPath+"model_decoder_epoch24.pkl", map_location=device)

	if torch.cuda.is_available():
		encoder = encoder.cuda()
		decoder = decoder.cuda()
	encoder.eval()
	decoder.eval()

	# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)
	# RE_optimizer = optim.Adam(RE_model.parameters(), lr=learning_rate, weight_decay=l2)

	# ********************Train data*********************
	if args.test:
		mini_batches = get_minibatches(train_datasets, args.batchsize)
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

			RE_output = eval_model(encoder, decoder, input_tensor, args.batchsize)

			print('*****TRAIN*****seq-seq model: (%d %.2f%%)' % (b, float(b) / batchcnt * 100))

			# out_losses.append(loss_NER)
			# entity_actions = torch.max(NER_output, 2)[1]  # evaluate.sample(NER_output, False)
			action_realtion = torch.max(RE_output, 2)[1]  # torch.max(RE_output, 1)[1]  # evaluate.sample(RE_output, False)

			evaluate.cal_F_score(np.array(action_realtion.cpu()), tags, flag="TRAIN")

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

			RE_output = eval_model(encoder, decoder, input_tensor, args.batchsize)

			print('*****DEV*****seq-seq model: (%d %.2f%%)' % (b, float(b) / batchcnt * 100))

			# out_losses.append(loss_NER)
			# entity_actions = torch.max(NER_output, 2)[1]  # evaluate.sample(NER_output, False)
			action_realtion = torch.max(RE_output, 2)[1]  # torch.max(RE_output, 1)[1]  # evaluate.sample(RE_output, False)

			evaluate.cal_F_score(np.array(action_realtion.cpu()), tags,  flag="DEV")

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

			RE_output = eval_model(encoder, decoder, input_tensor, args.batchsize)

			print('*****TEST*****seq-seq model: (%d %.2f%%)' % (b, float(b) / batchcnt * 100))

			# out_losses.append(loss_NER)
			# entity_actions = torch.max(NER_output, 2)[1]  # evaluate.sample(NER_output, False)
			action_realtion = torch.max(RE_output, 2)[1]  # torch.max(RE_output, 1)[1]  # evaluate.sample(RE_output, False)

			evaluate.cal_F_score(np.array(action_realtion.cpu()), tags, flag="TEST")
