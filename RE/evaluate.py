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
from general_utils import padding_sequence, get_minibatches, padding_sequence_recurr
# import model
from main_LSTM import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(encoder, decoder, RE_model, RL_model, BATCH):
	encoder_hidden = encoder.initHidden_bilstm()
	# input batch
	encoder_outputs, encoder_hidden = encoder(input_tensor, pos_tensor, encoder_hidden)
	# for 2 layer
	h1 = torch.cat((encoder_hidden[0][0], encoder_hidden[0][1]), 1).view(1, BATCH, -1)  # concat forward and backward hidden at layer1
	h2 = torch.cat((encoder_hidden[0][2], encoder_hidden[0][3]), 1).view(1, BATCH, -1)  # layer2
	c1 = torch.cat((encoder_hidden[1][0], encoder_hidden[1][1]), 1).view(1, BATCH, -1)
	c2 = torch.cat((encoder_hidden[1][2], encoder_hidden[1][3]), 1).view(1, BATCH, -1)
	decoder_hidden = (torch.cat((h1, h2), 0), torch.cat((c1, c2), 0))  # (layer*direction, batch, hidden_dim)
	decoder_input = encoder_outputs
	decoder_output, decoder_output_tag, decoder_hidden = decoder(decoder_input, decoder_hidden)
	# RE_output = RE_model(decoder_input, decoder_hidden)
	RL_RE_loss, RE_rewards, TOTAL_rewards =	RL_model(input_tensor, encoder_outputs, decoder_output,
			 decoder_output_tag, decoder_hidden,
			 sentence_reward_noisy, noisy_sentences_vec, RE_optimizer,
			 RL_optimizer, RE_model,
			 args.sampleround, tags, sentences_words,
			 relation_tags, relation_names,
			 relation_target_tensor, criterion=0, seq_loss=0, flag="TEST")

	return RE_rewards, TOTAL_rewards


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
	# load models
	# encoder = model.EncoderRNN(args, wv).to(device)
	# decoder = model.DecoderRNN(args, wv).to(device)
	# RE_model = model.RE_RNN(args, wv, relation_count).to(device)

	encoder = torch.load(args.modelPath+"model_encoder_epoch24.pkl", map_location=device)
	decoder = torch.load(args.modelPath+"model_decoder_epoch24.pkl", map_location=device)
	RE_model = torch.load(args.modelPath+"relation_model_epoch24.pkl", map_location=device)
	RL_model = torch.load(args.modelPath+"RL_model_epoch24.pkl", map_location=device)
	# criterion = nn.NLLLoss()  # CrossEntropyLoss()
	# criterion_RE = nn.BCELoss()
	# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
	if torch.cuda.is_available():
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		RE_model = RE_model.cuda()
		RL_model = RL_model.cuda()
		# criterion = criterion.cuda()
		# criterion_RE = criterion_RE.cuda()
	encoder.eval()
	decoder.eval()
	RE_model.eval()
	RL_model.eval()

	# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)
	RE_optimizer = optim.Adam(RE_model.parameters(), lr=learning_rate, weight_decay=l2)
	RL_optimizer = optim.Adam(RL_model.parameters(), lr=args.lr_RL, weight_decay=l2)
	sentence_reward_noisy = [0 for i in range(args.batchsize)]
	noisy_sentences_vec = Variable(torch.FloatTensor(1, dim).fill_(0))
	# ********************dev data*********************
	if args.test:
		mini_batches = get_minibatches(dev_datasets, args.batchsize)
		batchcnt = len(dev_datasets[0]) // args.batchsize  # len(list(mini_batches))
		for b, data in enumerate(mini_batches):
			if b >= batchcnt:
				break
			sentences, pos_lambda, tags, sentences_words, relation_tags, relation_names = data

			input_tensor, input_length = padding_sequence(sentences, pad_token=args.embedding_size)
			pos_tensor, input_length = padding_sequence(pos_lambda, pad_token=0)
			target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
			relation_target_tensor = padding_sequence_recurr(relation_tags)
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

			RE_rewards, TOTAL_rewards = eval_model(encoder, decoder, RE_model, RL_model, args.batchsize)

			print('*****DEV*****seq-seq model: (%d %.2f%%), %s, %s' % (b, float(b) / batchcnt * 100, RE_rewards, TOTAL_rewards))

			# out_losses.append(loss_NER)
			# entity_actions = torch.max(NER_output, 2)[1]  # evaluate.sample(NER_output, False)
			# action_realtion = torch.multinomial(RE_output, 1)  # torch.max(RE_output, 1)[1]  # evaluate.sample(RE_output, False)
			#
			# evaluate.cal_F_score(np.array(action_realtion.cpu()), relation_tags, tags, np.array(entity_actions.cpu()),
			# 					 flag="DEV")

	# ********************test data*********************
	if args.test:
		mini_batches = get_minibatches(test_datasets, args.batchsize)
		batchcnt = len(test_datasets[0]) // args.batchsize  # len(list(mini_batches))
		for b, data in enumerate(mini_batches):
			if b >= batchcnt:
				break
			sentences, pos_lambda, tags, sentences_words, relation_tags, relation_names = data

			input_tensor, input_length = padding_sequence(sentences, pad_token=args.embedding_size)
			pos_tensor, input_length = padding_sequence(pos_lambda, pad_token=0)
			target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
			relation_target_tensor = padding_sequence_recurr(relation_tags)
			if torch.cuda.is_available():
				input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
				target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
				pos_tensor = Variable(torch.cuda.FloatTensor(pos_tensor, device=device)).cuda()
				relation_target_tensor = Variable(
					torch.cuda.LongTensor(relation_target_tensor, device=device)).cuda()
			else:
				input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
				target_tensor = Variable(torch.LongTensor(target_tensor, device=device))
				pos_tensor = Variable(torch.Tensor(pos_tensor, device=device))
				relation_target_tensor = Variable(torch.LongTensor(relation_target_tensor, device=device))

			RE_rewards, TOTAL_rewards = eval_model(encoder, decoder, RE_model, RL_model, args.batchsize)

			print('*****TEST*****seq-seq model: (%d %.2f%%), %s, %s' % (b, float(b) / batchcnt * 100, RE_rewards, TOTAL_rewards))

