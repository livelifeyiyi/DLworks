import os
import pickle
import random
import sys
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
from general_utils import padding_sequence, get_minibatches, padding_sequence_recurr
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_transformers.tokenization_bert import BertTokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(e, datasets, optimizer, criterion, args):
	mini_batches = get_minibatches(datasets, args.batchsize)
	batchcnt = len(datasets[0]) // args.batchsize  # len(list(mini_batches))
	print("training epoch: %s, number of batches: %s" % (e, batchcnt))
	NER_correct, NER_total = 0., 0.
	RE_correct, RE_total = 0., 0.
	# NER_target_all, NER_output_all = None, None
	# RE_target_all, RE_output_all = None, None

	for b, data in enumerate(mini_batches):
		if b >= batchcnt:
			break
		sentences, pos_lambda, tags, sentences_words, relation_tags, relation_names = data
		input_tensor, input_length = padding_sequence(sentences, pad_token=0)
		pos_tensor, input_length = padding_sequence(pos_lambda, pad_token=0)
		target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
		relation_target_tensor = padding_sequence_recurr(relation_tags)
		if torch.cuda.is_available():
			input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
			mask = torch.cuda.ByteTensor((1 - (input_tensor == 0))).to(device)
			target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
			pos_tensor = Variable(torch.cuda.FloatTensor(pos_tensor, device=device)).cuda()
			relation_target_tensor = Variable(torch.cuda.LongTensor(relation_target_tensor, device=device)).cuda()
		else:
			input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
			mask = torch.ByteTensor((1 - (input_tensor == 0)))

			target_tensor = Variable(torch.LongTensor(target_tensor, device=device))
			pos_tensor = Variable(torch.Tensor(pos_tensor, device=device))
			relation_target_tensor = Variable(torch.LongTensor(relation_target_tensor, device=device))

		seq_loss, NER_active_logits, NER_active_labels, RE_output_tag, relation_target_tensor = BiLSTM_LSTM.train(
								input_tensor, pos_tensor, target_tensor,
								relation_target_tensor, JointModel, optimizer, criterion, args.batchsize, False,
								mask)  # , input_length, target_length

		NER_correct += (torch.argmax(NER_active_logits, -1) == NER_active_labels).sum().item()
		NER_total += len(NER_active_logits)
		# if NER_target_all is None:
		# 	NER_target_all = NER_active_labels
		# 	NER_output_all = NER_active_logits
		# else:
		# 	NER_target_all = torch.cat((NER_target_all, NER_active_labels), dim=0)
		# 	NER_output_all = torch.cat((NER_output_all, NER_active_logits), dim=0)
		# for i in range(len(relation_target_tensor[0])):
		# 	target = torch.transpose(relation_target_tensor, 0, 1)[i]
		# 	RE_correct += (torch.argmax(RE_output_tag, -1) == target).sum().item()
		RE_correct /= len(relation_target_tensor[0])
		RE_total += len(RE_output_tag)
		# if RE_target_all is None:
		# 	RE_target_all = relation_target_tensor
		# 	RE_output_all = RE_output_tag
		# else:
		# 	RE_target_all = torch.cat((RE_target_all, relation_target_tensor), dim=0)
		# 	RE_output_all = torch.cat((RE_output_all, RE_output_tag), dim=0)

		out_losses.append(seq_loss)
		# print_loss_total += seq_loss
		# # plot_loss_total += loss
		# print_every = 10
		# if (b+1) % print_every == 0:
		# 	print_loss_avg = print_loss_total / (print_every*b//print_every)
		# 	print_loss_total = 0
		print('seq-seq model: (%d %.2f%%), loss: %.4f, NER acc: %.4f, RE acc: %.4f' %
			  (b, float(b) / batchcnt * 100, seq_loss, NER_correct / NER_total, RE_correct / RE_total))

	# print("Training RL based RE......")
	# sentences, encoder_output, decoder_output, decoder_output_prob, round_num, train_entity_tags,
	# train_sentences_words, train_relation_tags, train_relation_names, seq_loss, TEST=False
	# input_tensor, encoder_outputs, decoder_output, decoder_output_tag,
	# RL_model = Jointly_RL.RLModel(input_tensor, encoder_outputs, decoder_output, decoder_output_tag,
	# 							  args.batchsize_test,
	# 							  dim, statedim, relation_count,
	# 							  vec_model)
	# if torch.cuda.is_available():
	# 	RL_model.cuda()

	# print("Testing RL based RE......")
	# RL_RE_loss, RE_rewards, TOTAL_rewards = RL_model(input_tensor, encoder_outputs, decoder_output,
	# 												 decoder_output_tag, decoder_hidden,
	# 												 sentence_reward_noisy, noisy_sentences_vec, RE_optimizer,
	# 												 RL_optimizer, relation_model,
	# 												 args.sampleround, tags, sentences_words,
	# 												 relation_tags, relation_names,
	# 												 relation_target_tensor, criterion, seq_loss)  #  relation_target_tensor, criterion
	# RL_RE_losses.append(RL_RE_loss)
	# RE_rewardsall.append(RE_rewards)
	# TOTAL_rewardsall.append(TOTAL_rewards)
	# NER_pred_res = metrics.classification_report(NER_target_all.cpu(), torch.argmax(NER_output_all, -1).cpu(),
	# 							  target_names=['O', 'S_I', 'T_I', 'O_I', 'S_B', 'T_B', 'O_B'])
	# print('NER Prediction results: \n{}'.format(NER_pred_res))
	# RE_pred_res = metrics.classification_report(RE_target_all.cpu(), torch.argmax(RE_output_all, -1).cpu())
	# print('RE Prediction results: \n{}'.format(RE_pred_res))

	np.save("loss_train", out_losses)
	# np.save("RL_RE_loss_train", RL_RE_losses)
	# np.save("RE_rewards_train", RE_rewardsall)
	# np.save("TOTAL_rewards_train", TOTAL_rewardsall)

	if e % 10 == 0 or e == args.epochRL - 1:
		try:
			model_name = "./model/model_epoch%s.pkl" % e
			torch.save(JointModel, model_name)
			# model_name = "./model/model_decoder_epoch%s.pkl" % e
			# torch.save(decoder, model_name)
			# model_name = "./model/relation_model_epoch%s.pkl" % e
			# torch.save(relation_model, model_name)
			# model_name = "./model/RL_model_epoch%s.pkl" % e
			# torch.save(RL_model, model_name)
			print("Model has been saved")
		except Exception as e:
			print(e)


def test(model, datasets, mode):
	model.eval()
	mini_batches = get_minibatches(datasets, args.batchsize_test)
	batchcnt = len(datasets[0]) // args.batchsize_test  # len(list(mini_batches))
	print("********************%s data*********************"%mode)
	print("epoch: %s, number of batches: %s" % (e, batchcnt))
	NER_correct, NER_total = 0., 0.
	RE_correct, RE_total = 0., 0.
	NER_target_all, NER_output_all = None, None
	RE_target_all, RE_output_all = None, None
	for b, data in enumerate(mini_batches):
		if b >= batchcnt:
			break
		sentences, pos_lambda, tags, sentences_words, relation_tags, relation_names = data
		input_tensor, input_length = padding_sequence(sentences, pad_token=0)
		pos_tensor, input_length = padding_sequence(pos_lambda, pad_token=0)
		target_tensor, target_length = padding_sequence(tags, pad_token=args.entity_tag_size)
		relation_target_tensor = padding_sequence_recurr(relation_tags)
		if torch.cuda.is_available():
			input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
			mask = torch.cuda.ByteTensor((1 - (input_tensor == 0))).to(device)
			target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
			pos_tensor = Variable(torch.cuda.FloatTensor(pos_tensor, device=device)).cuda()
			relation_target_tensor = Variable(
				torch.cuda.LongTensor(relation_target_tensor, device=device)).cuda()
		else:
			input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
			mask = torch.ByteTensor((1 - (input_tensor == 0)))

			target_tensor = Variable(torch.LongTensor(target_tensor, device=device))
			pos_tensor = Variable(torch.Tensor(pos_tensor, device=device))
			relation_target_tensor = Variable(torch.LongTensor(relation_target_tensor, device=device))

		NER_active_logits, NER_active_labels, RE_output_tag, relation_target_tensor = BiLSTM_LSTM.train(
							input_tensor, pos_tensor, target_tensor,
							relation_target_tensor, model, optimizer, criterion, args.batchsize, True,
							mask)  # , input_length, target_length

		NER_correct += (torch.argmax(NER_active_logits, -1) == NER_active_labels).sum().item()
		NER_total += len(NER_active_logits)
		if NER_target_all is None:
			NER_target_all = NER_active_labels
			NER_output_all = NER_active_logits
		else:
			NER_target_all = torch.cat((NER_target_all, NER_active_labels), dim=0)
			NER_output_all = torch.cat((NER_output_all, NER_active_logits), dim=0)
		for i in range(len(relation_target_tensor[0])):
			target = torch.transpose(relation_target_tensor, 0, 1)[i]
			RE_correct += (torch.argmax(RE_output_tag, -1) == target).sum().item()
		RE_correct /= len(relation_target_tensor[0])
		RE_total += len(RE_output_tag)
		if RE_target_all is None:
			RE_target_all = relation_target_tensor
			RE_output_all = RE_output_tag
		else:
			RE_target_all = torch.cat((RE_target_all, relation_target_tensor), dim=0)
			RE_output_all = torch.cat((RE_output_all, RE_output_tag), dim=0)

		print('seq-seq model: (%d %.2f%%), NER acc: %.4f, RE acc: %.4f' %
			  (b, float(b) / batchcnt * 100, NER_correct / NER_total, RE_correct / RE_total))

	NER_pred_res = metrics.classification_report(NER_target_all.cpu(), torch.argmax(NER_output_all, -1).cpu(),
												 target_names=['O', 'S_I', 'T_I', 'O_I', 'S_B', 'T_B', 'O_B'])
	print('NER Prediction results: \n{}'.format(NER_pred_res))
	RE_pred_res = metrics.classification_report(RE_target_all.cpu(), torch.argmax(RE_output_all, -1).cpu())
	print('RE Prediction results: \n{}'.format(RE_pred_res))


if __name__ == "__main__":
	argv = sys.argv[1:]
	parser = Parser().getParser()
	args, _ = parser.parse_known_args(argv)
	print("Load data start...")
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
		train_sentences_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_sentences_words))
	# train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
	print("train_data count: ", len(train_sentences_id))
	print("test_data  count: ", len(test_sentences_id))
	print("dev_data  count: ", len(dev_sentences_id))

	train_datasets = [train_sentences_id, train_position_lambda, train_entity_tags, train_sentences_words, train_relation_tags, train_relation_names]
	test_datasets = [test_sentences_id, test_position_lambda, test_entity_tags, test_sentences_words, test_relation_tags, test_relation_names]
	dev_datasets = [dev_sentences_id, dev_position_lambda, dev_entity_tags, dev_sentences_words, dev_relation_tags, dev_relation_names]
	# print("dev_data   count: ", len(dev_data))

	# if use the pre-trained word vector
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
	vec_model = KeyedVectors.load_word2vec_format(args.datapath + 'vector_demo', binary=False)  # vector_demo vector2.txt
	# vec_model = KeyedVectors.load_word2vec_format('/home/xiaoya/data/GoogleNews-vectors-negative300.bin.gz', binary=True)

	# load models
	# if args.encoder_model == "BiLSTM":
	# 	encoder = BiLSTM_LSTM.EncoderRNN(args, wv).to(device)
	# else:
	# 	encoder = BiLSTM_LSTM.EncoderBert(args).to(device)
	# decoder = BiLSTM_LSTM.DecoderRNN(args).to(device)
	JointModel = BiLSTM_LSTM.JointModel(args, wv, dim, relation_count)
	# relation_model = Jointly_RL.RelationModel(args, dim, statedim, relation_count, noisy_count)
	# RL_model = Jointly_RL.RLModel(args.batchsize, dim, statedim, relation_count, learning_rate, relation_model, args.datapath)
	# RL_model = Jointly_RL.RLModel(args.batchsize, dim, statedim, relation_count, vec_model)

	criterion = nn.CrossEntropyLoss()  # ()NLLLoss
	# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
	if torch.cuda.is_available():
		# encoder = encoder.cuda()
		JointModel = JointModel.cuda()
		criterion = criterion.cuda()
		# relation_model = relation_model.cuda()
		# RL_model = RL_model.cuda()
		# RL_model.cuda()
	out_losses = []
	# RL_RE_losses = []
	# RE_rewardsall = []
	# TOTAL_rewardsall = []
	print_loss_total = 0  # Reset every print_every
	# plot_loss_total = 0  # Reset every plot_every
	_params = filter(lambda p: p.requires_grad, JointModel.parameters())
	optimizer = optim.Adam(_params, lr=learning_rate, weight_decay=l2)
	# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)
	# RE_optimizer = optim.Adam(relation_model.parameters(), lr=learning_rate, weight_decay=l2)
	# RL_optimizer = optim.Adam(RL_model.parameters(), lr=args.lr_RL, weight_decay=l2)
	# sentence_reward_noisy = [0 for i in range(args.batchsize)]
	# noisy_sentences_vec = Variable(torch.FloatTensor(1, dim).fill_(0))
	for e in range(args.epochRL):
		# random.shuffle(train_data)
		# batchcnt = (len(train_data) - 1) // args.batchsize + 1
		# for b in range(batchcnt):
		# 	# start = time.time()
		# 	datas = train_data[b * args.batchsize: (b + 1) * args.batchsize]
		train(e, train_datasets, optimizer, criterion, args)

		# ********************dev data*********************
		if args.test:
			test(JointModel, dev_datasets, mode='dev')

	# ********************test data*********************
	if args.test:
		test(JointModel, test_datasets, mode='test')