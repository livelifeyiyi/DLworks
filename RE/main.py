import random, sys, time, os
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

from TFgirl.RE import Jointly_RL
from TFgirl.RE import BiLSTM_LSTM
from TFgirl.RE.general_utils import padding_sequence
from TFgirl.RE.data_manager import DataManager
from TFgirl.RE.Parser import Parser


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
	argv = sys.argv[1:]
	parser = Parser().getParser()
	args, _ = parser.parse_known_args(argv)
	print("Load data start...")
	dm = DataManager(args.datapath, args.testfile)
	wv = dm.vector

	train_data, test_data, dev_data = dm.data['train'], dm.data['test'], dm.data['dev']
	print("train_data count: ", len(train_data))
	print("test_data  count: ", len(test_data))
	print("dev_data   count: ", len(dev_data))

	# load models
	encoder = BiLSTM_LSTM.EncoderRNN(config, embedding_pre).to(device)
	decoder = BiLSTM_LSTM.DecoderRNN(config, embedding_pre).to(device)
	relation_model = Jointly_RL.RelationModel(dim, statedim, relation_count, noisy_count)

	criterion = nn.NLLLoss()  # CrossEntropyLoss()
	# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
	if torch.cuda.is_available():
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		criterion = criterion.cuda()
		relation_model = relation_model.cuda()
	out_losses = []
	print_loss_total = 0  # Reset every print_every
	# plot_loss_total = 0  # Reset every plot_every

	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2)  # SGD
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=l2)

	for e in range(args.epochRL):
		random.shuffle(train_data)
		print("training epoch ", e)
		batchcnt = (len(train_data) - 1) // args.batchsize + 1
		for b in range(batchcnt):
			start = time.time()
			datas = train_data[b * args.batchsize: (b + 1) * args.batchsize]

			sentences = []
			tags = []
			for data in datas:
				sentences.append(data["text"])
				tags.append(data["relations"]["tags"])

			input_tensor, input_length = padding_sequence(sentences, pad_token=EMBEDDING_SIZE)
			target_tensor, target_length = padding_sequence(tags, pad_token=TAG_SIZE)
			if torch.cuda.is_available():
				input_tensor = Variable(torch.cuda.LongTensor(input_tensor, device=device)).cuda()
				target_tensor = Variable(torch.cuda.LongTensor(target_tensor, device=device)).cuda()
			else:
				input_tensor = Variable(torch.LongTensor(input_tensor, device=device))
				target_tensor = Variable(torch.LongTensor(target_tensor, device=device))

			loss, encoder_outputs, decoder_output = BiLSTM_LSTM.train(input_tensor, target_tensor, encoder,
						decoder, encoder_optimizer, decoder_optimizer, criterion)  # , input_length, target_length
			out_losses.append(loss)
			print_loss_total += loss
			# plot_loss_total += loss
			print_every = 10
			if b % print_every == 0:
				print_loss_avg = print_loss_total / print_every
				print_loss_total = 0
				print(' (%d %d%%) %.4f' % (b, float(b) / batchcnt * 100, print_loss_avg))


			RL_model = Jointly_RL.RLModel(sentences, encoder_outputs, decoder_output, dim, statedim, wv, relation_count, lr)
			RL_model.cuda()

			RL_model()
