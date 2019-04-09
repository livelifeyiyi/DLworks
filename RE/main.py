import random, sys, time, os
import torch
from data_manager import DataManager
# from Model import Model
from Parser import Parser
# from TrainProcess import train, test, worker
# import torch.multiprocessing as mp
# from AccCalc import calcF1
from TFgirl.RE import Jointly_RL
from TFgirl.RE.BiLSTM_LSTM import EncoderRNN, DecoderRNN

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

	for e in range(args.epochRL):
		random.shuffle(train_data)
		print("training epoch ", e)
		batchcnt = (len(train_data) - 1) // args.batchsize + 1
		for b in range(batchcnt):
			start = time.time()
			data = train_data[b * args.batchsize: (b + 1) * args.batchsize]

	relation_model = Jointly_RL.RelationModel(dim, statedim, relation_count, noisy_count)
	relation_model.cuda()

