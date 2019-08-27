import argparse


class Parser(object):
	def getParser(self):
		parser = argparse.ArgumentParser()
		parser.add_argument('--encoder_model', type=str, default='BiLSTM', choices='BiLSTM, BERT')
		parser.add_argument('--pretrained_dir', type=str, default='../bert-base-uncased')

		parser.add_argument('--hidden_dim', type=int, default=600, help="Dimension of hidden layer")
		parser.add_argument('--state_dim', type=int, default=300, help="Dimension of state")
		parser.add_argument('--embedding_dim', type=int, default=300, help="Dimension of the word vector of seq-seq model")
		parser.add_argument('--pretrain_vec', type=bool, default=True, help="Set to True to use the pre-trained w2v")
		parser.add_argument('--embedding_size', type=int, default=300, help="Dimension of the word vector")

		parser.add_argument('--dropout', type=float, default=0.1, help="Dropout")
		parser.add_argument('--lr', type=float, default=0.01, help="Learning rate sequence")  # 0.001
		parser.add_argument('--lr_RL', type=float, default=0.003, help="Learning rate RL")
		parser.add_argument('--l2', type=float, default=0.001, help="L2 regularization parameter")

		parser.add_argument('--entity_tag_size', type=int, default=7, help="Size of entity tags")
		parser.add_argument('--relation_tag_size', type=int, default=30, help="Size of relation tags")  # dm.reltaion_count
		parser.add_argument('--noisy_tag_size', type=int, default=2, help="Size of noisy tags")
		parser.add_argument('--test', type=bool, default=True, help="Set to True to inference")

		parser.add_argument('--epochRL', type=int, default=10, help="Number of epoch on training with RL")
		parser.add_argument('--sampleround', type=int, default=5, help="Sample round in RL")  # 50

		parser.add_argument('--batchsize', type=int, default=128, help="Batch size on training")
		parser.add_argument('--datapath', type=str, default='../NYT11/', help="Data directory")
		parser.add_argument('--batchsize_test', type=int, default=32, help="Batch size on testing")

		parser.add_argument('--logfile', type=str, default='HRL', help="Filename of log file")
		parser.add_argument('--epochPRE', type=int, default=15, help="Number of epoch on pretraining")
		parser.add_argument('--print_per_batch', type=int, default=200, help="Print results every XXX batches")
		parser.add_argument('--numprocess', type=int, default=4, help="Number of process")
		parser.add_argument('--start', type=str, default='', help="Directory to load model")
		parser.add_argument('--pretrain', type=bool, default=False, help="Set to True to pretrain")
		parser.add_argument('--testfile', type=str, default='test', help="Filename of test file")


		return parser
