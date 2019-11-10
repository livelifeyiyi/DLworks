import argparse


class Parser(object):
	def getParser(self):
		# Read parameters from command line
		parser = argparse.ArgumentParser()
		parser.add_argument('--train', type=str, default='D:\\data\\cross_lingual_NER\\CoNLL-2003\\eng.testa',
							help='path for train file')
		parser.add_argument('--bi_train', type=str, default='D:\\data\\cross_lingual_NER\\conll2002\\esp.testb',
							help='path for train file')
		parser.add_argument('--dev', type=str, default='D:\\data\\cross_lingual_NER\\conll2002\\esp.testa',
							help='path for dev file')
		parser.add_argument('--test', type=str, default='D:\\data\\cross_lingual_NER\\conll2002\\esp.testb',
							help='path for test file')
		# parser.add_argument("--pre_emb", default='D:\\data\\cross_lingual_NER\\embedding\\wiki.multi.en.vec',
		# 					help="Location of pretrained embeddings")
		# parser.add_argument("--bi_pre_emb", default='D:\\data\\cross_lingual_NER\\embedding\\wiki.multi.es.vec',
		# 					help="Location of bi_pretrained embeddings")
		# reload pre-trained embeddings
		parser.add_argument("--target_emb", type=str,
							default="D:\\data\\cross_lingual_NER\\embedding\\wiki.multi.en.vec",
							help="target embeddings")
		parser.add_argument("--related_emb", type=str,
							default="D:\\data\\cross_lingual_NER\\embedding\\wiki.multi.es.vec",
							help="related embeddings")
		parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
		parser.add_argument("--word_dim", default="300", type=int, help="Token embedding dimension")
		parser.add_argument("--word_lstm_dim", default="300", type=int, help="Token LSTM hidden layer size")

		parser.add_argument("--model_dp", default="model", help="model directory path")
		parser.add_argument("--tag_scheme", default="iobes", help="Tagging scheme (IOB or IOBES)")
		parser.add_argument("--lower", default='0', type=int,
							help="Lowercase words (this will not affect character inputs)")
		parser.add_argument("--zeros", default="1", type=int, help="Replace digits with 0")
		parser.add_argument("--char_dim", default="25", type=int, help="Char embedding dimension")
		parser.add_argument("--char_lstm_dim", default="25", type=int, help="Char LSTM hidden layer size")
		parser.add_argument("--char_cnn", default="1", type=int,
							help="Use CNN to generate char embeddings.(0 to disable)")
		parser.add_argument("--char_conv", default="25", type=int, help="filter number")
		parser.add_argument("--all_emb", default="0", type=int, help="Load all embeddings")
		# parser.add_argument("--feat", default="0", type=int, help="file path of external features.")
		parser.add_argument("--crf", default="1", type=int, help="Use CRF (0 to disable)")
		parser.add_argument("--dropout", default="0.5", type=float, help="Droupout on the input (0 = no dropout)")
		parser.add_argument("--tagger_learning_rate", default="0.01", type=float, help="learning rate for the tagger")
		parser.add_argument("--tagger_optimizer", default="sgd", help="Learning method (SGD, Adadelta, Adam..)")
		parser.add_argument("--dis_seq_learning_rate", default="0.01", type=float,
							help="learning rate for the discriminator")
		parser.add_argument("--dis_seq_optimizer", default="sgd", help="Learning method (SGD, Adadelta, Adam..)")
		parser.add_argument("--seq_dis_smooth", default=0.3) 
		parser.add_argument("--mapping_seq_learning_rate", default="0.01", type=float,
							help="learning rate for the mapper")
		parser.add_argument("--mapping_seq_optimizer", default="sgd", help="Learning method (SGD, Adadelta, Adam..)")
		parser.add_argument("--lr_method", default="sgd-lr_.005", help="Learning method (SGD, Adadelta, Adam..)")
		parser.add_argument("--optimizer", default="sgd", help="Learning method (SGD, Adadelta, Adam..)")
		parser.add_argument("--num_epochs", default="2", type=int, help="Number of training epochs")
		parser.add_argument("--batch_size", default="16", type=int, help="Batch size.")
		parser.add_argument("--gpu", default="0", type=int, help="default is 0. set 1 to use gpu.")
		parser.add_argument("--cuda", default="-1", type=int, help="gpu number.")
		parser.add_argument("--signal", default="", type=str)
		# parameters for word adversarial training
		parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
		parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
		parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
		parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
		parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
		parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
		# data
		parser.add_argument("--target_lang", type=str, default='en', help="target language")
		parser.add_argument("--related_lang", type=str, default='es', help="related language")
		parser.add_argument("--max_vocab", type=int, default=500000, help="Maximum vocabulary size (-1 to disable)")
		# mapping
		parser.add_argument("--map_id_init", default=True, action='store_true',
							help="Initialize the mapping as an identity matrix")
		parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
		# discriminator
		parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
		parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
		parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
		parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
		parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
		parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
		parser.add_argument("--dis_most_frequent", type=int, default=100000,
							help="Select embeddings of the k most frequent "
								 "words for discrimination (0 to disable)")
		parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
		parser.add_argument("--dis_clip_weights", type=float, default=0,
							help="Clip discriminator weights (0 to disable)")
		# training adversarial
		parser.add_argument("--adversarial", default=True, action='store_true', help="Use adversarial training")
		parser.add_argument("--adv_epochs", type=int, default=1, help="Number of epochs")
		parser.add_argument("--adv_iteration", type=int, default=100, help="Iterations per epoch")  # 1000000
		parser.add_argument("--adv_batch_size", type=int, default=32, help="Batch size")
		parser.add_argument("--map_learning_rate", type=float, default=0.1, help="learning rate")
		parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
		parser.add_argument("--dis_learning_rate", type=float, default=0.1, help="learning rate")
		parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
		parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
		parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
		parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation "
																		 "metric decreases (1 to disable)")
		# training refinement
		parser.add_argument("--n_refinement", type=int, default=1,
							help="Number of refinement iterations (0 to disable the "
								 "refinement procedure)")
		# dictionary creation parameters (for refinement)
		parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
		parser.add_argument("--dico_method", type=str, default='csls_knn_10',
							help="Method used for dictionary generation "
								 "(nn/invsm_beta_30/csls_knn_10)")
		parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
		parser.add_argument("--dico_threshold", type=float, default=0,
							help="Threshold confidence for dictionary generation")
		parser.add_argument("--dico_max_rank", type=int, default=15000,
							help="Maximum dictionary words rank (0 to disable)")
		parser.add_argument("--dico_min_size", type=int, default=0,
							help="Minimum generated dictionary size (0 to disable)")
		parser.add_argument("--dico_max_size", type=int, default=0,
							help="Maximum generated dictionary size (0 to disable)")

		return parser
