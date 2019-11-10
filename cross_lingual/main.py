import os
import sys
import random
import time
import itertools
import torch
import torch.nn as nn

from Parser import Parser
from pre_process.parse_args import parse_args
from evaluate import evaluate_ner
from pre_process.tag_schema import update_tag_scheme
from pre_process.loader import augment_with_pretrained, augment_with_pretrained_bi, load_sentences
from pre_process.preprocessing import *
from wordAdversarial import build_word_adversarial_model, WordAdversarialTrainer
from seqAdversarial import build_sequence_adversarial_model, SeqAdversarialTrainer
from muse.evaluator import all_eval

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import initLogger
logger = initLogger.init_logger()
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'


def train():
	# initialize model
	logger.info('******model initializing...******')

	target_emb, related_emb, embedding_mapping, word_discriminator = build_word_adversarial_model(params)
	adv_trainer = WordAdversarialTrainer(target_emb, related_emb, embedding_mapping, word_discriminator, params)

	target_word_embedding, related_word_embedding, char_embedding, target_char_cnn_word, related_char_cnn_word, \
	adv_lstm, seq_discriminator, context_lstm, linear_proj, tagger_criterion, dis_criterion = \
		build_sequence_adversarial_model(params, mappings)

	seq_trainer = SeqAdversarialTrainer(target_word_embedding, related_word_embedding, embedding_mapping,
										char_embedding,
										target_char_cnn_word, related_char_cnn_word, adv_lstm, seq_discriminator,
										context_lstm, linear_proj, tagger_criterion, dis_criterion, params)

	module_list = nn.ModuleList([target_emb, related_emb, embedding_mapping, word_discriminator, target_word_embedding,
								 related_word_embedding, char_embedding, target_char_cnn_word, related_char_cnn_word,
								 adv_lstm, seq_discriminator, context_lstm, linear_proj, tagger_criterion,
								 dis_criterion])

	'''
	logger.info('----> WORD ADVERSARIAL TRAINING <----\n')
	best_valid_metric = 0
	for n_epoch in range(params['adv_epochs']):
		print('Starting adversarial training epoch %i...' % n_epoch)
		for n_iter in range(0, params['adv_iteration'], params['adv_batch_size']):
			# discriminator training
			dis_loss = 0
			for _ in range(params['dis_steps']):
				# select random word IDs
				target_ids = torch.LongTensor(params['adv_batch_size']).random_(params['dis_most_frequent'])
				related_ids = torch.LongTensor(params['adv_batch_size']).random_(params['dis_most_frequent'])

				if params['gpu']:
					target_ids = target_ids.cuda()  # pre_emb
					related_ids = related_ids.cuda()
				dis_loss += adv_trainer.dis_step(target_ids, related_ids)
			dis_loss /= params['dis_steps']

			# mapping training (discriminator fooling)
			# select random word IDs
			target_ids = torch.LongTensor(params['adv_batch_size']).random_(params['dis_most_frequent'])
			related_ids = torch.LongTensor(params['adv_batch_size']).random_(params['dis_most_frequent'])

			if params['gpu']:
				target_ids = target_ids.cuda()  # pre_emb
				related_ids = related_ids.cuda()
			p, map_loss = adv_trainer.mapping_step(target_ids, related_ids)

			# sys.stdout.write(
			#     'epoch %i, iter %i, discriminator loss: %f, mapping loss: %f\r' % (
			#         n_epoch, n_iter, dis_loss, map_loss))
			# sys.stdout.flush()
			if n_iter % 10000 == 0:
				print('epoch %i, iter %i, discriminator loss: %f, mapping loss: %f\r' %
					  (n_epoch, n_iter, dis_loss, map_loss))

		# embeddings / discriminator evaluation
		projected_related_emb = embedding_mapping.forward(related_emb.weight)
		valid_metric = all_eval(projected_related_emb, target_emb, params['dico_eval'], VALIDATION_METRIC)

		if valid_metric > best_valid_metric:
			path = os.path.join(params['model_dp'], 'best_mapping.pth')
			print('* Saving the mapping to %s ...' % path)
			state = {'state_dict': embedding_mapping.state_dict()}
			torch.save(state, path)
			best_valid_metric = valid_metric
		print('End of epoch %i.\n' % n_epoch)

	if params['n_refinement'] > 0:
		# training loop
		for n_iter in range(params['n_refinement']):
			print('Starting refinement iteration %i...' % n_iter)
			# build a dictionary from aligned embeddings
			adv_trainer.build_dictionary()
			# apply the Procrustes solution
			adv_trainer.procrustes()
			# embeddings evaluation
			projected_related_emb = embedding_mapping.forward(related_emb.weight)
			valid_metric = all_eval(projected_related_emb, target_emb, params['dico_eval'], VALIDATION_METRIC)

			# JSON log / save best model / end of epoch
			if valid_metric > best_valid_metric:
				path = os.path.join(params['model_dp'], 'best_mapping.pth')
				state = {'state_dict': embedding_mapping.state_dict()}
				torch.save(state, path)
				print('End of refinement iteration %i.\n' % n_iter)
	'''
	# training starts
	since = time.time()
	best_dev = 0.0
	best_test = 0.0
	best_test_global = 1.0
	metric = 'f1'  # use metric 'f1' or 'acc'
	num_epochs = params['num_epochs']
	batch_size = params['batch_size']
	current_patience = 0
	patience = 50
	num_batches = 0

	logger.info('----> SEQUENCE ADVERSARIAL TRAINING <----\n')
	for epoch in range(60):
		logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
		time_epoch_start = time.time()  # epoch start time

		# training
		target_train_batches = [dataset['train'][i:i + batch_size] for i in range(0, len(dataset['train']), batch_size)]
		related_train_batches_tmp = [dataset['bi_train'][i:i + batch_size] for i in
									 range(0, len(dataset['bi_train']), batch_size)]
		dev_batches = [dataset['dev'][i:i + batch_size] for i in
					   range(0, len(dataset['dev']), batch_size)]
		test_batches = [dataset['test'][i:i + batch_size] for i in
						range(0, len(dataset['test']), batch_size)]

		if len(related_train_batches_tmp) < len(target_train_batches):
			related_train_batches = []
			while len(related_train_batches) < len(target_train_batches):
				related_train_batches += related_train_batches_tmp
		else:
			related_train_batches = random.sample(related_train_batches_tmp, len(target_train_batches))

		random.shuffle(target_train_batches)
		random.shuffle(related_train_batches)
		for target_batch, related_batch in zip(target_train_batches, related_train_batches):
			num_batches += 1
			target_inputs = create_input(target_batch, is_cuda=params['gpu'])
			target_word_ids = target_inputs['words']
			target_char_ids = target_inputs['chars']
			target_char_len = target_inputs['char_len']
			target_seq_len = target_inputs['seq_len']
			target_reference = target_inputs['tags']

			related_inputs = create_input(related_batch, is_cuda=params['gpu'])
			related_word_ids = related_inputs['words']
			related_char_ids = related_inputs['chars']
			related_char_len = related_inputs['char_len']
			related_seq_len = related_inputs['seq_len']
			related_reference = related_inputs['tags']

			if torch.cuda.is_available():
				target_word_ids = to_var(target_word_ids)
				target_char_ids = to_var(target_char_ids)
				related_word_ids = to_var(related_word_ids)
				related_char_ids = to_var(related_char_ids)

			dis_loss = seq_trainer.dis_step(target_word_ids, target_char_ids, target_char_len, target_seq_len,
											related_word_ids, related_char_ids, related_char_len, related_seq_len)

			map_loss = seq_trainer.tagger_step(target_word_ids, target_char_ids, target_char_len, target_seq_len,
											   target_reference, related_word_ids, related_char_ids,
											   related_char_len, related_seq_len, related_reference)

			for _ in range(5):
				rand_idx_related = random.randint(0, len(related_train_batches) - 1)
				rand_idx_target = random.randint(0, len(target_train_batches) - 1)
				target_batch_dis = target_train_batches[rand_idx_target]
				related_batch_dis = related_train_batches[rand_idx_related]

				target_inputs_dis = create_input(target_batch_dis, is_cuda=params['gpu'])
				target_word_ids_dis = target_inputs_dis['words']
				target_char_ids_dis = target_inputs_dis['chars']
				target_char_len_dis = target_inputs_dis['char_len']
				target_seq_len_dis = target_inputs_dis['seq_len']
				target_reference_dis = target_inputs_dis['tags']

				related_inputs_dis = create_input(related_batch_dis, is_cuda=params['gpu'])
				related_word_ids_dis = related_inputs_dis['words']
				related_char_ids_dis = related_inputs_dis['chars']
				related_char_len_dis = related_inputs_dis['char_len']
				related_seq_len_dis = related_inputs_dis['seq_len']
				related_reference_dis = related_inputs_dis['tags']

				if torch.cuda.is_available():
					target_word_ids_dis = to_var(target_word_ids_dis)
					target_char_ids_dis = to_var(target_char_ids_dis)
					related_word_ids_dis = to_var(related_word_ids_dis)
					related_char_ids_dis = to_var(related_char_ids_dis)

				loss = seq_trainer.tagger_step(target_word_ids_dis, target_char_ids_dis, target_char_len_dis,
											   target_seq_len_dis, target_reference_dis, related_word_ids_dis,
											   related_char_ids_dis, related_char_len_dis, related_seq_len_dis,
											   related_reference_dis)
			if num_batches % 100 == 0:
				logger.info('adversarial epoch %i , current discriminator loss: %f,  current mapping loss: %f\r' %
					  (epoch, dis_loss.item(), map_loss.item()))

		dev_preds = []
		for dev_batch in dev_batches:
			dev_input = create_input(dev_batch, is_cuda=params['gpu'])
			dev_word_ids = dev_input['words']
			dev_char_ids = dev_input['chars']
			dev_char_len = dev_input['char_len']
			dev_seq_len = dev_input['seq_len']
			dev_reference = dev_input['tags']

			if torch.cuda.is_available():
				dev_word_ids = to_var(dev_word_ids)
				dev_char_ids = to_var(dev_char_ids)

			dev_pred_seq = seq_trainer.tagging_dev_step(dev_word_ids, dev_char_ids, dev_char_len, dev_seq_len,
														dev_reference)
			dev_preds += dev_pred_seq

		dev_f1, dev_acc, dev_predicted_bio = evaluate_ner(
			params, dev_preds,
			list(itertools.chain.from_iterable(dev_batches)),
			mappings['id_to_tag'],
			mappings['id_to_word']
		)
		if dev_f1 > best_dev:
			best_dev = dev_f1
			current_patience = 0
			logger.info('new best score on dev: %.4f and on test: %.4f and best on global test: %.4f  (adversarial)'
				  % (best_dev, best_test, best_test_global))
			logger.info('saving the current model to disk...')

			state = {
				'epoch': epoch + 1,
				'parameters': params,
				'mappings': mappings,
				'state_dict': module_list.state_dict(),
				'best_prec1': best_dev,
			}
			torch.save(state, os.path.join(model_dir, 'best_model.pth.tar'))
			with open(os.path.join(model_dir, 'best_dev.ner.bio'), 'w') as f:
				f.write(dev_predicted_bio)
		else:
			current_patience += 1

		logger.info('{} epoch: {} batch: {} F1: {:.4f} Acc: {:.4f}  Current best dev: {:.4f}\n'.format(
			"dev", epoch, num_batches, dev_f1, dev_acc, best_dev))

		time_epoch_end = time.time()  # epoch end time
		logger.info('epoch training time: %f seconds' % round(
			(time_epoch_end - time_epoch_start), 2))


if __name__ == "__main__":
	# args = sys.argv[1:]
	parser = Parser().getParser()
	args = parser.parse_args()
	params, model_names = parse_args(args)
	# params, model_names = parser.parse_known_args(args)

	torch.cuda.set_device(args.cuda)

	# generate model name
	model_dir = args.model_dp
	model_name = []
	for k in model_names:
		v = params[k]
		if not v:
			continue
		if k == 'pre_emb':
			v = os.path.basename(v)
		model_name.append('='.join((k, str(v))))
	logger.info(model_name)
	# model_dir = os.path.join(model_dir, ','.join(model_name[:-1]))
	# os.makedirs(model_dir, exist_ok=True)
	# params['model_dp'] = model_dir

	# print model parameters
	logger.info('Training data: %s' % params['train'])
	logger.info('Bi_Training data: %s' % params['bi_train'])
	logger.info('Dev data: %s' % params['dev'])
	logger.info('Test data: %s' % params['test'])
	logger.info("Model location: %s" % params['model_dp'])
	logger.info("Target embedding: %s " % params['target_emb'])
	logger.info("Related embedding: %s " % params['related_emb'])

	eval_path = os.path.join(os.path.dirname(__file__), "./evaluation")
	eval_script = os.path.join(eval_path, 'conlleval')
	params['eval_script'] = eval_script

	# Data parameters
	lower = params['lower']
	zeros = params['zeros']
	tag_scheme = params['tag_scheme']
	params['max_word_length'] = 25
	params['filter_withs'] = [2, 3, 4]
	params['char_cnn_word_dim'] = params['word_dim'] + params['char_conv'] * len(params['filter_withs'])
	params['shared_lstm_hidden_dim'] = params['word_lstm_dim']

	# Load sentences
	train_sentences = load_sentences(params['train'], to_sort=False)
	bi_train_sentences = load_sentences(params['bi_train'], to_sort=False)
	dev_sentences = load_sentences(params['dev'])
	test_sentences = load_sentences(params['test'])

	# Use selected tagging scheme (IOB / IOBES), also check if tagging scheme is valid
	update_tag_scheme(train_sentences, tag_scheme)
	update_tag_scheme(bi_train_sentences, tag_scheme)
	update_tag_scheme(dev_sentences, tag_scheme)
	update_tag_scheme(test_sentences, tag_scheme)

	# prepare mappings
	all_sentences = train_sentences + dev_sentences + test_sentences
	mappings = prepare_mapping_bi(all_sentences, bi_train_sentences, **params)

	# If pretrained embeddings is used and all_emb flag is on,
	# we augment the words by pretrained embeddings.
	# if parameters['pre_emb'] and parameters['all_emb']:
	updated_word_mappings = augment_with_pretrained(all_sentences, params['target_emb'])
	mappings.update(updated_word_mappings)

	updated_word_mappings = augment_with_pretrained_bi(bi_train_sentences, params['related_emb'])
	mappings.update(updated_word_mappings)

	# compute vocab size
	params['label_size'] = len(mappings['id_to_tag'])
	params['word_vocab_size'] = len(mappings['id_to_word'])
	params['bi_word_vocab_size'] = len(mappings['bi_id_to_word'])
	params['char_vocab_size'] = len(mappings['id_to_char'])
	params['feat_vocab_size'] = [len(item) for item in mappings['id_to_feat_list']]

	logger.info("word vocab size: %s" % params['word_vocab_size'])
	logger.info("bi word vocab size: %s" % params['bi_word_vocab_size'])

	# Index data
	dataset = dict()
	dataset['train'] = prepare_dataset(train_sentences, mappings, zeros, lower, is_train=True)
	dataset['bi_train'] = prepare_dataset_bi(bi_train_sentences, mappings, zeros, lower, is_train=True)
	dataset['dev'] = prepare_dataset(dev_sentences, mappings, zeros, lower, is_train=True)
	dataset['test'] = prepare_dataset(test_sentences, mappings, zeros, lower, is_train=True)

	logger.info("%i / %i / %i / %i sentences in train / bi-train / dev / test." % (
		len(dataset['train']), len(dataset['bi_train']), len(dataset['dev']), len(dataset['test'])))

	train()
