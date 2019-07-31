import os

import numpy as np
import torch
from sklearn import metrics
from tensorboardX import SummaryWriter

import bertTransformer.distributed as distributed
from bertTransformer.models.data_loader import get_minibatches
# import onmt
from bertTransformer.models.reporter import ReportMgr
from bertTransformer.models.stats import Statistics
from bertTransformer.others.logging import logger
from bertTransformer.others.utils import test_rouge, rouge_results_to_str


def _tally_parameters(model):
	n_params = sum([p.nelement() for p in model.parameters()])
	return n_params


def build_trainer(args, device_id, model,
				  optim):
	"""
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
	# device = "cpu" if args.visible_gpus == '-1' else "cuda"

	grad_accum_count = args.accum_count
	n_gpu = args.world_size

	# if device_id >= 0:  # != 'cpu':  # >= 0:
	# 	gpu_rank = int(args.gpu_ranks)
	# else:
	gpu_rank = 0
	n_gpu = 0

	print('gpu_rank %d' % gpu_rank)

	tensorboard_log_dir = args.model_path

	writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

	report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

	trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

	# print(tr)
	if (model):
		n_params = _tally_parameters(model)
		logger.info('* number of parameters: %d' % n_params)

	return trainer


class Trainer(object):
	"""
    Class that controls the training process.
    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

	def __init__(self, args, model, optim,
				 grad_accum_count=1, n_gpu=1, gpu_rank=1,
				 report_manager=None):
		# Basic attributes.
		self.args = args
		self.check_steps = args.check_steps
		self.model = model
		self.optim = optim
		self.grad_accum_count = grad_accum_count
		self.n_gpu = n_gpu
		self.gpu_rank = gpu_rank
		self.report_manager = report_manager
		self.best_acc = 0.
		self.loss = torch.nn.CrossEntropyLoss()  # torch.nn.BCELoss(reduction='none')
		assert grad_accum_count > 0
		# Set model in training mode.
		if (model):
			self.model.train()

	def train(self, train_dataset, device):  # , valid_iter_fct=None, valid_steps=-1)
		"""
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`
        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):
        Return:
            None
        """
		# step =  self.optim._step + 1
		# step = self.optim._step + 1
		# epoch = 0
		true_batchs = []
		accum = 0
		normalization = 0
		# train_iter = train_iter_fct()

		total_stats = Statistics()
		report_stats = Statistics()
		self._start_report_manager(start_time=total_stats.start_time)
		if self.args.do_eval:
			test_dataset = torch.load(self.args.bert_data_path + 'test.data')
			logger.info('Loading test dataset from %s, number of examples: %d' %
			            (self.args.bert_data_path, len(test_dataset)))

		for epoch in range(self.args.train_epochs):
			n_correct, n_total = 0., 0.
			reduce_counter = 0
			loss_total = 0

			logger.info('Getting minibatches')
			mini_batches = get_minibatches(train_dataset, self.args.batch_size, self.args.max_seq_length)
			logger.info('Number of minibatches: %s' % (len(train_dataset) // self.args.batch_size))
			logger.info('Start training...')
			for step, batch in enumerate(mini_batches):
				# if self.n_gpu == 0 or (step % self.n_gpu == self.gpu_rank):
				self.optim.zero_grad()
					# true_batchs.append(batch)
					# normalization += batch.batch_size
					# accum += 1
					# if accum == self.grad_accum_count:
					# 	reduce_counter += 1
					# 	if self.n_gpu > 1:
					# 		normalization = sum(distributed.all_gather_list(normalization))
				src, labels, segs, clss = batch[0], batch[1], batch[2], batch[3]
				if torch.cuda.is_available():
					src = torch.cuda.LongTensor(src).to(device)  # .reshape(-1, self.args.max_seq_length)
					labels = torch.cuda.LongTensor(labels).to(device)  # .reshape(1, -1)
					segs = torch.cuda.LongTensor(segs).to(device)  # .reshape(1, -1)

					clss = [(cls + [-1] * (max([len(i) for i in clss]) - len(cls))) for cls in clss]
					clss = torch.cuda.LongTensor(clss).to(device)
					mask = torch.cuda.ByteTensor((1 - (src == 0))).to(device)
					mask_cls = torch.cuda.ByteTensor((1 - (clss == -1)))
				else:
					src = torch.LongTensor(src).to(device)  		# .reshape(-1, self.args.max_seq_length)
					labels = torch.LongTensor(labels).to(device)  	# .reshape(1, -1)
					segs = torch.LongTensor(segs).to(device)		# .reshape(1, -1)

					clss = [(cls + [-1] * (max([len(i) for i in clss]) - len(cls))) for cls in clss]
					clss = torch.LongTensor(clss).to(device)
					mask = torch.ByteTensor((1 - (src == 0))).to(device)
					mask_cls = torch.ByteTensor((1 - (clss == -1)))  # torch.ByteTensor(mask_cls).to(device)
				# src = batch.src
				# labels = batch.labels
				# segs = batch.segs
				# clss = batch.clss
				# mask = batch.mask
				# mask_cls = batch.mask_cls

				logits = self.model(src, segs, clss, mask, mask_cls)  # , mask

				loss = self.loss(logits, labels)
				n_correct += (torch.argmax(logits, -1) == labels).sum().item()
				n_total += len(logits)
				loss_total += loss.item() * len(logits)
				# loss = (loss * mask.float()).sum()
				# (loss / loss.numel()).backward()
				loss.backward()
				# loss.div(float(normalization)).backward()
				# 4. Update the parameters and statistics.
				# if self.grad_accum_count == 1:
				# Multi GPU gradient gather
				if self.n_gpu > 1:
					grads = [p.grad.data for p in self.model.parameters()
							 if p.requires_grad
							 and p.grad is not None]
					distributed.all_reduce_and_rescale_tensors(
						grads, float(1))
				self.optim.step()

				batch_stats = Statistics(float(loss.cpu().item()), normalization)
				total_stats.update(batch_stats)
				report_stats.update(batch_stats)

				logger.info('step-{}, loss:{:.4f}, acc:{:.4f}'.format(step, loss_total / n_total, n_correct / n_total))
				if step != 0 and step % self.check_steps == 0:
					valid_acc = self.test(self.model, test_dataset, device)
					if valid_acc > self.best_acc:
						self.best_acc = valid_acc
						self._save(self.model, epoch, self.best_acc)
				# 	self._save(epoch, step)
				# report_stats = self._maybe_report_training(step, epoch, self.optim.learning_rate, report_stats)

				# in case of multi step gradient accumulation,
				# update only after accum batches
			valid_acc = self.test(self.model, test_dataset, device)
			if valid_acc > self.best_acc:
				self.best_acc = valid_acc
				self._save(self.model, epoch, self.best_acc)
			if self.grad_accum_count > 1:
				if self.n_gpu > 1:
					grads = [p.grad.data for p in self.model.parameters()
							 if p.requires_grad
							 and p.grad is not None]
					distributed.all_reduce_and_rescale_tensors(
						grads, float(1))
				self.optim.step()

			# return n_correct, n_total, loss_total

			if self.args.do_eval:
				# model = trainer.model
				# self.model.eval()
				# trainer = build_trainer(args, device_id, model, None)
				try:
					self.test(self.model, test_dataset, device)
				except Exception as e:
					logger.error(e)
				# true_batchs = []
				# accum = 0
				# normalization = 0
				# step += 1
				# if step > train_steps:
				# 	break

		# return total_stats

	def validate(self, valid_dataset, device, epoch=0):
		""" Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
		# Set model in validating mode.
		self.model.eval()
		stats = Statistics()

		with torch.no_grad():
			mini_batches = get_minibatches(valid_dataset, self.args.batch_size, self.args.max_seq_length)
			logger.info('Number of minibatches: %s' % (len(valid_dataset) // self.args.batch_size))
			for step, batch in enumerate(mini_batches):
				src, labels, segs, clss = batch[0], batch[1], batch[2], batch[3]
				if torch.cuda.is_available():
					src = torch.cuda.LongTensor(src).to(device)  # .reshape(-1, self.args.max_seq_length)
					labels = torch.cuda.LongTensor(labels).to(device)  # .reshape(1, -1)
					segs = torch.cuda.LongTensor(segs).to(device)  # .reshape(1, -1)

					clss = [(cls + [-1] * (max([len(i) for i in clss]) - len(cls))) for cls in clss]
					clss = torch.cuda.LongTensor(clss).to(device)
					mask = torch.cuda.ByteTensor((1 - (src == 0))).to(device)
					mask_cls = torch.cuda.ByteTensor((1 - (clss == -1)))
				else:
					src = torch.LongTensor(src).to(device)  # .reshape(-1, self.args.max_seq_length)
					labels = torch.LongTensor(labels).to(device)  # .reshape(1, -1)
					segs = torch.LongTensor(segs).to(device)  # .reshape(1, -1)

					clss = [(cls + [-1] * (max([len(i) for i in clss]) - len(cls))) for cls in clss]
					clss = torch.LongTensor(clss).to(device)
					mask = torch.ByteTensor((1 - (src == 0))).to(device)
					mask_cls = torch.ByteTensor((1 - (clss == -1)))  # torch.ByteTensor(mask_cls).to(device)

				logits = self.model(src, segs, clss, mask, mask_cls)  # , mask

				loss = self.loss(logits, labels)
				# loss = (loss * mask.float()).sum()
				batch_stats = Statistics(float(loss.cpu().item()), len(labels))
				stats.update(batch_stats)
			self._report_step(0, epoch, valid_stats=stats)
			return stats

	def test(self, model, test_dataset, device, cal_lead=False, cal_oracle=False):
		""" Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
		model.eval()
		stats = Statistics()
		mini_batches = get_minibatches(test_dataset, self.args.batch_size, self.args.max_seq_length)
		logger.info('Number of minibatches: %s' % (len(test_dataset) // self.args.batch_size))
		with torch.no_grad():
			n_correct = 0.
			n_total = 0.
			target_all = None
			output_all = None
			for step, batch in enumerate(mini_batches):
				src, labels, segs, clss = batch[0], batch[1], batch[2], batch[3]
				if torch.cuda.is_available():
					src = torch.cuda.LongTensor(src).to(device)  # .reshape(-1, self.args.max_seq_length)
					labels = torch.cuda.LongTensor(labels).to(device)  # .reshape(1, -1)
					segs = torch.cuda.LongTensor(segs).to(device)  # .reshape(1, -1)

					clss = [(cls + [-1] * (max([len(i) for i in clss]) - len(cls))) for cls in clss]
					clss = torch.cuda.LongTensor(clss).to(device)
					mask = torch.cuda.ByteTensor((1 - (src == 0))).to(device)
					mask_cls = torch.cuda.ByteTensor((1 - (clss == -1)))
				else:
					src = torch.LongTensor(src).to(device)  # .reshape(-1, self.args.max_seq_length)
					labels = torch.LongTensor(labels).to(device)  # .reshape(1, -1)
					segs = torch.LongTensor(segs).to(device)  # .reshape(1, -1)

					clss = [(cls + [-1] * (max([len(i) for i in clss]) - len(cls))) for cls in clss]
					clss = torch.LongTensor(clss).to(device)
					mask = torch.ByteTensor((1 - (src == 0))).to(device)
					mask_cls = torch.ByteTensor((1 - (clss == -1)))  # torch.ByteTensor(mask_cls).to(device)

				logits = self.model(src, segs, clss, mask, mask_cls)  # , mask
				# loss = self.loss(logits, labels)
				n_correct += (torch.argmax(logits, -1) == labels).sum().item()
				n_total += len(logits)
				if target_all is None:
					target_all = labels
					output_all = logits
				else:
					target_all = torch.cat((target_all, labels), dim=0)
					output_all = torch.cat((output_all, logits), dim=0)

				# batch_stats = Statistics(float(loss.cpu().item()), len(labels))
				# stats.update(batch_stats)

				# sent_scores = sent_scores + mask.float()
				# sent_scores = sent_scores.cpu().data.numpy()
				# selected_ids = np.argsort(-sent_scores, 1)
			acc = n_correct / n_total
			pred_res = metrics.classification_report(target_all.cpu(), torch.argmax(output_all, -1).cpu(),
												 target_names=['NEG', 'NEU', 'POS'])
			logger.info('Prediction results for test dataset: \n{}'.format(pred_res))
			# self._report_step(0, step, valid_stats=stats)
		return acc

		def orig():
			# Set model in validating mode.
			def _get_ngrams(n, text):
				ngram_set = set()
				text_length = len(text)
				max_index_ngram_start = text_length - n
				for i in range(max_index_ngram_start + 1):
					ngram_set.add(tuple(text[i:i + n]))
				return ngram_set

			def _block_tri(c, p):
				tri_c = _get_ngrams(3, c.split())
				for s in p:
					tri_s = _get_ngrams(3, s.split())
					if len(tri_c.intersection(tri_s)) > 0:
						return True
				return False

			if not cal_lead and not cal_oracle:
				model.eval()
			stats = Statistics()

			can_path = '%s_step%d.candidate' % (self.args.result_path, step)
			gold_path = '%s_step%d.gold' % (self.args.result_path, step)
			with open(can_path, 'w') as save_pred:
				with open(gold_path, 'w') as save_gold:
					with torch.no_grad():
						target_all = []
						output_all = []
						# n_correct, n_total = 0., 0.
						mini_batches = get_minibatches(test_dataset, self.args.batch_size, self.args.max_seq_length)
						for i, batch in enumerate(mini_batches):
							src = batch.src
							labels = batch.labels
							segs = batch.segs
							clss = batch.clss
							mask = batch.mask
							mask_cls = batch.mask_cls

							gold = []
							pred = []

							if (cal_lead):
								selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
							elif (cal_oracle):
								selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
												range(batch.batch_size)]
							else:
								logits = model(src, segs, clss, mask, mask_cls)

								loss = self.loss(logits, labels)  # loss = self.loss(sent_scores, labels.float())
								# loss = (loss * mask.float()).sum()
								# n_correct += (torch.argmax(logits, -1) == labels).sum().item()
								# n_total += len(logits)
								if target_all is None:
									target_all = labels
									output_all = logits
								else:
									target_all = torch.cat((target_all, labels), dim=0)
									output_all = torch.cat((output_all, logits), dim=0)

								batch_stats = Statistics(float(loss.cpu().item()), len(labels))
								stats.update(batch_stats)

								sent_scores = sent_scores + mask.float()
								sent_scores = sent_scores.cpu().data.numpy()
								selected_ids = np.argsort(-sent_scores, 1)
							# selected_ids = np.sort(selected_ids,1)
							for i, idx in enumerate(selected_ids):
								_pred = []
								if len(batch.src_str[i]) == 0:
									continue
								for j in selected_ids[i][:len(batch.src_str[i])]:
									if j >= len(batch.src_str[i]):
										continue
									candidate = batch.src_str[i][j].strip()
									if self.args.block_trigram:
										if not _block_tri(candidate, _pred):
											_pred.append(candidate)
									else:
										_pred.append(candidate)

									if (not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3:
										break

								_pred = '<q>'.join(_pred)
								if self.args.recall_eval:
									_pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

								pred.append(_pred)
								gold.append(batch.tgt_str[i])

							for i in range(len(gold)):
								save_gold.write(gold[i].strip() + '\n')
							for i in range(len(pred)):
								save_pred.write(pred[i].strip() + '\n')
					pred_res = metrics.classification_report(target_all.cpu(), torch.argmax(output_all, -1).cpu(),
															 target_names=['NEG', 'NEU', 'POS'])
					logger.info('Prediction results for test dataset: \n{}'.format(pred_res))
			if step != -1 and self.args.report_rouge:
				rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
				logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
			self._report_step(0, step, valid_stats=stats)

			return stats

	def _gradient_accumulation(self, true_batchs, normalization, total_stats,
							   report_stats, n_correct, n_total):
		if self.grad_accum_count > 1:
			self.model.zero_grad()
		loss_total = 0.
		for batch in true_batchs:
			if self.grad_accum_count == 1:
				self.model.zero_grad()

			src = batch.src
			labels = batch.labels
			segs = batch.segs
			clss = batch.clss
			mask = batch.mask
			mask_cls = batch.mask_cls

			logits = self.model(src, segs, clss, mask, mask_cls)  # , mask

			loss = self.loss(logits, labels)
			n_correct += (torch.argmax(logits, -1) == labels).sum().item()
			n_total += len(logits)
			loss_total += loss.item() * len(logits)
			# loss = (loss * mask.float()).sum()
			# (loss / loss.numel()).backward()
			loss.backward()
			# loss.div(float(normalization)).backward()

			batch_stats = Statistics(float(loss.cpu().item()), normalization)

			total_stats.update(batch_stats)
			report_stats.update(batch_stats)

			# 4. Update the parameters and statistics.
			if self.grad_accum_count == 1:
				# Multi GPU gradient gather
				if self.n_gpu > 1:
					grads = [p.grad.data for p in self.model.parameters()
							 if p.requires_grad
							 and p.grad is not None]
					distributed.all_reduce_and_rescale_tensors(
						grads, float(1))
				self.optim.step()

		# in case of multi step gradient accumulation,
		# update only after accum batches
		if self.grad_accum_count > 1:
			if self.n_gpu > 1:
				grads = [p.grad.data for p in self.model.parameters()
						 if p.requires_grad
						 and p.grad is not None]
				distributed.all_reduce_and_rescale_tensors(
					grads, float(1))
			self.optim.step()

		return n_correct, n_total, loss_total

	def _save(self, model_name, epoch, acc):
		real_model = self.model
		# real_generator = (self.generator.module
		#                   if isinstance(self.generator, torch.nn.DataParallel)
		#                   else self.generator)

		model_state_dict = real_model.state_dict()
		# generator_state_dict = real_generator.state_dict()
		checkpoint = {
			'model': model_state_dict,
			# 'generator': generator_state_dict,
			'opt': self.args,
			'optim': self.optim,
		}
		checkpoint_path = os.path.join(self.args.model_path, 'model_{}_epoch_{}_acc_{:.4f}.pt'.format(model_name, epoch, acc))
		logger.info("Saving checkpoint %s" % checkpoint_path)
		# checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
		if not os.path.exists(checkpoint_path):
			torch.save(checkpoint, checkpoint_path)
			return checkpoint, checkpoint_path

	def _start_report_manager(self, start_time=None):
		"""
        Simple function to start report manager (if any)
        """
		if self.report_manager is not None:
			if start_time is None:
				self.report_manager.start()
			else:
				self.report_manager.start_time = start_time

	def _maybe_gather_stats(self, stat):
		"""
        Gather statistics in multi-processes cases
        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)
        Returns:
            stat: the updated (or unchanged) stat object
        """
		if stat is not None and self.n_gpu > 1:
			return Statistics.all_gather_stats(stat)
		return stat

	def _maybe_report_training(self, step, num_steps, learning_rate,
							   report_stats):
		"""
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
		if self.report_manager is not None:
			return self.report_manager.report_training(
				step, num_steps, learning_rate, report_stats,
				multigpu=self.n_gpu > 1)

	def _report_step(self, learning_rate, step, train_stats=None,
					 valid_stats=None):
		"""
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
		if self.report_manager is not None:
			return self.report_manager.report_step(
				learning_rate, step, train_stats=train_stats,
				valid_stats=valid_stats)

	def _maybe_save(self, step):
		"""
        Save the model if a model saver is set
        """
		if self.model_saver is not None:
			self.model_saver.maybe_save(step)
