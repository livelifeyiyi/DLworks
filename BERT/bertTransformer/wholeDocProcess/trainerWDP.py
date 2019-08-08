import os

import numpy as np
import torch
from sklearn import metrics
from tensorboardX import SummaryWriter

import bertTransformer.distributed as distributed
# import onmt
from bertTransformer.models.reporter import ReportMgr
from bertTransformer.models.stats import Statistics
from bertTransformer.others.logging import logger
# from bertTransformer.others.utils import test_rouge, rouge_results_to_str
# from bertTransformer.evaluate import Evaluation

from bertTransformer.models.data_loader import get_minibatches_WDP


def _tally_parameters(model):
	n_params = sum([p.nelement() for p in model.parameters()])
	return n_params


def build_trainer(args, device_id, model, optim):
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

	def train(self, train_dataset, device, test_dataset):  # , valid_iter_fct=None, valid_steps=-1)

		normalization = 0
		total_stats = Statistics()
		report_stats = Statistics()
		self._start_report_manager(start_time=total_stats.start_time)

		for epoch in range(self.args.train_epochs):
			n_correct, n_total = 0., 0.
			reduce_counter = 0
			loss_total = 0

			logger.info('Getting minibatches')
			mini_batches = get_minibatches_WDP(train_dataset, self.args.batch_size)
			logger.info('Number of minibatches: %s' % (len(train_dataset[0]) // self.args.batch_size))
			logger.info('Start training...')
			for step, batch in enumerate(mini_batches):
				# if self.n_gpu == 0 or (step % self.n_gpu == self.gpu_rank):
				x, labels = batch
				if torch.cuda.is_available():
					x = torch.cuda.Tensor(x).to(device)
					labels = torch.cuda.Tensor(labels).to(device)
				else:
					x = torch.Tensor(x).to(device)
					labels = torch.Tensor(labels).to(device)
				self.optim.zero_grad()

				logits = self.model(x)  # , mask

				loss = self.loss(logits, labels)
				n_correct += (torch.argmax(logits, -1) == labels).sum().item()
				n_total += len(logits)
				loss_total += loss.item() * len(logits)

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
						self._save(str(self.args.model_name)+str(self.args.lr)+'valid', epoch, self.best_acc)
				# self._save(epoch, step)
				# report_stats = self._maybe_report_training(step, epoch, self.optim.learning_rate, report_stats)

				# in case of multi step gradient accumulation,
				# update only after accum batches
			valid_acc = self.test(self.model, test_dataset, device)
			# if valid_acc > self.best_acc:
			# 	self.best_acc = valid_acc
			self._save(str(self.args.model_name)+str(self.args.lr), epoch, valid_acc)
			if self.grad_accum_count > 1:
				if self.n_gpu > 1:
					grads = [p.grad.data for p in self.model.parameters()
							 if p.requires_grad
							 and p.grad is not None]
					distributed.all_reduce_and_rescale_tensors(
						grads, float(1))
				self.optim.step()

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
			mini_batches = get_minibatches_WDP(valid_dataset, self.args.batch_size, self.args.max_seq_length)
			logger.info('Number of minibatches: %s' % (len(valid_dataset) // self.args.batch_size))
			for step, batch in enumerate(mini_batches):
				x, labels = batch
				x = torch.cuda.Tensor(x)
				labels = torch.cuda.Tensor(labels)
				logits = self.model(x)  # , mask

				loss = self.loss(logits, labels)
				# loss = (loss * mask.float()).sum()
				batch_stats = Statistics(float(loss.cpu().item()), len(labels))
				stats.update(batch_stats)
			self._report_step(0, epoch, valid_stats=stats)
			return stats

	def test(self, model, test_dataset, device):
		""" Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
		model.eval()
		mini_batches = get_minibatches_WDP(test_dataset, self.args.batch_size, self.args.max_seq_length)
		logger.info('Number of minibatches: %s' % (len(test_dataset) // self.args.batch_size))
		with torch.no_grad():
			n_correct = 0.
			n_total = 0.
			target_all = None
			output_all = None
			full_pred = []
			full_label_ids = []
			for step, batch in enumerate(mini_batches):
				x, labels = batch
				if torch.cuda.is_available():
					x = torch.cuda.Tensor(x).to(device)
					labels = torch.cuda.Tensor(labels).to(device)
				else:
					x = torch.Tensor(x).to(device)
					labels = torch.Tensor(labels).to(device)

				logits = self.model(x)  # , mask
				# loss = self.loss(logits, labels)
				n_correct += (torch.argmax(logits, -1) == labels).sum().item()
				n_total += len(logits)
				full_pred.extend(torch.argmax(logits, -1).tolist())
				full_label_ids.extend(labels.tolist())

				if target_all is None:
					target_all = labels
					output_all = logits
				else:
					target_all = torch.cat((target_all, labels), dim=0)
					output_all = torch.cat((output_all, logits), dim=0)

			acc = n_correct / n_total
			pred_res = metrics.classification_report(target_all.cpu(), torch.argmax(output_all, -1).cpu(),
												 target_names=['NEG', 'NEU', 'POS'])
			logger.info('Prediction results for test dataset: \n{}'.format(pred_res))

			# self._report_step(0, step, valid_stats=stats)
		return acc

	def _save(self, model_name, epoch, acc):
		real_model = self.model
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
