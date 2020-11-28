from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
from utils import fix_random_seed_as

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAETrainer(AbstractTrainer):
	def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
		super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
		self.max_len = args.max_len
		self.recover_len = self.max_len

	@classmethod
	def code(cls):
		return 'vae'

	def add_extra_loggers(self):
		pass

	def log_extra_train_info(self, log_data):
		pass

	def calculate_loss(self, batch):
		d = self.model(batch)
		recovering_label = batch['label'][-self.recover_len:]
		recon_x, x = d['logits'], batch['data']
		mu, logvar = d['mu'], d['logvar']
		CE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1)[:, recovering_label] * x[:, recovering_label], -1))
		KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
		loss = CE + KLD
		return loss

	def calculate_metrics(self, batch):
		data, labels = batch['data'], batch['c_label']
		logits = self.model(batch)['logits']
		logits[data != 0] = -float("inf")

		metrics = recalls_and_ndcgs_for_ks(logits, labels, self.metric_ks)
		return metrics

	def train(self):
		epoch = self.epoch_start
		best_epoch = self.best_epoch
		accum_iter = self.accum_iter_start
		# self.validate(epoch-1, accum_iter, self.val_loader)
		best_metric = self.best_metric_at_best_epoch
		stop_training = False

		self.recover_len = self.max_len

		for epoch in range(self.epoch_start, self.num_epochs):
			if self.pilot:
				print('epoch', epoch)
			fix_random_seed_as(epoch)  # fix random seed at every epoch to make it perfectly resumable
			accum_iter = self.train_one_epoch(epoch, accum_iter, self.train_loader)
			self.lr_scheduler.step()  # step before val because state_dict is saved at val. it doesn't affect val result

			val_log_data = self.validate(epoch, accum_iter, mode='val')
			metric = val_log_data[self.best_metric]
			if metric > best_metric:
				best_metric = metric
				best_epoch = epoch
			elif (self.saturation_wait_epochs is not None) and\
					(epoch - best_epoch >= self.saturation_wait_epochs):
				if self.recover_len <= 1:
					stop_training = True  # stop training if val perf doesn't improve for saturation_wait_epochs
				else:
					best_metric = self.best_metric_at_best_epoch
					best_epoch = self.best_epoch
					self.recover_len //= 2

			if stop_training:
				# load best model
				best_model_logger = self.val_loggers[-1]
				assert isinstance(best_model_logger, BestModelLogger)
				weight_path = best_model_logger.filepath()
				if self.use_parallel:
					self.model.module.load(weight_path)
				else:
					self.model.load(weight_path)
				# self.validate(epoch, accum_iter, mode='test')  # test result at best model
				self.validate(best_epoch, accum_iter, mode='test')  # test result at best model
				break

		self.logger_service.complete({
			'state_dict': (self._create_state_dict(epoch, accum_iter)),
		})