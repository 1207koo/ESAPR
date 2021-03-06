from loggers import *

from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
from utils import AverageMeterSet
from utils import fix_random_seed_as

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class VAETrainer(AbstractTrainer):
	def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
		super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
		self.max_len = args.max_len
		self.total_anneal_steps = args.total_anneal_steps
		self.anneal_cap = args.anneal_cap
		self.update_count = 0
		self.recover_len = self.max_len
		self.train_transfer = args.train_transfer
		self.scaled_loss = args.scaled_loss
		self.best_model_transfer = args.best_model_transfer
		self.decrease_dropout = args.decrease_dropout

	@classmethod
	def code(cls):
		return 'vae'

	def add_extra_loggers(self):
		pass

	def log_extra_train_info(self, log_data):
		pass
	
	def calculate_loss(self, batch, anneal):
		d = self.model(batch)

		recon_x = d['logits']
		weight = batch['weight']
		recon_sum = torch.sum(F.log_softmax(recon_x, 1).gather(1, batch['label'][:, -self.recover_len:]) * weight[:, -self.recover_len:], dim=-1)
		if self.scaled_loss:
			recon_sum = recon_sum * torch.sum(weight, dim=-1) / torch.sum(weight[:, -self.recover_len:], dim=-1)
		CE = -torch.mean(recon_sum)

		mu, logvar = d['mu'], d['logvar']
		KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
		loss = CE + anneal * KLD
		return loss

	def calculate_metrics(self, batch):
		data, labels = batch['data'], batch['c_label']
		candidates = batch['candidates']
		logits = self.model(batch)['logits']
		logits[data != 0] = -float("inf")
		logits[:, 0] = -float("inf")
		scores = logits.gather(1, candidates)

		metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
		return metrics

	def train(self):
		epoch = self.epoch_start
		best_epoch = self.best_epoch
		in_best_epoch = self.best_epoch
		accum_iter = self.accum_iter_start
		# self.validate(epoch-1, accum_iter, self.val_loader)
		best_metric = self.best_metric_at_best_epoch
		stop_training = False
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
				in_best_epoch = epoch
			elif (self.saturation_wait_epochs is not None) and\
					(epoch - in_best_epoch >= self.saturation_wait_epochs):
				# stop training if val perf doesn't improve for saturation_wait_epochs
				if self.recover_len > 1 and self.train_transfer:
					recover_len_before = self.recover_len
					self.recover_len //= 2
					print('recover_len decreased:', recover_len_before, '->', self.recover_len)
					if self.decrease_dropout:
						self.model.module.drop = nn.Dropout(self.model.module.dropout * self.recover_len / self.max_len)
					in_best_epoch = epoch
					if self.best_model_transfer:
						print('Loading Best Model...')
						best_model_logger = self.val_loggers[-1]
						assert isinstance(best_model_logger, BestModelLogger)
						weight_path = best_model_logger.filepath()
						if self.use_parallel:
							self.model.module.load(weight_path)
						else:
							self.model.load(weight_path)
						print('Validating Model...')
						self.validate(epoch, accum_iter, mode='val', doLog=False)
				else:
					stop_training = True 

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
		
	def train_one_epoch(self, epoch, accum_iter, train_loader, **kwargs):
		self.model.train()

		average_meter_set = AverageMeterSet()
		num_instance = 0
		tqdm_dataloader = tqdm(train_loader) if not self.pilot else train_loader

		for batch_idx, batch in enumerate(tqdm_dataloader):
			if self.pilot and batch_idx >= self.pilot_batch_cnt:
				# print('Break training due to pilot mode')
				break
			batch_size = next(iter(batch.values())).size(0)
			batch = {k:v.to(self.device) for k, v in batch.items()}
			num_instance += batch_size

			if self.total_anneal_steps > 0:
				anneal = min(self.anneal_cap,  1. * self.update_count / self.total_anneal_steps)
			else:
				anneal = self.anneal_cap
				
			self.optimizer.zero_grad()
			loss = self.calculate_loss(batch, anneal)
			if isinstance(loss, tuple):
				loss, extra_info = loss
				for k, v in extra_info.items():
					average_meter_set.update(k, v)
			loss.backward()

			if self.clip_grad_norm is not None:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

			self.optimizer.step()
			
			self.update_count += 1

			average_meter_set.update('loss', loss.item())
			if not self.pilot:
				tqdm_dataloader.set_description(
					'Epoch {}, loss {:.3f} '.format(epoch, average_meter_set['loss'].avg))

			accum_iter += batch_size

			if self._needs_to_log(accum_iter):
				if not self.pilot:
					tqdm_dataloader.set_description('Logging')
				log_data = {
					# 'state_dict': (self._create_state_dict()),
					'epoch': epoch,
					'accum_iter': accum_iter,
				}
				log_data.update(average_meter_set.averages())
				log_data.update(kwargs)
				self.log_extra_train_info(log_data)
				self.logger_service.log_train(log_data)

		log_data = {
			# 'state_dict': (self._create_state_dict()),
			'epoch': epoch,
			'accum_iter': accum_iter,
			'num_train_instance': num_instance,
		}
		log_data.update(average_meter_set.averages())
		log_data.update(kwargs)
		self.log_extra_train_info(log_data)
		self.logger_service.log_train(log_data)
		return accum_iter

	def validate(self, epoch, accum_iter, mode, doLog=True, **kwargs):
		if mode == 'val':
			loader = self.val_loader
		elif mode == 'test':
			loader = self.test_loader
		else:
			raise ValueError

		self.model.eval()

		average_meter_set = AverageMeterSet()
		num_instance = 0

		with torch.no_grad():
			tqdm_dataloader = tqdm(loader) if not self.pilot else loader
			for batch_idx, batch in enumerate(tqdm_dataloader):
				if self.pilot and batch_idx >= self.pilot_batch_cnt:
					# print('Break validation due to pilot mode')
					break
				batch = {k:v.to(self.device) for k, v in batch.items()}
				batch_size = next(iter(batch.values())).size(0)
				num_instance += batch_size

				metrics = self.calculate_metrics(batch)

				for k, v in metrics.items():
					average_meter_set.update(k, v)
				if not self.pilot:
					description_metrics = ['NDCG@%d' % k for k in self.metric_ks] +\
										  ['Recall@%d' % k for k in self.metric_ks]
					description = '{}: '.format(mode.capitalize()) + ', '.join(s + ' {:.4f}' for s in description_metrics)
					description = description.replace('NDCG', 'N').replace('Recall', 'R')
					description = description.format(*(average_meter_set[k].avg for k in description_metrics))
					tqdm_dataloader.set_description(description)

			log_data = {
				'state_dict': (self._create_state_dict(epoch, accum_iter)),
				'epoch': epoch,
				'accum_iter': accum_iter,
				'num_eval_instance': num_instance,
			}
			log_data.update(average_meter_set.averages())
			log_data.update(kwargs)
			if doLog:
				if mode == 'val':
					self.logger_service.log_val(log_data)
				elif mode == 'test':
					self.logger_service.log_test(log_data)
				else:
					raise ValueError
		return log_data