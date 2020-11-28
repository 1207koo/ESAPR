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
        self.__beta = 0.0
        self.anneal_amount = 1.0 / 64
		self.current_best_metric = 0.0
		self.anneal_cap = 1.0

	@classmethod
	def code(cls):
		return 'vae'

	def add_extra_loggers(self):
		pass

	def log_extra_train_info(self, log_data):
		pass

    @property
    def beta(self):
        if self.model.training:
            self.__beta = min(self.__beta + self.anneal_amount, self.anneal_cap)
        return self.__beta
	
	def calculate_loss(self, batch):
		d = self.model(batch)
		recon_x, x = d['logits'], batch['data']
		mu, logvar = d['mu'], d['logvar']
		CE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
		KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
		loss = CE + self.beta * KLD
		return loss

	def calculate_metrics(self, batch):
		data, labels = batch['data'], batch['c_label']
		candidates = batch['candidates']
		logits = self.model(batch)['logits']
		logits[data != 0] = -float("inf")
		scores = logits.gather(1, candidates)

		metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        if self.finding_best_beta:
            if self.current_best_metric < metrics[self.best_metric]:
                self.current_best_metric = metrics[self.best_metric]
                self.best_beta = self.__beta
		return metrics
