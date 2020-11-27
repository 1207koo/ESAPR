import torch
import torch.nn as nn
from models.base import BaseModel


class VAEModel(BaseModel):
	def __init__(self, args):
		super().__init__(args)
		self.output_info = args.output_info
		self.max_len = args.max_len
		self.num_items = args.num_items
		self.encode_len = args.encode_len
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.max_len * (self.num_items + 1), 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 2 * self.encode_len)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.encode_len, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, self.max_len * (self.num_items + 1))
		)
		self.init_weights()

	@classmethod
	def code(cls):
		return 'vae'

	def forward(self, d):
		x = d['data']
		info = {} if self.output_info else None

		y = self.encoder(x)
		mu = y[:, :self.encode_len]
		logvar = y[:, self.encode_len:]
		sigma = torch.exp(0.5 * logvar)
		z = mu + torch.randn_like(sigma) * sigma
		x0 = self.decoder(z)
		x0 = x0.view(-1, self.max_len, self.num_items + 1)
		ret = {'logits':x0, 'info':info}
		if not self.training:
			# get scores (B x V) for validation
			last_logits = x0[:, -1, :].squeeze()  # B x H
			ret['scores'] = torch.FloatTensor([l[c] for l, c in zip(last_logits, d['candidates'])])  # B x C
		return ret