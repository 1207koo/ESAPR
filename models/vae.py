import torch.nn as nn
from .transformer_models.heads import *
from models.base import BaseModel


class VAEModel(BaseModel):
	def __init__(self, args):
		super().__init__(args)
		self.output_info = args.output_info
		self.max_len = args.max_len
		global num_items
		self.num_items = args.num_items
		print("Got num_items:",self.num_items)
		self.encode_len = args.encode_len
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.max_len * self.num_items, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(),
			nn.Linear(2048, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 2 * self.encode_len)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.encode_len, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(),
			nn.Linear(2048, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, self.max_len * self.num_items)
		)
		self.init_weights()
		if args.headtype == 'dot':
			self.head = BertDotProductPredictionHead(args, self.token_embedding.emb)
		elif args.headtype == 'linear':
			self.head = BertLinearPredictionHead(args)

	@classmethod
	def code(cls):
		return 'vae'

	def forward(self, x):
		info = {} if self.output_info else None

		y = self.encoder(x)
		mu = y[:self.encode_len]
		logvar = y[self.encode_len:]
		sigma = torch.exp(0.5 * logvar)
		z = mu + torch.randn(self.encode_len) * sigma
		x0 = self.decoder(z)
		x0 = x0.view(-1, self.max_len, self.num_items)
		ret = {'logits':x0, 'info':info}
		if not self.training:
			# get scores (B x V) for validation
			last_logits = logits[:, -1, :]  # B x H
			ret['scores'] = self.get_scores(d, last_logits)  # B x C
		return ret

	def get_scores(self, d, logits):
		# logits : B x H or M x H
		if self.training:  # logits : M x H, returns M x V
			h = self.head(logits)  # M x V
		else:  # logits : B x H,  returns B x C
			candidates = d['candidates']  # B x C
			h = self.head(logits, candidates)  # B x C
		return h