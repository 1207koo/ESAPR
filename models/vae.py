import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

import numpy as np


class VAEModel(BaseModel):
	def __init__(self, args):
		super().__init__(args)
		self.output_info = args.output_info
		self.max_len = args.max_len
		self.num_items = args.num_items
		self.encoder_hidden_layer = args.encoder_hidden_layer
		self.encode_len = args.encode_len

		self.encoder_shape = [self.num_items + 1] + self.encoder_hidden_layer + [2 * self.encoder_len]
		self.decoder_shape = [self.encoder_len] + self.encoder_hidden_layer[::-1] + [self.num_items + 1]
		self.dropout = args.dropout

		self.drop = nn.Dropout(self.dropout)
		self.encoder = nn.ModuleList(nn.Linear(c_in, c_out) for c_in, c_out in zip(self.encoder_shape[:-1], self.encoder_shape[1:]))
		self.decoder = nn.ModuleList(nn.Linear(c_in, c_out) for c_in, c_out in zip(self.decoder_shape[:-1], self.decoder_shape[1:]))
		
		self.init_weights()
	
	def init_weights(self):
		for layer in self.encoder:
			if isinstance(layer, nn.Linear):
				# Xavier Initialization for weights
				size = layer.weight.size()
				fan_out = size[0]
				fan_in = size[1]
				std = np.sqrt(2.0/(fan_in + fan_out))
				layer.weight.data.normal_(0.0, std)

				# Normal Initialization for Biases
				layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.decoder:
			if isinstance(layer, nn.Linear):
				# Xavier Initialization for weights
				size = layer.weight.size()
				fan_out = size[0]
				fan_in = size[1]
				std = np.sqrt(2.0/(fan_in + fan_out))
				layer.weight.data.normal_(0.0, std)

				# Normal Initialization for Biases
				layer.bias.data.normal_(0.0, 0.001)

	@classmethod
	def code(cls):
		return 'vae'

	def forward(self, d):
		x = d['data']
		x = F.normalize(x)
		x = self.drop(x)
		info = {} if self.output_info else None

		for i, layer in enumerate(self.encoder):
			x = layer(x)
			if i < len(self.encoder) - 1:
				 x = F.tanh(x)
		mu = x[:, :self.encode_len]
		logvar = x[:, self.encode_len:]
		sigma = torch.exp(0.5 * logvar)
		z = mu
		if self.training:
			z = mu + torch.randn_like(sigma) * sigma
		x0 = z
		for i, layer in enumerate(self.decoder):
			x0 = layer(x0)
			if i < len(self.decoder) - 1:
				 x0 = F.tanh(x0)
		ret = {'logits':x0, 'mu':mu, 'logvar':logvar, 'info':info}
		return ret