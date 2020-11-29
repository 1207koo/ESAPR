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
			nn.BatchNorm1d(self.num_items + 1),
			nn.Dropout(0.5),
			nn.Linear(self.num_items + 1, 1024),
			nn.Tanh(),
			nn.Linear(1024, 512),
			nn.Tanh(),
			nn.Linear(512, 2 * self.encode_len)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.encode_len, 512),
			nn.Tanh(),
			nn.Linear(512, 1024),
			nn.Tanh(),
			nn.Linear(1024, self.num_items + 1)
		)
		self.init_weights()
	
	def init_weights(self):
        for layer in self.encoder:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.decoder:
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
		info = {} if self.output_info else None

		y = self.encoder(x)
		mu = y[:, :self.encode_len]
		logvar = y[:, self.encode_len:]
		sigma = torch.exp(0.5 * logvar)
		z = mu
		if self.training:
			z = mu + torch.randn_like(sigma) * sigma
		x0 = self.decoder(z)
		ret = {'logits':x0, 'mu':mu, 'logvar':logvar, 'info':info}
		return ret