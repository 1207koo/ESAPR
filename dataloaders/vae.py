from .base import AbstractDataloader

import torch
from scipy import sparse
import torch.utils.data as data_utils


class VAEDataloader(AbstractDataloader):
	@classmethod
	def code(cls):
		return 'vae'

	def _get_dataset(self, mode):
		if mode == 'train':
			return self._get_train_dataset()
		elif mode == 'val':
			return self._get_eval_dataset('val')
		else:
			return self._get_eval_dataset('test')

	def _get_train_dataset(self):
		train_ranges = self.train_targets
		dataset = VAETrainDataset(self.args, self.dataset, train_ranges)
		return dataset

	def _get_eval_dataset(self, mode):
		positions = self.validation_targets if mode=='val' else self.test_targets
		dataset = VAEEvalDataset(self.args, self.dataset, self.test_negative_samples, positions)
		return dataset


class VAETrainDataset(data_utils.Dataset):
	def __init__(self, args, dataset, train_ranges):
		self.args = args
		self.user2dict = dataset['user2dict']
		self.users = sorted(self.user2dict.keys())
		self.train_window = args.train_window
		self.max_len = args.max_len
		self.num_users = args.num_users
		self.num_items = args.num_items
		self.train_ranges = train_ranges
		self.weight_type = args.weight_type
		self.weight_constant = args.weight_constant
		self.aug_prob = args.aug_prob

		self.index2user_and_offsets = self.populate_indices()

		self.output_timestamps = args.dataloader_output_timestamp
		self.output_days = args.dataloader_output_days
		self.output_user = args.dataloader_output_user

	def populate_indices(self):
		index2user_and_offsets = {}
		i = 0
		T = self.max_len
		W = self.train_window

		# offset is exclusive
		for user, pos in self.train_ranges:
			if W is None or W == 0:
				offsets = [pos]
			else:
				offsets = list(range(pos, T-1, -W))  # pos ~ T
				if len(offsets) == 0:
					offsets = [pos]
			for offset in offsets:
				index2user_and_offsets[i] = (user, offset)
				i += 1
		return index2user_and_offsets

	def __len__(self):
		return len(self.index2user_and_offsets)

	def __getitem__(self, index):
		user, offset = self.index2user_and_offsets[index]
		seq = self.user2dict[user]['items']
		beg = max(0, offset-self.max_len)
		end = offset  # exclude offset (meant to be)
		seq = seq[beg:end]

		label = torch.zeros((self.max_len), dtype=torch.long)
		label[-len(seq):] = torch.LongTensor(seq)

		data = torch.zeros(self.num_items + 1)
		N = label.size()[0]

		if self.weight_type == 'exp_stair':
			weight_index = N - 1 - torch.arange(N)
			weight = self.weight_constant ** (-torch.floor(torch.log2(weight_index + 0.5)))
		elif self.weight_type == 'exp':
			weight_index = N - 1 - torch.arange(N)
			weight = self.weight_constant ** (-torch.log2(weight_index + 0.5))
		elif self.weight_type == 'linear':
			weight = self.weight_constant * (torch.arange(N) + 1) / N
		else: # constant
			weight = self.weight_constant * torch.ones(N)

		aug = torch.ones(N - 1)
		prob = torch.rand(aug.size())
		aug[prob < self.aug_prob / 2.0] *= 0.5
		aug[prob >= 1.0 - self.aug_prob / 2.0] *= 2.0
		data[label[:-1]] += weight[:-1] * aug

		d = {}
		d['data'] = data
		d['label'] = label
		d['weight'] = weight

		padding_len = self.max_len - len(seq)

		if self.output_timestamps:
			timestamps = self.user2dict[user]['timestamps']
			timestamps = timestamps[beg:end]
			timestamps = [0] * padding_len + timestamps
			d['timestamps'] = torch.LongTensor(timestamps)

		if self.output_days:
			days = self.user2dict[user]['days']
			days = days[beg:end]
			days = [0] * padding_len + days
			d['days'] = torch.LongTensor(days)

		if self.output_user:
			d['users'] = torch.LongTensor([user])
		return d


class VAEEvalDataset(data_utils.Dataset):
	def __init__(self, args, dataset, negative_samples, positions):
		self.user2dict = dataset['user2dict']
		self.positions = positions
		self.max_len = args.max_len
		self.num_items = args.num_items
		self.negative_samples = negative_samples

		self.output_timestamps = args.dataloader_output_timestamp
		self.output_days = args.dataloader_output_days
		self.output_user = args.dataloader_output_user
		self.weight_type = args.weight_type
		self.weight_constant = args.weight_constant

	def __len__(self):
		return len(self.positions)

	def __getitem__(self, index):
		user, pos = self.positions[index]
		seq = self.user2dict[user]['items']

		beg = max(0, pos + 1 - self.max_len)
		end = pos + 1
		seq = seq[beg:end]

		negs = self.negative_samples[user]
		answer = [seq[-1]]
		candidates = answer + negs
		c_labels = [1] * len(answer) + [0] * len(negs)

		label = torch.zeros((self.max_len), dtype=torch.long)
		label[-len(seq):] = torch.LongTensor(seq)

		data = torch.zeros(self.num_items + 1)
		N = label.size()[0]

		if self.weight_type == 'exp_stair':
			weight_index = N - 1 - torch.arange(N)
			weight = self.weight_constant ** (-torch.floor(torch.log2(weight_index + 0.5)))
		elif self.weight_type == 'exp':
			weight_index = N - 1 - torch.arange(N)
			weight = self.weight_constant ** (-torch.log2(weight_index + 0.5))
		elif self.weight_type == 'linear':
			weight = self.weight_constant * (torch.arange(N) + 1) / N
		else: # constant
			weight = self.weight_constant * torch.ones(N)

		data[label[:-1]] += weight[:-1]

		d = {}
		d['data'] = data
		d['candidates'] = torch.LongTensor(candidates)
		d['c_label'] = torch.LongTensor(c_labels)
		d['label'] = label
		d['weight'] = weight

		padding_len = self.max_len - len(seq)

		if self.output_timestamps:
			timestamps = self.user2dict[user]['timestamps']
			timestamps = timestamps[beg:end]
			timestamps = [0] * padding_len + timestamps
			d['timestamps'] = torch.LongTensor(timestamps)

		if self.output_days:
			days = self.user2dict[user]['days']
			days = days[beg:end]
			days = [0] * padding_len + days
			d['days'] = torch.LongTensor(days)

		if self.output_user:
			d['users'] = torch.LongTensor([user])
		return d
