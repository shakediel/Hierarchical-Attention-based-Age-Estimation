import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from torch.nn import Dropout


class Noop(nn.Module):
	def __init__(self):
		super(Noop, self).__init__()

	def forward(self, x):
		return x


class NoopAddDim(nn.Module):
	def __init__(self):
		super(NoopAddDim, self).__init__()

	def forward(self, x):
		return x.unsqueeze(-1)


def add_to_dim(x, num_dims, dim=0):
	while len(x.shape) < num_dims:
		x = x.unsqueeze(dim)
	return x


class DummyEmbd(nn.Module):
	def __init__(self, out_size, dtype=torch.float32):
		super(DummyEmbd, self).__init__()
		self.out_size = out_size
		self.dtype = dtype

	def forward(self, x):
		return torch.zeros(x.shape + (self.out_size,), dtype=self.dtype, device=x.device)


def soft_cross_entropy(input, target):
	return -(target * F.log_softmax(input, dim=1)).sum(1).mean(0)


class FocalCrossEntropy:
	def __init__(self, gamma):
		self.gamma = gamma

	def __call__(self, pred, target):
		if pred.shape != target.shape:
			targets = torch.zeros_like(pred)
			targets[torch.arange(target.shape[0]), target] = 1
		else:
			targets = target
		return -(torch.pow(1 - F.softmax(pred, dim=1), self.gamma) * targets * F.log_softmax(pred, dim=1)).sum(1).mean(
			0)


class TransformerModel(nn.Module):
	def __init__(
			self,
			age_intareval,
			min_age,
			max_age,
			in_size,
			dim_feedforward,
			n_heads=4,
			n_encoders=4,
			num_outputs=8,
			dropout=0.1,
			mode='mean'
	):
		super(TransformerModel, self).__init__()

		self.mode = mode

		self.age_intareval = age_intareval
		self.min_age = min_age
		self.max_age = max_age
		self.labels = range(num_outputs)

		self.in_size = in_size
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=in_size,
			nhead=n_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout)
		self.encoder_norm = nn.LayerNorm(in_size)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, n_encoders, self.encoder_norm)

		self.cls_token = nn.Parameter(torch.randn(1, 1, in_size), requires_grad=True)
		self.to_cls_token = nn.Identity()

		self.batch_norm = nn.BatchNorm1d(in_size)

		self.fc_first_stage = nn.Linear(in_size, in_size)
		self.class_head = nn.Linear(in_size, num_outputs)

		self.fc_second_stage = nn.Linear(in_size, in_size)

		self.regression_heads = []
		self.centers = []
		for i in self.labels:
			self.regression_heads.append(nn.Linear(in_size, 1))
			center = min_age + 0.5 * age_intareval + i * age_intareval
			self.centers.append(center)

		self.regression_heads = nn.ModuleList(self.regression_heads)


	def forward_context(self, x):
		cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=x.shape[0])

		enc_in = torch.cat((cls_tokens, x), dim=1).permute(1, 0, -1)

		x = self.encoder(enc_in)
		x = x.permute(1, 0, -1)

		base_embedding = self.to_cls_token(x[:, 0])

		# base_embedding = Dropout(0.5)(base_embedding)
		base_embedding = self.batch_norm(base_embedding)

		# first stage
		class_embed = self.fc_first_stage(base_embedding)
		x = F.leaky_relu(class_embed)
		classification_logits = self.class_head(x)
		weights = nn.Softmax()(classification_logits)

		# second stage
		x = self.fc_second_stage(base_embedding)
		x = F.leaky_relu(x)

		t = []
		for i in self.labels:
			t.append(torch.squeeze(self.regression_heads[i](x) + self.centers[i]) * weights[:, i])

		age_pred = torch.stack(t, dim=0).sum(dim=0) / torch.sum(weights, dim=1)

		return classification_logits, age_pred

	def forward_mean(self, x):
		# enc_in = x.permute(1, 0, -1)
		#
		# x = self.encoder(enc_in)
		# x = x.permute(1, 0, -1)

		base_embedding = x.mean(axis=1)
		# base_embedding = x

		# first stage
		class_embed = self.fc_first_stage(base_embedding)
		x = F.leaky_relu(class_embed)
		classification_logits = self.class_head(x)
		weights = nn.Softmax()(classification_logits)

		# second stage
		x = self.fc_second_stage(base_embedding)
		x = F.leaky_relu(x)

		t = []
		for i in self.labels:
			t.append(torch.squeeze(self.regression_heads[i](x) + self.centers[i]) * weights[:, i])
			# t.append(torch.squeeze(self.regression_heads[i](x) + self.centers[i]) * weights[:, i])

		age_pred = torch.stack(t, dim=0).sum(dim=0) / torch.sum(weights, dim=1)

		return classification_logits, age_pred


	def forward(self, x):

		if self.mode == 'mean':
			return self.forward_mean(x)

		if self.mode == 'context':
			return self.forward_context(x)

