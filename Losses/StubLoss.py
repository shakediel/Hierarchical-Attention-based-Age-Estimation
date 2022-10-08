import torch
from torch import nn


class StubLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, preds, target):
		zeros = torch.zeros_like(target)
		return zeros
