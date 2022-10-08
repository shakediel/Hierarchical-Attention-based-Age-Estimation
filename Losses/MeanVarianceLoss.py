from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MeanVarianceLoss(nn.Module):

	def __init__(self, start_class, end_class, device, lambda_mean=0.2, lambda_variance=0.05):
		super().__init__()
		self.lambda_mean = lambda_mean
		self.lambda_variance = lambda_variance
		self.start_class = start_class
		self.end_class = end_class
		self.device = device

	def forward(self, logits, target):
		probas = nn.Softmax(dim=1)(logits)

		class_range = torch.arange(self.start_class, self.end_class, dtype=torch.float32).to(self.device)

		# mean loss
		mean = torch.squeeze((probas * class_range).sum(1, keepdim=True), dim=1)
		mean_loss = nn.MSELoss().to(self.device)(mean, target)

		# variance loss
		var = (class_range[None, :] - mean[:, None]) ** 2
		variance_loss = (probas * var).sum(1, keepdim=True).mean()

		return self.lambda_mean * mean_loss, self.lambda_variance * variance_loss


class keller_MeanVarianceLoss(nn.Module):

	def __init__(self,LamdaMean = 1,LamdaSTD  = 1,device=None,reduction = None):
		super(MeanVarianceLoss, self).__init__()

		self.LamdaMean = LamdaMean
		self.LamdaSTD  = LamdaSTD
		self.reduction = reduction
		self.device    = device


	def forward(self, input, target,IsProbability = False):

		if target.shape[0] != input.shape[0]:
			print('Error in input MeanVarianceLoss.forward')
			return 0

		Labels = torch.arange(input.shape[1]).float().cuda(self.device)

		if IsProbability == True:
			P = input
		else:
			P = F.softmax(input, 1)

		EstimatedMean = torch.mm(P,torch.unsqueeze(Labels,1))

		#soft estimate per sample
		LossMean = ((np.squeeze(EstimatedMean) - torch.squeeze(target).float()) ** 2)

		#STD per sample
		LossSTD = (  P*((Labels-EstimatedMean)**2)).sum(1)

		Result = self.LamdaMean*sqrt(LossMean.mean()) + self.LamdaSTD*sqrt(LossSTD.mean())

		#if self.reduction == 'mean':

		return Result