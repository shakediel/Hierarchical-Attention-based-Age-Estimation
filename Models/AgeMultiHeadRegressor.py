import torch
from torch import nn as nn
from torch.nn import Linear, functional as F, Dropout

from Models.inception_resnet_v1 import InceptionResnetV1


class AgeMultiHeadRegressor(nn.Module):
	def __init__(self, num_classes, age_intareval, min_age, max_age, device=torch.device("cuda:0")):
		super(AgeMultiHeadRegressor, self).__init__()
		self.device = device
		self.age_intareval = age_intareval
		self.min_age = min_age
		self.max_age = max_age
		self.labels = range(num_classes)

		k = 256

		self.FaceNet = InceptionResnetV1(pretrained='vggface2')

		self.fc_regress = Linear(512, k)

		# self.regression_level_1 = []
		self.regression_heads = []
		self.centers = []

		for i in self.labels:
			# self.regression_level_1.append(Linear(k, k))
			self.regression_heads.append(Linear(k, 1))
			center = min_age + 0.5 * age_intareval + i * age_intareval
			self.centers.append(center)

		# self.regression_level_1 = nn.ModuleList(self.regression_level_1)
		self.regression_heads = nn.ModuleList(self.regression_heads)

	def freeze_base_cnn(self, should_freeze=True):
		for param in self.FaceNet.parameters():
			param.requires_grad = not should_freeze

	def forward(self, input_images, weights, p=0.5):
		base_embedding = self.FaceNet(input_images)
		base_embedding = F.leaky_relu(base_embedding)

		x = self.fc_regress(base_embedding)
		x = F.leaky_relu(x)
		x = Dropout(p)(x)

		# _, preds = torch.max(weights, 1)
		# one_hot = torch.zeros(weights.size()).to(self.device)
		# one_hot[torch.arange(weights.size()[0]), preds] = 1

		t = []
		for i in self.labels:
			# x = self.regression_level_1[i](x)
			# x = F.leaky_relu(x)
			t.append(torch.squeeze(self.regression_heads[i](x) + self.centers[i]) * weights[:, i])

		result = torch.stack(t, dim=0).sum(dim=0) / torch.sum(weights, dim=1)

		return result