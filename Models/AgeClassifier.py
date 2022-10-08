from torch import nn as nn
from torch.nn import Linear, functional as F, Dropout

from Models.inception_resnet_v1 import InceptionResnetV1


class AgeClassifier(nn.Module):
	def __init__(self, num_classes):
		super(AgeClassifier, self).__init__()

		self.Labels = range(num_classes)

		k = 256

		self.FaceNet = InceptionResnetV1(pretrained='vggface2')

		self.fc_class = Linear(512, k)
		self.fc_class1 = Linear(k, num_classes)

	def freeze_base_cnn(self, should_freeze=True):
		for param in self.FaceNet.parameters():
			param.requires_grad = not should_freeze

	def freeze_classification_layers(self, should_freeze=True):
		for param in self.fc_class.parameters():
			param.requires_grad = not should_freeze

		for param in self.fc_class1.parameters():
			param.requires_grad = not should_freeze

	# output CNN
	def forward(self, input_images, p=0.5):

		base_embedding = self.FaceNet(input_images)
		base_embedding = F.leaky_relu(base_embedding)

		class_embed = self.fc_class(base_embedding)

		# L2 normalization
		class_embed = F.normalize(class_embed, dim=1, p=2) #todo: check this line influence on results

		x = F.leaky_relu(class_embed)
		x = Dropout(p)(x)
		x = self.fc_class1(x)

		result = dict()
		result['Class'] = x
		result['ClassEmbed'] = class_embed
		result['Base'] = base_embedding
		return result['Class']
