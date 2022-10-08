import os

import torch
from torch import nn
from torch.nn import Linear, functional as F, Dropout

from Models.UnifiedClassificaionAndRegressionAgeModel import UnifiedClassificaionAndRegressionAgeModel
from Models.transformer import TransformerModel


class JoinedTransformerModel(nn.Module):
	def __init__(self, num_classes, age_interval, min_age, max_age, device=torch.device("cuda:0"), mid_feature_size=1024):
		super(JoinedTransformerModel, self).__init__()
		self.labels = range(num_classes)

		self.pretrained_model = UnifiedClassificaionAndRegressionAgeModel(num_classes, age_interval, min_age, max_age)
		pretrained_model_path = 'weights/Morph2/unified/RangerLars_lr_5e4_4096_epochs_60_batch_32_mean_var_vgg16_pretrained_recognition_bin_10_more_augs_RandomApply_warmup_cosine_recreate'
		pretrained_model_file = os.path.join(pretrained_model_path, "weights.pt")
		self.pretrained_model.load_state_dict(torch.load(pretrained_model_file), strict=False)

		self.pretrained_model.eval()
		self.pretrained_model.to(device)

		self.dimenaion_reduction_layer = nn.Linear(self.pretrained_model.num_features, mid_feature_size)

		self.transformer = TransformerModel(
			age_interval, min_age, max_age,
			mid_feature_size, mid_feature_size,
			num_outputs=num_classes,
			n_heads=8, n_encoders=4, dropout=0.3,
			mode='context').to(device)

		self.alpha = nn.Parameter(torch.randn(1, 1, 1), requires_grad=True)
		# self.alpha = torch.randn((1, 1, 1), requires_grad=True, device='cuda:0')

	def forward(self, input_images, p=0.5):
		unpacked_input = input_images.view(input_images.shape[0] * input_images.shape[1], input_images.shape[2], input_images.shape[3], input_images.shape[4])
		# unpacked_input = input_images

		# classic model
		# with torch.no_grad():
		self.pretrained_model.eval()

		base_embedding = self.pretrained_model.base_net(unpacked_input)
		base_embedding = F.leaky_relu(base_embedding)

		# first stage
		class_embed = self.pretrained_model.fc_first_stage(base_embedding)
		class_embed = F.normalize(class_embed, dim=1, p=2)
		x = F.leaky_relu(class_embed)
		x = Dropout(p)(x)

		classification_logits = self.pretrained_model.class_head(x)

		weights = nn.Softmax()(classification_logits)

		# second stage
		x = self.pretrained_model.fc_second_stage(base_embedding)
		x = F.leaky_relu(x)
		x = Dropout(p)(x)

		t = []
		for i in self.labels:
			t.append(torch.squeeze(self.pretrained_model.regression_heads[i](x) + self.pretrained_model.centers[i]) * weights[:, i])

		age_pred_per_image = torch.stack(t, dim=0).sum(dim=0) / torch.sum(weights, dim=1)

		classic_age_predictions = torch.mean(age_pred_per_image.view(input_images.shape[0], input_images.shape[1], -1), 1).squeeze()
		clasic_classification_logits = torch.mean(classification_logits.view(input_images.shape[0], input_images.shape[1], -1), 1).squeeze()

		embs = self.dimenaion_reduction_layer(base_embedding)
		embs = F.leaky_relu(embs)
		embs = F.normalize(embs, dim=1, p=2)

		packed_embs = embs.view(input_images.shape[0], input_images.shape[1], -1)

		transformer_classification_logits, transformer_age_predictions = self.transformer(packed_embs)
		transformer_classification_logits = transformer_classification_logits.squeeze()
		transformer_age_predictions = transformer_age_predictions.squeeze()

		age_pred = (self.alpha * transformer_age_predictions + (1 - self.alpha) * classic_age_predictions).squeeze()

		classification_logits = (self.alpha * clasic_classification_logits + (1 - self.alpha) * transformer_classification_logits).squeeze()

		return classification_logits, age_pred



