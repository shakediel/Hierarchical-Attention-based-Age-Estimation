import json

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class Morph2RegressorDataset(Dataset):
	def __init__(self, images, metadata, min_age, age_interval, num_labels, transform=None):
		self.images = images[:, :, :, [2, 1, 0]]
		self.metadata = metadata
		self.transform = transform
		self.min_age = min_age
		self.age_interval = age_interval
		self.num_labels = num_labels

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = self.images[idx]
		image = Image.fromarray(image)
		if self.transform:
			image = self.transform(image)

		metadata = json.loads(self.metadata[idx])
		age = int(metadata['age'])
		classification_label = (age - self.min_age) // self.age_interval

		weights = np.zeros(self.num_labels, dtype=np.float32)
		weights[classification_label] = 1.0
		if classification_label > 0:
			weights[classification_label - 1] = 1
		if classification_label < self.num_labels - 1:
			weights[classification_label + 1] = 1

		if classification_label > 1:
			weights[classification_label - 2] = 0.2
		if classification_label < self.num_labels - 2:
			weights[classification_label + 2] = 0.2

		# weights = weights / np.sum(weights)

		sample = {'image': image, 'label': age, 'weights': weights}

		return sample


