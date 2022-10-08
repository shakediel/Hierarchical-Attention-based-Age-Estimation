import json

import torch
from torch.utils.data import Dataset
from PIL import Image


class Morph2CoralDataset(Dataset):
	def __init__(self, images, metadata, num_classes, transform=None, min_age=15):
		self.images = images[:, :, :, [2, 1, 0]]
		self.metadata = metadata
		self.transform = transform
		self.num_classes = num_classes
		self.min_age = min_age

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
		class_label = int(metadata['age']) - self.min_age
		levels = [1] * class_label + [0] * (self.num_classes - 1 - class_label)
		levels = torch.tensor(levels, dtype=torch.float32)

		return image, class_label, levels
