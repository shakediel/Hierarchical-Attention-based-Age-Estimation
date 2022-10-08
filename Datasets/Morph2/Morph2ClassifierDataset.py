import json

import torch
from torch.utils.data import Dataset
from PIL import Image


class Morph2ClassifierDataset(Dataset):
	def __init__(self, images, metadata, min_age, age_interval, transform=None, copies=1):
		self.images = images[:, :, :, [2, 1, 0]]
		self.metadata = metadata
		self.transform = transform
		self.min_age = min_age
		self.age_interval = age_interval
		self.copies = copies

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		orig_image = self.images[idx]
		orig_image = Image.fromarray(orig_image)
		if self.copies > 1:
			images = []
			for i in range(self.copies):
				images.append(self.transform(orig_image))
			image = torch.stack(images)
		else:
			image = self.transform(orig_image)

		metadata = json.loads(self.metadata[idx])
		age = int(metadata['age'])
		label = (age - self.min_age) // self.age_interval

		sample = {'image': image, 'classification_label': label, 'age': age}

		return sample