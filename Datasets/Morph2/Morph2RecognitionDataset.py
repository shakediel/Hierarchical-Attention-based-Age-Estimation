import json

import torch
from PIL import Image
from torch.utils.data import Dataset


class Morph2RecognitionDataset(Dataset):
	def __init__(self, images, metadata, ids, transform=None):
		self.images = images[:, :, :, [2, 1, 0]]
		self.metadata = metadata
		self.transform = transform
		self.ids = list(ids)

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
		id = metadata['id_num']

		label = self.ids.index(id)

		return {'image': image, 'label': label}
