import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class AFADRegressorDataset(Dataset):
	def __init__(self, hdf5_file_path, min_age=15, max_age=40, age_interval=5, transform=None):
		file = h5py.File(hdf5_file_path, 'r')
		self.images = file['images'].value
		self.ages = file['ages'].value
		self.min_age = min_age
		self.max_age = max_age
		self.age_interval = age_interval
		self.num_labels = int(max_age / self.age_interval - min_age / self.age_interval + 1)
		self.transform = transform

	def __len__(self):
		return len(self.ages)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = self.images[idx]

		image = Image.fromarray(image)
		if self.transform:
			image = self.transform(image)

		age = self.ages[idx]

		classification_label = (age - self.min_age) // self.age_interval

		weights = np.zeros(self.num_labels, dtype=np.float32)
		weights[classification_label] = 1.0

		if classification_label > 0:
			weights[classification_label - 1] = 1

		if classification_label < self.num_labels - 1:
			weights[classification_label + 1] = 1

		# weights = weights / np.sum(weights)

		sample = {'image': image, 'label': age, 'weights': weights}

		return sample


