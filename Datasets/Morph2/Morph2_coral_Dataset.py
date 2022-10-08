import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image


class Morph2_coral_Dataset(Dataset):
	def __init__(self, hdf5_file_path, min_age, age_interval, transform=None):
		file = h5py.File(hdf5_file_path, 'r')
		self.images = file['images'].value
		self.ages = file['ages'].value
		self.races = file['races'].value
		self.genders = file['genders'].value
		self.min_age = min_age
		self.age_interval = age_interval
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
		label = (age - self.min_age) // self.age_interval

		sample = {'image': image, 'label': label}

		return sample

