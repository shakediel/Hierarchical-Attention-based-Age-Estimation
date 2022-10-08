import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image


class AFADClassifierDataset(Dataset):
	def __init__(self, hdf5_file_path, min_age=15, max_age=40, age_interval=5, transform=None, copies=1):
		file = h5py.File(hdf5_file_path, 'r')
		self.images = list(file['images'])
		self.ages = list(file['ages'])
		self.min_age = min_age
		self.max_age = max_age
		self.age_interval = age_interval
		self.transform = transform
		self.copies = copies

	def __len__(self):
		return len(self.ages)

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

		age = self.ages[idx]
		label = (age - self.min_age) // self.age_interval

		sample = {'image': image, 'classification_label': label, 'age': age}

		return sample


