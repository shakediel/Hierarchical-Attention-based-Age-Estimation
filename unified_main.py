import json
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim import lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split

from Datasets.AFAD.AFADClassifierDataset import AFADClassifierDataset
from Datasets.Morph2.DataParser import DataParser
from Datasets.Morph2.Morph2ClassifierDataset import Morph2ClassifierDataset
from Losses.StubLoss import StubLoss
from Models.UnifiedClassificaionAndRegressionAgeModel import UnifiedClassificaionAndRegressionAgeModel
from Optimizers.RangerLars import RangerLars
from Losses.MeanVarianceLoss import MeanVarianceLoss
from Training.train_unified_model import train_unified_model
from Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Training.train_unified_model_iter import train_unified_model_iter


if __name__ == "__main__":
	seed = 42
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	torch.cuda.empty_cache()

	min_age = 15 #Morph
	max_age = 80 #Morph
	age_interval = 1 #Morph
	# min_age = 12 #AFAD
	# max_age = 43 #AFAD
	# age_interval = 8  # AFAD
	batch_size = 32
	# num_epochs = 60
	num_iters = int(1.5e5)
	random_split = False

	num_classes = int((max_age - min_age) / age_interval + 1)
	# num_classes = 5

	# Load data
	data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
	data_parser.initialize_data()

	x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test,
	if random_split:
		all_images = np.concatenate((x_train, x_test), axis=0)
		all_labels = np.concatenate((y_train, y_test), axis=0)

		x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.20, random_state=42)

	train_ds = Morph2ClassifierDataset(
		x_train,
		y_train,
		min_age,
		age_interval,
		transforms.Compose([
			transforms.RandomResizedCrop(224, (0.9, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomApply([transforms.ColorJitter(
				brightness=0.1,
				contrast=0.1,
				saturation=0.1,
				hue=0.1
			)], p=0.5),
			transforms.RandomApply([transforms.RandomAffine(
				degrees=10,
				translate=(0.1, 0.1),
				scale=(0.9, 1.1),
				shear=5,
				resample=Image.BICUBIC
			)], p=0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			transforms.RandomErasing(p=0.5)
		])
	)

	test_ds = Morph2ClassifierDataset(
		x_test,
		y_test,
		min_age,
		age_interval,
		transform=transforms.Compose([
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	)

	# train_ds = AFADClassifierDataset(
	# './Datasets/AFAD/aligned_data/afad_train.h5',
	# min_age=min_age,
	# max_age=max_age,
	# age_interval=age_interval,
	# 	transform=transforms.Compose([
	# 		transforms.RandomResizedCrop(224, (0.9, 1.0)),
	# 		transforms.RandomHorizontalFlip(),
	# 		transforms.RandomApply([transforms.ColorJitter(
	# 			brightness=0.2,
	# 			contrast=0.2,
	# 			saturation=0.2,
	# 			hue=0.2
	# 		)], p=0.5),
	# 		transforms.RandomApply([transforms.RandomAffine(
	# 			degrees=20,
	# 			translate=(0.2, 0.2),
	# 			scale=(0.8, 1.2),
	# 			shear=10,
	# 			resample=Image.BICUBIC
	# 		)], p=0.5),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	# 		transforms.RandomErasing(p=0.5)
	# 	]),
	# 	copies=1
	# )
	#
	# test_ds = AFADClassifierDataset(
	# 	'./Datasets/AFAD/aligned_data/afad_test.h5',
	# 	min_age=min_age,
	# 	max_age=max_age,
	# 	age_interval=age_interval,
	# 	transform=transforms.Compose([
	# 		transforms.Resize(224),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	# 	]),
	# 	copies=1
	# )

	image_datasets = {
		'train': train_ds,
		'val': test_ds
	}

	data_loaders = {
		'train': DataLoader(train_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True),
		'val': DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
	}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	# create model and parameters
	model = UnifiedClassificaionAndRegressionAgeModel(num_classes, age_interval, min_age, max_age)
	model.to(device)

	criterion_reg = nn.MSELoss().to(device)
	criterion_cls = torch.nn.CrossEntropyLoss().to(device)
	mean_var_criterion = MeanVarianceLoss(0, num_classes, device, lambda_mean=0.2, lambda_variance=0.05).to(device)

	optimizer = RangerLars(model.parameters(), lr=1e-3)

	# num_steps = num_epochs * len(data_loaders['train'])
	num_epochs = int(num_iters / len(data_loaders['train'])) + 1

	cosine_scheduler = CosineAnnealingLR(
		optimizer,
		T_max=num_iters
	)
	scheduler = GradualWarmupScheduler(
		optimizer,
		multiplier=1,
		total_epoch=10000,
		after_scheduler=cosine_scheduler
	)

	### Train ###
	writer = SummaryWriter('logs/Morph2/unified/iter/RangerLars_lr_1e3_4096_epochs_60_batch_32_vgg16_warmup_10k_cosine_bin_1_2')

	model_path = 'weights/Morph2/unified/iter/RangerLars_lr_1e3_4096_epochs_60_batch_32_vgg16_warmup_10k_cosine_bin_1_2'
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	best_model = train_unified_model_iter(
		model,
		criterion_reg,
		criterion_cls,
		mean_var_criterion,
		optimizer,
		scheduler,
		data_loaders,
		dataset_sizes,
		device,
		writer,
		model_path,
		num_classes,
		num_epochs=num_epochs,
		validate_at_k=500
	)

	print('saving best model')

	FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
	torch.save(best_model.state_dict(), FINAL_MODEL_FILE)

	print('fun fun in the sun, the training is done :)')
