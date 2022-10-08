import os

import torch
from PIL import Image
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim import lr_scheduler

from Datasets.Morph2.DataParser import DataParser
from Datasets.Morph2.Morph2ClassifierDataset import Morph2ClassifierDataset
from Datasets.Morph2.Morph2CoralDataset import Morph2CoralDataset
from Datasets.Morph2.Morph2RegressorDataset import Morph2RegressorDataset
from Optimizers.RangerLars import RangerLars
from Training.train_mean_variance_model import train_mean_variance_model
from Losses.MeanVarianceLoss import MeanVarianceLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

min_age = 15
max_age = 80
age_interval = 5
# num_classes = max_age - min_age + 1
batch_size = 128
num_epochs = 50

num_classes = int((max_age - min_age) / age_interval + 1)

# Load data
data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
data_parser.initialize_data()

train_ds = Morph2ClassifierDataset(
	data_parser.x_train,
	data_parser.y_train,
	min_age,
	age_interval,
	transforms.Compose([
		transforms.RandomResizedCrop(224, (0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(
			brightness=0.1,
			contrast=0.1,
			saturation=0.1,
			hue=0.1
		),
		transforms.RandomAffine(
			degrees=10,
			translate=(0.1, 0.1),
			scale=(0.9, 1.1),
			shear=5,
			resample=Image.BICUBIC
		),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
)

test_ds = Morph2ClassifierDataset(
	data_parser.x_test,
	data_parser.y_test,
	min_age,
	age_interval,
	transform=transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
)

image_datasets = {
	'train': train_ds,
	'val': test_ds
}

data_loaders = {
	x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
	for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# create model and parameters

model = models.resnet34(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(device)
optimizer = RangerLars(model.parameters(), lr=1e-3, weight_decay=5e-5)

mean_var_criterion = MeanVarianceLoss(0, num_classes, device, lambda_mean=0.2, lambda_variance=0.05).to(device)
cce_criterion = torch.nn.CrossEntropyLoss().to(device)

scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Train
writer = SummaryWriter('logs/Morph2/mean_variance_loss/RangerLars_lr_1e3_weight_decay_5e5_augs_stepsize_15')

best_model = train_mean_variance_model(
	model,
	mean_var_criterion,
	cce_criterion,
	optimizer,
	scheduler,
	data_loaders,
	dataset_sizes,
	device,
	writer,
	num_epochs=num_epochs
)

print('saving best model')

model_path = 'weights/Morph2/mean_variance_loss/RangerLars_lr_1e3_weight_decay_5e5_augs_stepsize_15'
if not os.path.exists(model_path):
	os.makedirs(model_path)
FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
torch.save(best_model.state_dict(), FINAL_MODEL_FILE)

print('exiting')
