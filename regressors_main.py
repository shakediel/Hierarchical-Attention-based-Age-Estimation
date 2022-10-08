import os

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from Datasets.Morph2.DataParser import DataParser
from Datasets.AFAD.AFADRegressorDataset import AFADRegressorDataset
from Training.train_regression_model import train_regression_model
from Models.AgeMultiHeadRegressor import AgeMultiHeadRegressor

from Optimizers.RangerLars import RangerLars

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

age_interval = 5
min_age = 15
max_age = 40
BATCH_SIZE = 128

num_labels = int(max_age / age_interval - min_age / age_interval + 1)

# Load data
# data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
# data_parser.initialize_data()
#
# train_ds = Morph2RegressorDataset(
# 	data_parser.x_train,
# 	data_parser.y_train,
# 	min_age,
# 	age_intareval,
# 	num_labels,
# 	transform=transforms.Compose([
# 		transforms.RandomResizedCrop(160, (0.95, 1.0)),
# 		transforms.RandomHorizontalFlip(),
# 		transforms.ToTensor()
# 	])
# )
#
# test_ds = Morph2RegressorDataset(
# 	data_parser.x_test,
# 	data_parser.y_test,
# 	min_age,
# 	age_intareval,
# 	num_labels,
# 	transform=transforms.Compose([
# 		transforms.ToTensor()
# 	])
# )


train_ds = AFADRegressorDataset(
	'./Datasets/AFAD/aligned_data/afad_train.h5',
	min_age=min_age,
	max_age=max_age,
	age_interval=age_interval,
	transform=transforms.Compose([
		# transforms.Normalize((103.939, 116.779, 123.68), (1, 1, 1)),
		transforms.RandomResizedCrop(160, (0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(degrees=10, resample=Image.BICUBIC),
		transforms.ToTensor()
	])
)
test_ds = AFADRegressorDataset(
	'./Datasets/AFAD/aligned_data/afad_test.h5',
	min_age=min_age,
	max_age=max_age,
	age_interval=age_interval,
	transform=transforms.Compose([
		transforms.ToTensor()
	])
)

image_datasets = {
	'train': train_ds,
	'val': test_ds
}

data_loaders = {
	x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# create model and parameters
multihead_regression_model = AgeMultiHeadRegressor(num_labels, age_interval, min_age, max_age)
multihead_regression_model.to(device)
multihead_regression_model.freeze_base_cnn(True)

criterion = nn.MSELoss().to(device)
optimizer = RangerLars(multihead_regression_model.parameters(), lr=1e-2)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

### Train ###

writer = SummaryWriter(
	'logs/AFAD/multihead_regression/RangerLars_unfreeze_at_15_lr_1e2_steplr_10_01_256_dropout')

best_classification_model = train_regression_model(
	multihead_regression_model,
	criterion,
	optimizer,
	exp_lr_scheduler,
	data_loaders,
	dataset_sizes,
	device,
	writer,
	num_labels,
	num_epochs=30
)

print('saving best model')

model_path = 'weights/AFAD/multihead_regression/RangerLars_unfreeze_at_15_lr_1e2_steplr_10_01_256_dropout'
if not os.path.exists(model_path):
	os.makedirs(model_path)
FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
torch.save(best_classification_model.state_dict(), FINAL_MODEL_FILE)

print('exiting')
