import os

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from Datasets.AFAD.AFADRegressorDataset import AFADRegressorDataset
from Datasets.Morph2.DataParser import DataParser
from Datasets.Morph2.Morph2RegressorDataset import Morph2RegressorDataset
from Models.AgeClassifier import AgeClassifier
from Optimizers.RangerLars import RangerLars
from Training.train_error_correction_model import train_error_correction_model
from Models.AgeMultiHeadRegressor import AgeMultiHeadRegressor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

age_interval = 5
min_age = 15
max_age = 80
BATCH_SIZE = 64

num_classes = int(max_age / age_interval - min_age / age_interval + 1)

# Load data

# train_ds = AFADRegressorDataset(
# 	'./Datasets/AFAD/aligned_data/afad_train.h5',
# 	min_age=min_age,
# 	max_age=max_age,
# 	age_interval=age_interval,
# 	transform=transforms.Compose([
# 		# transforms.Normalize((103.939, 116.779, 123.68), (1, 1, 1)),
# 		transforms.RandomResizedCrop(160, (0.9, 1.0)),
# 		transforms.RandomHorizontalFlip(),
# 		transforms.ColorJitter(
# 			brightness=0.1,
# 			contrast=0.1,
# 			saturation=0.1,
# 			hue=0.1
# 		),
# 		# transforms.RandomRotation(degrees=10, resample=Image.BICUBIC),
# 		transforms.RandomAffine(
# 			degrees=10,
# 			translate=(0.1, 0.1),
# 			scale=(0.9, 1.1),
# 			shear=5,
# 			resample=Image.BICUBIC
# 		),
# 		transforms.ToTensor()
# 	])
# )
#
# test_ds = AFADRegressorDataset(
# 	'./Datasets/AFAD/aligned_data/afad_test.h5',
# 	min_age=min_age,
# 	max_age=max_age,
# 	age_interval=age_interval,
# 	transform=transforms.Compose([
# 		transforms.ToTensor()
# 	])
# )

data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
data_parser.initialize_data()

train_ds = Morph2RegressorDataset(
	data_parser.x_train,
	data_parser.y_train,
	min_age,
	age_interval,
	num_classes,
	transforms.Compose([
		transforms.RandomResizedCrop(224, (0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(
			brightness=0.2,
			contrast=0.2,
			saturation=0.2,
			hue=0.2
		),
		transforms.RandomAffine(
			degrees=15,
			translate=(0.15, 0.15),
			scale=(0.85, 1.15),
			shear=15,
			resample=Image.BICUBIC
		),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		transforms.RandomErasing(p=0.5, scale=(0.02, 0.25))
	])
)

test_ds = Morph2RegressorDataset(
	data_parser.x_test,
	data_parser.y_test,
	min_age,
	age_interval,
	num_classes,
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
	x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Create model and parameters

classification_model_path = \
	'weights/Morph2/mean_variance_loss/RangerLars_lr_1e3_weight_decay_1e5_augs/weights.pt'
# 'weights/AFAD/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_256/weights.pt'
# trained_classification_model = AgeClassifier(num_labels)
trained_classification_model = models.resnet34()
trained_classification_model.fc = torch.nn.Linear(trained_classification_model.fc.in_features, num_classes)
trained_classification_model.to(device)
trained_classification_model.load_state_dict(torch.load(classification_model_path))
for param in trained_classification_model.parameters():
	param.requires_grad = False
trained_classification_model.eval()

multihead_regression_model = AgeMultiHeadRegressor(num_classes, age_interval, min_age, max_age)
multihead_regression_model.to(device)
multihead_regression_model.freeze_base_cnn(True)

criterion = nn.MSELoss().to(device)
optimizer = RangerLars(multihead_regression_model.parameters(), lr=5e-3, weight_decay=1e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train

writer = SummaryWriter(
	'logs/Morph2/error_correction/mean_variance/RangerLars_unfreeze_at_15_lr_5e3_weight_decay_1e5_steplr_10_01_256_dropout_even_more_augs_mean_extended_batchsize_64')

best_classification_model = train_error_correction_model(
	multihead_regression_model,
	trained_classification_model,
	criterion,
	optimizer,
	exp_lr_scheduler,
	data_loaders,
	dataset_sizes,
	device,
	writer,
	num_classes,
	num_epochs=50
)

print('saving best model')

model_path = \
	'weights/Morph2/error_correction/mean_variance/RangerLars_unfreeze_at_15_lr_5e3_weight_decay_1e5_steplr_10_01_256_dropout_even_more_augs_mean_extended_batchsize_64'
if not os.path.exists(model_path):
	os.makedirs(model_path)
FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
torch.save(best_classification_model.state_dict(), FINAL_MODEL_FILE)

print('exiting')
