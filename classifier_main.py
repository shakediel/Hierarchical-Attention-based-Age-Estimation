import os

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from Datasets.AFAD.AFADClassifierDataset import AFADClassifierDataset
from Datasets.Morph2.Morph2_coral_Dataset import Morph2_coral_Dataset
from Models.AgeClassifier import AgeClassifier
from Training.train_classification_model import train_classification_model
from Optimizers.RangerLars import RangerLars

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

# classes = [(0, 10), (5, 15), (10, 20), (15, 25), (20, 30), (25, 35), (30, 40), (35, 45), (40, 50),
#            (45, 55), (50, 60), (55, 65), (60, 70), (65, 75), (70, 80)]
age_interval = 5
min_age = 15
max_age = 80
BATCH_SIZE = 128
NUM_EPOCHS = 70

NumLabels = int(max_age / age_interval - min_age / age_interval + 1)

# Load data
# data_parser = DataParser()
# data_parser.initialize_data()
#
# train_ds = Morph2ClassifierDataset(
# 	data_parser.x_train,
# 	data_parser.y_train,
# 	MinAge,
# 	AgeIntareval,
# 	transform=transforms.Compose([
# 		# RandomCrop((160, 160), 10),
# 		# transforms.Normalize((103.939, 116.779, 123.68), (1, 1, 1)),
# 		transforms.RandomResizedCrop(160, (0.95, 1.0)),
# 		transforms.RandomHorizontalFlip(),
# 		transforms.ToTensor()
# 	])
# )
#
# test_ds = Morph2ClassifierDataset(
# 	data_parser.x_test,
# 	data_parser.y_test,
# 	MinAge,
# 	AgeIntareval,
# 	transform=transforms.Compose([
# 		# transforms.Normalize((103.939, 116.779, 123.68), (1, 1, 1)),
# 		transforms.ToTensor()
# 	])
# )


# train_ds = AFADClassifierDataset(
# 	'./Datasets/AFAD/aligned_data/afad_train.h5',
# 	min_age=min_age,
# 	max_age=max_age,
# 	age_interval=age_interval,
# 	transform=transforms.Compose([
# 		# transforms.Normalize((103.939, 116.779, 123.68), (1, 1, 1)),
# 		transforms.RandomResizedCrop(160, (0.9, 1.0)),
# 		transforms.RandomHorizontalFlip(),
# 		transforms.RandomRotation(degrees=10, resample=Image.BICUBIC),
# 		transforms.ToTensor()
# 	])
# )
# test_ds = AFADClassifierDataset(
# 	'./Datasets/AFAD/aligned_data/afad_test.h5',
# 	min_age=min_age,
# 	max_age=max_age,
# 	age_interval=age_interval,
# 	transform=transforms.Compose([
# 		transforms.ToTensor()
# 	])
# )

train_ds = Morph2_coral_Dataset(
	'./Datasets/Morph2/coral/coral_morph2_train.h5',
	min_age,
	age_interval,
	transform=transforms.Compose([
		# transforms.Normalize((103.939, 116.779, 123.68), (1, 1, 1)),
		transforms.RandomResizedCrop(160, (0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(degrees=10, resample=Image.BICUBIC),
		transforms.ToTensor()
	])
)

test_ds = Morph2_coral_Dataset(
	'./Datasets/Morph2/coral/coral_morph2_test.h5',
	min_age,
	age_interval,
	transform=transforms.Compose([
		transforms.ToTensor()
	])
)



image_datasets = {
	'train': train_ds,
	'val': test_ds
}

data_loaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

### create model and parameters ###

classification_model = AgeClassifier(NumLabels)
# classification_model = ArcMarginAgeClassifier(NumLabels, margin_m=0.1)
classification_model.to(device)
classification_model.freeze_base_cnn(True)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = RangerLars(classification_model.parameters(), lr=1e-2)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

### Train ###

writer = SummaryWriter('logs/Morph2_coral/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_256_epochs_70')

best_classification_model = train_classification_model(
	classification_model,
	criterion,
	optimizer,
	scheduler,
	data_loaders,
	dataset_sizes,
	device,
	writer,
	num_epochs=NUM_EPOCHS
)

print('saving best model')

model_path = 'weights/Morph2_coral/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_256_epochs_70'
if not os.path.exists(model_path):
	os.makedirs(model_path)
FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
torch.save(best_classification_model.state_dict(), FINAL_MODEL_FILE)

print('exiting')


# model = AgeClassifier(NumLabels)
# best_classification_model.to(device)
# model.load_state_dict(torch.load(PATH))
# model.eval()
