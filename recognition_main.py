import json
import os

import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Datasets.Morph2.DataParser import DataParser
from Datasets.Morph2.Morph2RecognitionDataset import Morph2RecognitionDataset
from Models.ArcMarginClassifier import ArcMarginClassifier
from Optimizers.RangerLars import RangerLars
from Training.train_recognition_model import train_recognition_model

BATCH_SIZE = 64
NUM_EPOCHS = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
data_parser.initialize_data()

ids = np.unique([json.loads(m)['id_num'] for m in data_parser.y_train])

X_train, X_test, y_train, y_test = train_test_split(data_parser.x_train, data_parser.y_train, test_size=0.33, random_state=42)

train_ds = Morph2RecognitionDataset(
	X_train,
	y_train,
	ids,
	transforms.Compose([
		transforms.RandomResizedCrop(224, (0.9, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(
			brightness=0.125,
			contrast=0.125,
			saturation=0.125,
			hue=0.125
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
		# transforms.RandomErasing(p=0.5, scale=(0.02, 0.25))
	])
)

test_ds = Morph2RecognitionDataset(
	X_test,
	y_test,
	ids,
	transforms.Compose([
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
model = ArcMarginClassifier(len(ids))
model.to(device)
model.freeze_base_cnn(True)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = RangerLars(model.parameters(), lr=1e-2)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

### Train ###

writer = SummaryWriter('logs/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64')

best_classification_model = train_recognition_model(
	model,
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

model_path = 'weights/Morph2_recognition/vgg16/RangerLars_unfreeze_at_15_lr_1e2_steplr_01_batchsize_64'
if not os.path.exists(model_path):
	os.makedirs(model_path)
FINAL_MODEL_FILE = os.path.join(model_path, "weights.pt")
torch.save(best_classification_model.state_dict(), FINAL_MODEL_FILE)

print('exiting')

