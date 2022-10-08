import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Datasets.Morph2.DataParser import DataParser
from Datasets.Morph2.Morph2ClassifierDataset import Morph2ClassifierDataset
from Models.JoinedTransformerModel import JoinedTransformerModel
from Models.UnifiedClassificaionAndRegressionAgeModel import UnifiedClassificaionAndRegressionAgeModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

min_age = 15
max_age = 80
age_interval = 10
batch_size = 1
num_epochs = 60
random_split = False

num_classes = int((max_age - min_age) / age_interval + 1)

# Load data
data_parser = DataParser('./Datasets/Morph2/aligned_data/aligned_dataset_with_metadata_uint8.hdf5')
data_parser.initialize_data()

x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test,
if random_split:
	all_images = np.concatenate((x_train, x_test), axis=0)
	all_labels = np.concatenate((y_train, y_test), axis=0)

	x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.20, random_state=42)

test_ds = Morph2ClassifierDataset(
	x_test,
	y_test,
	min_age,
	age_interval,
	transforms.Compose([
		transforms.ToTensor(),
	])
)

tta = transforms.Compose([
	transforms.ToPILImage(),
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

test_data_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True)

pretrained_model = UnifiedClassificaionAndRegressionAgeModel(num_classes, age_interval, min_age, max_age)
pretrained_model_path = 'weights/Morph2/unified/RangerLars_lr_5e4_4096_epochs_60_batch_32_mean_var_vgg16_pretrained_recognition_bin_10_more_augs_RandomApply_warmup_cosine_recreate'
pretrained_model_file = os.path.join(pretrained_model_path, "weights.pt")
pretrained_model.load_state_dict(torch.load(pretrained_model_file), strict=False)
pretrained_model.eval()
pretrained_model.to(device)

mae_metric = nn.L1Loss().to(device)

running_mae = 0.0
running_corrects = 0.0
running_p1_error = 0.0
running_p2_and_above_error = 0.0

num_tta = 6
tta_preds = []
labels = []
for i, batch in enumerate(tqdm(test_data_loader)):
	image = batch['image']
	classification_labels = batch['classification_label'].to(device).float()
	ages = batch['age'].to(device).float()

	predicted_ages = []
	for j in range(num_tta):
		# batch size must be 1
		inputs = tta(image.squeeze()).unsqueeze(dim=0).to(device)

		with torch.set_grad_enabled(False):
			classification_logits, age_pred = pretrained_model(inputs)

			predicted_ages.append(age_pred.cpu().detach().numpy())

	tta_preds.append(np.mean(predicted_ages))
	labels.append(ages[0].cpu().detach().numpy())

mae = mean_absolute_error(tta_preds, labels)
print('mae with tta is {}'.format(mae))




















