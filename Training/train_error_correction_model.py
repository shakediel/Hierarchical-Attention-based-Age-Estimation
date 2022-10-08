import copy
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from tqdm import tqdm


def train_error_correction_model(
		regression_model, trained_classification_model, criterion, optimizer, scheduler,
		data_loaders, dataset_sizes, device, writer, num_classes=1, num_epochs=25):

	since = time.time()

	# images = next(iter(data_loaders['train']))['image'].to(device)
	# writer.add_graph(model, (images, torch.ones(NumLabels)))

	best_model_wts = copy.deepcopy(regression_model.state_dict())
	best_mae = 100.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		if epoch == 15:
			regression_model.freeze_base_cnn(False)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				regression_model.train()  # Set model to training mode
			else:
				regression_model.eval()  # Set model to evaluate mode

			running_loss = 0.0
			running_mae = 0.0

			# Iterate over data.
			for i, batch in enumerate(tqdm(data_loaders[phase])):
				inputs = batch['image'].to(device)
				labels = batch['label'].to(device).float()

				logits = trained_classification_model(inputs)
				probas = nn.Softmax()(logits)

				class_range = torch.arange(0, num_classes, dtype=torch.float32).to(device)
				batch_mean = torch.squeeze((probas * class_range).sum(1, keepdim=True), dim=1).round()
				weights = np.zeros((probas.shape[0], probas.shape[1]), dtype=np.float32)
				batch_mean = batch_mean.cpu().int().numpy()
				weights[np.arange(len(batch_mean)), batch_mean] = 1.0

				for j, sample_mean in enumerate(batch_mean):
					if sample_mean > 0:
						weights[j, sample_mean - 1] = 0.5
					if sample_mean < num_classes - 1:
						weights[j, sample_mean + 1] = 0.5

				weights = torch.tensor(weights).to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					# outputs = regression_model(inputs, probas)
					outputs = regression_model(inputs, weights)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_mae += torch.nn.L1Loss()(outputs, labels) * inputs.size(0)

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_mae = running_mae / dataset_sizes[phase]

			writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
			writer.add_scalar('Mae/{}'.format(phase), epoch_mae, epoch)

			print('{} Loss: {:.4f} mae: {:.4f}'.format(phase, epoch_loss, epoch_mae))

			# deep copy the model
			if phase == 'val' and epoch_mae < best_mae:
				best_mae = epoch_mae
				best_model_wts = copy.deepcopy(regression_model.state_dict())
				print('best val mae so far...')

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Mae: {:4f}'.format(best_mae))

	# load best model weights
	regression_model.load_state_dict(best_model_wts)
	return regression_model

