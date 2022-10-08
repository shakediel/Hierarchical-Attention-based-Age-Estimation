import copy
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train_regression_model(model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, device, writer,  NumLabels=1, num_epochs=25):
	since = time.time()

	# images = next(iter(data_loaders['train']))['image'].to(device)
	# writer.add_graph(model, (images, torch.ones(NumLabels)))

	best_model_wts = copy.deepcopy(model.state_dict())
	best_mae = 100.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		if epoch == 15:
			model.freeze_base_cnn(False)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()  # Set model to evaluate mode

			running_loss = 0.0
			running_mae = 0.0

			# Iterate over data.
			for i, batch in enumerate(tqdm(data_loaders[phase])):
				inputs = batch['image'].to(device)
				labels = batch['label'].to(device).float()
				weights = batch['weights'].to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs, weights)
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
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Mae: {:4f}'.format(best_mae))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model


def visualize_model(model, data_loaders, class_names, device, num_images=6):
	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure()

	with torch.no_grad():
		for i, (inputs, labels) in enumerate(data_loaders['val']):
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			for j in range(inputs.size()[0]):
				images_so_far += 1
				ax = plt.subplot(num_images // 2, 2, images_so_far)
				ax.axis('off')
				ax.set_title('predicted: {}'.format(class_names[preds[j]]))
				tensor_imshow(inputs.cpu().data[j])

				if images_so_far == num_images:
					model.train(mode=was_training)
					return
		model.train(mode=was_training)


def tensor_imshow(inp, title=None):
	"""Imshow for Tensor."""
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # pause a bit so that plots are updated
