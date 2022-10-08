import copy
import time

import torch
from tqdm import tqdm


def train_mean_variance_model(
		model,
		mean_var_criterion,
		cce_criterion,
		optimizer,
		scheduler,
		data_loaders,
		dataset_sizes,
		device,
		writer,
		num_epochs=30):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()  # Set model to evaluate mode

			running_loss = 0.0
			running_mean_loss = 0
			running_var_loss = 0
			running_cce_loss = 0
			running_corrects = 0
			running_p1_error = 0
			running_p2_and_above_error = 0

			for i, batch in enumerate(tqdm(data_loaders[phase])):
				inputs = batch['image'].to(device)
				labels = batch['classification_label'].to(device).float()

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					mean_loss, var_loss = mean_var_criterion(outputs, labels)
					cce_loss = cce_criterion(outputs, labels.long())

					loss = cce_loss + mean_loss + var_loss

					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_mean_loss += mean_loss.item() * inputs.size(0)
				running_var_loss += var_loss.item() * inputs.size(0)
				running_cce_loss += cce_loss.item() * inputs.size(0)

				classification_offset = torch.abs(preds - labels.data)

				running_corrects += torch.sum(classification_offset == 0)
				running_p1_error += torch.sum(classification_offset == 1)
				running_p2_and_above_error += torch.sum(classification_offset >= 2)

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_mean_loss = running_mean_loss / dataset_sizes[phase]
			epoch_var_loss = running_var_loss / dataset_sizes[phase]
			epoch_cce_loss = running_cce_loss / dataset_sizes[phase]

			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			epoch_p1_error = running_p1_error.double() / dataset_sizes[phase]
			epoch_p2_and_above_error = running_p2_and_above_error.double() / dataset_sizes[phase]

			writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
			writer.add_scalar('Mean Loss/{}'.format(phase), epoch_mean_loss, epoch)
			writer.add_scalar('Var Loss/{}'.format(phase), epoch_var_loss, epoch)
			writer.add_scalar('Cce loss/{}'.format(phase), epoch_cce_loss, epoch)

			writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)
			writer.add_scalar('Accuracy_+-1/{}'.format(phase), epoch_p1_error, epoch)
			writer.add_scalar('Accuracy_+-2_and_above/{}'.format(phase), epoch_p2_and_above_error, epoch)

			print('{} Loss: {:.4f} mean_loss: {:.4f}'.format(phase, epoch_loss, epoch_mean_loss))

			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Mean: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model









