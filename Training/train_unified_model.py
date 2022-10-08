import copy
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train_unified_model(
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
		NumLabels=1,
		num_epochs=25):

	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_mae = 100.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# if epoch == 15:
		# 	model.FreezeBaseCnn(False)

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_mae = 0.0
			running_corrects = 0.0
			running_p1_error = 0.0
			running_p2_and_above_error = 0.0

			for i, batch in enumerate(tqdm(data_loaders[phase])):
				inputs = batch['image'].to(device)
				classification_labels = batch['classification_label'].to(device).float()
				ages = batch['age'].to(device).float()

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					classification_logits, age_pred = model(inputs)
					_, class_preds = torch.max(classification_logits, 1)

					reg_loss = criterion_reg(age_pred, ages)
					cls_loss = criterion_cls(classification_logits, classification_labels.long())
					mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
					loss = reg_loss + cls_loss + mean_loss + var_loss

					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

				classification_offset = torch.abs(class_preds - classification_labels.data)
				running_corrects += torch.sum(classification_offset == 0)
				running_p1_error += torch.sum(classification_offset == 1)
				running_p2_and_above_error += torch.sum(classification_offset >= 2)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_mae = running_mae / dataset_sizes[phase]

			if phase == 'train':
				scheduler.step(epoch_mae)

			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			epoch_p1_error = running_p1_error.double() / dataset_sizes[phase]
			epoch_p2_and_above_error = running_p2_and_above_error.double() / dataset_sizes[phase]

			writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
			writer.add_scalar('Mae/{}'.format(phase), epoch_mae, epoch)

			writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)
			writer.add_scalar('Accuracy_+-1/{}'.format(phase), epoch_p1_error, epoch)
			writer.add_scalar('Accuracy_+-2_and_above/{}'.format(phase), epoch_p2_and_above_error, epoch)

			print('{} Loss: {:.4f} mae: {:.4f}'.format(phase, epoch_loss, epoch_mae))

			# if phase == 'val':
			# 	writer.add_scalar('alpha', model.alpha.cpu().detach().numpy().squeeze(), epoch)

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






