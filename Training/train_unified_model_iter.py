import copy
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def validate(model, val_data_loader, iter, device, criterion_reg, criterion_cls, mean_var_criterion, dataset_size, writer, use_cls_mean_var):
	print('running on validation set...')

	model.eval()

	running_loss = 0.0
	running_mae = 0.0
	running_corrects = 0.0
	running_p1_error = 0.0
	running_p2_and_above_error = 0.0

	for i, batch in enumerate(val_data_loader):
		inputs = batch['image'].to(device)
		classification_labels = batch['classification_label'].to(device).float()
		ages = batch['age'].to(device).float()

		with torch.no_grad():
			classification_logits, age_pred = model(inputs)
			_, class_preds = torch.max(classification_logits, 1)

		reg_loss = criterion_reg(age_pred, ages)
		loss = reg_loss
		if use_cls_mean_var:
			cls_loss = criterion_cls(classification_logits, classification_labels.long())
			mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
			loss += cls_loss + mean_loss + var_loss

		running_loss += loss.item() * inputs.size(0)
		running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

		if use_cls_mean_var:
			classification_offset = torch.abs(class_preds - classification_labels.data)
			running_corrects += torch.sum(classification_offset == 0)
			running_p1_error += torch.sum(classification_offset == 1)
			running_p2_and_above_error += torch.sum(classification_offset >= 2)

	epoch_loss = running_loss / dataset_size
	epoch_mae = running_mae / dataset_size

	if use_cls_mean_var:
		epoch_acc = running_corrects.double() / dataset_size
		epoch_p1_error = running_p1_error.double() / dataset_size
		epoch_p2_and_above_error = running_p2_and_above_error.double() / dataset_size

	writer.add_scalar('Loss/val', epoch_loss, iter)
	writer.add_scalar('Mae/val', epoch_mae, iter)

	if use_cls_mean_var:
		writer.add_scalar('Accuracy/val', epoch_acc, iter)
		writer.add_scalar('Accuracy_+-1/val', epoch_p1_error, iter)
		writer.add_scalar('Accuracy_+-2_and_above/val', epoch_p2_and_above_error, iter)

	# writer.add_scalar('alpha', model.alpha.cpu().detach().numpy().squeeze(), iter)

	print('{} Loss: {:.4f} mae: {:.4f}'.format('val', epoch_loss, epoch_mae))

	return epoch_mae


def train_unified_model_iter(
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
		model_path,
		NumLabels=1,
		num_epochs=25,
		validate_at_k=100,
		use_cls_mean_var=True
):

	since = time.time()

	scaler = GradScaler()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_mae = 100.0

	running_loss = 0.0
	running_mae = 0.0
	running_corrects = 0.0
	running_p1_error = 0.0
	running_p2_and_above_error = 0.0

	iter = 0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# if epoch == 15:
		# 	model.FreezeBaseCnn(False)

		phase = 'train'
		for batch in tqdm(data_loaders[phase]):
			if iter % validate_at_k == 0 and iter != 0:
				norm = validate_at_k * inputs.size(0)

				epoch_loss = running_loss / norm
				epoch_mae = running_mae / norm

				if use_cls_mean_var:
					epoch_acc = running_corrects.double() / norm
					epoch_p1_error = running_p1_error.double() / norm
					epoch_p2_and_above_error = running_p2_and_above_error.double() / norm

				writer.add_scalar('Loss/{}'.format(phase), epoch_loss, iter)
				writer.add_scalar('Mae/{}'.format(phase), epoch_mae, iter)

				if use_cls_mean_var:
					writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, iter)
					writer.add_scalar('Accuracy_+-1/{}'.format(phase), epoch_p1_error, iter)
					writer.add_scalar('Accuracy_+-2_and_above/{}'.format(phase), epoch_p2_and_above_error, iter)

				print('{} Loss: {:.4f} mae: {:.4f}'.format(phase, epoch_loss, epoch_mae))

				val_mae = validate(model, data_loaders['val'], iter, device, criterion_reg, criterion_cls, mean_var_criterion, dataset_sizes['val'], writer, use_cls_mean_var=use_cls_mean_var)

				# deep copy the model
				if val_mae < best_mae:
					best_mae = val_mae
					best_model_wts = copy.deepcopy(model.state_dict())

					FINAL_MODEL_FILE = os.path.join(model_path, "weights_{}_{:.4f}.pt".format(iter, val_mae))
					torch.save(best_model_wts, FINAL_MODEL_FILE)

				model.train()

				running_loss = 0.0
				running_mae = 0.0
				running_corrects = 0.0
				running_p1_error = 0.0
				running_p2_and_above_error = 0.0

			iter += 1

			inputs = batch['image'].to(device)
			classification_labels = batch['classification_label'].to(device).float()
			ages = batch['age'].to(device).float()

			optimizer.zero_grad()

			with autocast():
				classification_logits, age_pred = model(inputs)
				_, class_preds = torch.max(classification_logits, 1)

				reg_loss = criterion_reg(age_pred, ages)
				loss = reg_loss
				if use_cls_mean_var:
					cls_loss = criterion_cls(classification_logits, classification_labels.long())
					mean_loss, var_loss = mean_var_criterion(classification_logits, classification_labels)
					loss += cls_loss + mean_loss + var_loss

			# loss.backward()
			# optimizer.step()

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			running_loss += loss.item() * inputs.size(0)
			running_mae += torch.nn.L1Loss()(age_pred, ages) * inputs.size(0)

			if use_cls_mean_var:
				classification_offset = torch.abs(class_preds - classification_labels.data)
				running_corrects += torch.sum(classification_offset == 0)
				running_p1_error += torch.sum(classification_offset == 1)
				running_p2_and_above_error += torch.sum(classification_offset >= 2)

			# scheduler.step(epoch_mae)
			scheduler.step()


		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Mae: {:4f}'.format(best_mae))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model






