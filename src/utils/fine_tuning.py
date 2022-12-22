import time
import copy
import torch
from tqdm import tqdm

from src.configs import DEVICE


def complex_criterion(criterion_func, outputs, labels):
    loss = 0
    for i, key in enumerate(outputs.keys()):
        loss += criterion_func(outputs[key], labels[key].to(DEVICE))
    return loss


def fine_tune_model(model, data_loaders, dataset_sizes,
                    criterion, optimizer, scheduler, num_epochs):
    """
    Reuse code from PyTorch documentation:
     https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_sum_acc = 0.0
    best_acc_classes, best_acc_superclasses = 0.0, 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_correct_classes = 0
            running_correct_superclasses = 0

            # Iterate over data.
            for imgs, true_class_ids, true_superclass_ids in tqdm(data_loaders[phase]):
                inputs = imgs.to(DEVICE)
                true_class_ids = true_class_ids.to(DEVICE)
                true_superclass_ids = true_superclass_ids.to(DEVICE)
                # model returns superclasses which are indexed from zero, not 1,
                # hence we need to subtract 1
                true_class_ids -= 1
                true_superclass_ids -= 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds_classes = torch.max(outputs['class_id'], 1)
                    _, preds_superclasses = torch.max(outputs['superclass_id'], 1)
                    labels = {'superclass_id': true_superclass_ids,
                              'class_id': true_class_ids}
                    loss = complex_criterion(criterion, outputs, labels)
                    # loss = criterion(outputs, true_superclass_ids)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_correct_classes += torch.sum(preds_classes == labels['class_id'].data)
                running_correct_superclasses += torch.sum(preds_superclasses == labels['superclass_id'].data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_classes = running_correct_classes.double() / dataset_sizes[phase]
            epoch_acc_superclasses = running_correct_superclasses.double() / dataset_sizes[phase]

            print(
                f'\n{phase} Loss: {epoch_loss:.4f} Acc [Classes]: {epoch_acc_classes:.4f} Acc [Superclasses]: {epoch_acc_superclasses:.4f}')
            # print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and (epoch_acc_classes + epoch_acc_superclasses) > best_sum_acc:
                best_acc_classes = epoch_acc_classes
                best_acc_superclasses = epoch_acc_superclasses
                best_sum_acc = (best_acc_classes + best_acc_superclasses)
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc [Classes]: {best_acc_classes:4f}\nBests val Acc [Superclasses]: {best_acc_superclasses:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, optimizer
