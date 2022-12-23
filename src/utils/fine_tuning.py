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


def fine_tune_model(model, data_loaders, dataset_sizes, criterion, model_optimizer, scheduler, num_epochs):
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
                model_optimizer.zero_grad()

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
                        model_optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_correct_classes += torch.sum(preds_classes == labels['class_id'].data)
                running_correct_superclasses += torch.sum(preds_superclasses == labels['superclass_id'].data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_classes = running_correct_classes.double() / dataset_sizes[phase]
            epoch_acc_superclasses = running_correct_superclasses.double() / dataset_sizes[phase]

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc [Classes]: {epoch_acc_classes:.4f} Acc [Superclasses]: {epoch_acc_superclasses:.4f}')

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
    return model, model_optimizer


def fine_tune_arcface(model, data_loaders, dataset_sizes,
                      class_criterion, class_loss_optimizer,
                      superclass_criterion, superclass_loss_optimizer,
                      model_optimizer, scheduler, num_epochs):
    """
    Reuse code from PyTorch documentation:
     https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_class_loss, best_superclass_loss = 0.0, 0.0
    best_sum_losses = 1000_000

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_class_loss = 0.0
            running_superclass_loss = 0.0

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
                model_optimizer.zero_grad()
                class_loss_optimizer.zero_grad()
                superclass_loss_optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = {'superclass_id': true_superclass_ids,
                              'class_id': true_class_ids}
                    class_loss = class_criterion(outputs['class_id'], labels['class_id'].to(DEVICE))
                    superclass_loss = superclass_criterion(outputs['superclass_id'], labels['superclass_id'].to(DEVICE))

                    # loss = criterion(outputs, true_superclass_ids)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        class_loss.backward(retain_graph=True)
                        superclass_loss.backward()
                        model_optimizer.step()
                        class_loss_optimizer.step()
                        superclass_loss_optimizer.step()

                # statistics
                running_class_loss += class_loss.item() * inputs.size(0)
                running_superclass_loss += superclass_loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_class_loss = running_class_loss / dataset_sizes[phase]
            epoch_superclass_loss = running_superclass_loss / dataset_sizes[phase]
            print(f'\n[{phase}] Class Loss: {epoch_class_loss:.4f} Superclass Loss: {epoch_superclass_loss:.4f}')

            # deep copy the model
            if phase == 'val' and (epoch_class_loss + epoch_superclass_loss) < best_sum_losses:
                best_class_loss = epoch_class_loss
                best_superclass_loss = epoch_superclass_loss
                best_sum_losses = (best_class_loss + best_superclass_loss)
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss [Classes]: {best_class_loss:4f}\nBests val Loss [Superclasses]: {best_superclass_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, model_optimizer, class_loss_optimizer, superclass_loss_optimizer


def fine_tune_siamese(model, data_loaders, dataset_sizes, custom_criterion,
                      model_optimizer, scheduler, num_epochs):
    """
    Reuse code from PyTorch documentation:
     https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_class_loss, best_superclass_loss = 0.0, 0.0
    best_sum_losses = 1000_000

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_class_loss = 0.0
            running_superclass_loss = 0.0

            # Iterate over data.
            for imgs1, imgs2, same_superclasses_labels, same_classes_labels in tqdm(data_loaders[phase]):
                inputs1 = imgs1.to(DEVICE)
                inputs2 = imgs2.to(DEVICE)
                same_superclasses_labels = same_superclasses_labels.to(DEVICE)
                same_classes_labels = same_classes_labels.to(DEVICE)

                # zero the parameter gradients
                model_optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1 = model(inputs1)
                    outputs2 = model(inputs2)

                    class_loss = custom_criterion(outputs1['class_id'], outputs1['class_id'], same_classes_labels)
                    superclass_loss = custom_criterion(outputs1['superclass_id'], outputs1['superclass_id'],
                                                       same_superclasses_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        class_loss.backward(retain_graph=True)
                        superclass_loss.backward()
                        model_optimizer.step()

                # statistics
                running_class_loss += class_loss.item() * inputs1.size(0)
                running_superclass_loss += superclass_loss.item() * inputs1.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_class_loss = running_class_loss / dataset_sizes[phase]
            epoch_superclass_loss = running_superclass_loss / dataset_sizes[phase]
            print(f'\n[{phase}] Class Loss: {epoch_class_loss:.4f} Superclass Loss: {epoch_superclass_loss:.4f}')

            # deep copy the model
            if phase == 'val' and (epoch_class_loss + epoch_superclass_loss) < best_sum_losses:
                best_class_loss = epoch_class_loss
                best_superclass_loss = epoch_superclass_loss
                best_sum_losses = (best_class_loss + best_superclass_loss)
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss [Classes]: {best_class_loss:4f}\nBests val Loss [Superclasses]: {best_superclass_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, model_optimizer


def complex_triplet_criterion(criterion_func, outputs, labels, mining_func):
    loss = 0
    for i, key in enumerate(outputs.keys()):
        indices_tuple = mining_func(outputs[key], labels[key])
        loss += criterion_func(outputs[key], labels[key].to(DEVICE), indices_tuple)
    return loss


def fine_tune_triplet(model, data_loaders, dataset_sizes, criterion, model_optimizer, scheduler,
                      mining_func, num_epochs):
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
                model_optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds_classes = torch.max(outputs['class_id'], 1)
                    _, preds_superclasses = torch.max(outputs['superclass_id'], 1)
                    labels = {'superclass_id': true_superclass_ids,
                              'class_id': true_class_ids}
                    loss = complex_triplet_criterion(criterion, outputs, labels, mining_func)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        model_optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_correct_classes += torch.sum(preds_classes == labels['class_id'].data)
                running_correct_superclasses += torch.sum(preds_superclasses == labels['superclass_id'].data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_classes = running_correct_classes.double() / dataset_sizes[phase]
            epoch_acc_superclasses = running_correct_superclasses.double() / dataset_sizes[phase]

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc [Classes]: {epoch_acc_classes:.4f} Acc [Superclasses]: {epoch_acc_superclasses:.4f}')

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
    return model, model_optimizer
