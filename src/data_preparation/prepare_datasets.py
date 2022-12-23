import torch
import pandas as pd
import albumentations as albu

from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2 as ToTensor

from src.data_preparation.online_products_datasets import OnlineProductsDataset, OnlineProductsSiameseDataset


def get_data_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation
    return {
        'train': albu.Compose([
            albu.Resize(224, 224),
            # albu.RandomResizedCrop(224, 224),
            albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.15, rotate_limit=15, p=0.5),
            albu.HorizontalFlip(),
            albu.OneOf(
                [
                    albu.MedianBlur(),
                    albu.RandomBrightnessContrast(),
                    albu.RandomGamma(),
                ],
                p=0.5
            ),
            albu.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # albu.Normalize(),
            ToTensor(),
        ]),
        'val': albu.Compose([
            albu.Resize(224, 224),
            # albu.Resize(256, 256),
            # albu.CenterCrop(224, 224),
            albu.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # albu.Normalize(),
            ToTensor(),
        ]),
    }


def get_files_metadata_dfs(root_data_dir, val_fraction, seed):
    train_val_files_metadata_df = pd.read_csv(f"{root_data_dir}/Ebay_train.txt", sep=" ")
    train_val_files_metadata_df.set_index('image_id')
    test_files_metadata_df = pd.read_csv(f"{root_data_dir}/Ebay_test.txt", sep=" ")
    test_files_metadata_df.set_index('image_id')

    train_files_metadata_df, val_files_metadata_df = train_test_split(
        train_val_files_metadata_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val_files_metadata_df[['class_id']]
    )
    # train_files_metadata_df = train_files_metadata_df.reset_index(drop=True)
    # val_files_metadata_df = val_files_metadata_df.reset_index(drop=True)

    print('Train shape: ', train_files_metadata_df.shape)
    print('Val shape: ', val_files_metadata_df.shape)
    print('Test shape: ', test_files_metadata_df.shape)

    return {
        'train': train_files_metadata_df,
        'val': val_files_metadata_df,
        'test': test_files_metadata_df,
    }


def get_datasets(files_metadata_dfs, root_data_dir):
    data_transforms = get_data_transforms()

    train_dataset = OnlineProductsDataset(files_metadata_dfs['train'], root_data_dir, data_transforms['train'])
    val_dataset = OnlineProductsDataset(files_metadata_dfs['val'], root_data_dir, data_transforms['val'])
    test_dataset = OnlineProductsDataset(files_metadata_dfs['test'], root_data_dir, data_transforms['val'])

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    }


def create_siamese_datasets(datasets):
    siamese_datasets = dict()
    for phase in datasets.keys():
        phase_dataset = datasets[phase]
        siamese_datasets[phase] = OnlineProductsSiameseDataset(phase_dataset.files_metadata_df,
                                                               phase_dataset.root_data_dir, phase_dataset.transform)

    return siamese_datasets


def get_data_loaders(datasets, batch_size, num_workers):
    data_loaders = dict()
    for phase in ['train', 'val', 'test']:
        shuffle = True if phase == 'train' else False
        data_loaders[phase] = torch.utils.data.DataLoader(
            datasets[phase],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return data_loaders
