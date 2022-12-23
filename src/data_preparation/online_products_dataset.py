import cv2
from torch.utils.data import DataLoader, Dataset


class OnlineProductsDataset(Dataset):
    def __init__(self, files_metadata_df, root_data_dir, transform=None):
        self.files_metadata_df = files_metadata_df
        self.transform = transform
        self.root_data_dir = root_data_dir

    def __len__(self):
        return len(self.files_metadata_df)

    def __getitem__(self, idx):
        _, class_id, superclass_id, img_path = self.files_metadata_df.iloc[idx]

        # Read an image with OpenCV
        image = cv2.imread(f"{self.root_data_dir}/{img_path}")

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, class_id, superclass_id
