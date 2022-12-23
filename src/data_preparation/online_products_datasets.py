import cv2
import random
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
        image = self.get_image(img_path)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, class_id, superclass_id

    def get_image(self, img_path):
        # Read an image with OpenCV
        image = cv2.imread(f"{self.root_data_dir}/{img_path}")

        # By default, OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def get_random_image_from_ids(self, superclass_id, class_id):
        df = self.files_metadata_df.loc[(self.files_metadata_df["super_class_id"] == superclass_id) &
                                        (self.files_metadata_df["class_id"] == class_id)]
        if len(df) == 0:
            return None
        _, _, _, image_path = df.iloc[random.randint(0, len(df) - 1)].to_list()
        return self.get_image(image_path)


class OnlineProductsSiameseDataset(OnlineProductsDataset):
    def __init__(self, files_metadata_df, root_data_dir, transform=None):
        super().__init__(files_metadata_df, root_data_dir, transform)
        self.unique_pairs = self.files_metadata_df[['super_class_id', 'class_id']].drop_duplicates().values.tolist()

    def get_pair_superclass_and_class(self, true_superclass_id, true_class_id, choices):
        same_superclass, same_class = random.choice(choices)
        if same_superclass:
            pair_superclass = true_superclass_id
        else:
            pair_superclass = random.choice([i for i in self.superclasses if i != true_superclass_id])

        if same_class:
            pair_class = true_class_id
        else:
            pair_class = random.choice([i for i in self.classes if i != true_class_id])

        return same_superclass, same_class, pair_superclass, pair_class

    def __getitem__(self, idx):
        _, true_class_id, true_superclass_id, img_path = self.files_metadata_df.iloc[idx]

        # Read an image with OpenCV
        image1 = self.get_image(img_path)

        same_img_type = random.randint(0, 1)
        if same_img_type:
            pair_superclass, pair_class = true_superclass_id, true_class_id
        else:
            pair_superclass, pair_class = random.choice(
                [[superclass_id, class_id] for superclass_id, class_id in self.unique_pairs \
                 if superclass_id != true_superclass_id and class_id != true_class_id])

        image2 = self.get_random_image_from_ids(pair_superclass, pair_class)
        if self.transform:
            augmented1 = self.transform(image=image1)
            image1 = augmented1['image']
            augmented2 = self.transform(image=image2)
            image2 = augmented2['image']

        return image1, image2, same_img_type, same_img_type
