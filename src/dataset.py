import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.io import decode_image


class SeaTurtleDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # Create a mapping from string labels to integer IDs
        self.labels = self.img_annotations["label"].unique()

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.img_annotations.iloc[idx]["file_name"]
        )
        text_label = self.img_annotations.iloc[idx]["identity"]
        # image = decode_image(img_path)
        image = Image.open(img_path).convert("RGB")
        label = self.img_annotations.iloc[idx]["label"]
        identity = self.img_annotations.iloc[idx]["identity"]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, identity
