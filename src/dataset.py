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
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]["file_name"])
        # image = decode_image(img_path)
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx]["identity"]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
