import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class SeaTurtleDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Create a mapping from string labels to integer indices
        self.label_to_int = {
            label: i for i, label in enumerate(self.annotations["id"].unique())
        }
        self.annotations["id"] = self.annotations["id"].map(self.label_to_int)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, label
