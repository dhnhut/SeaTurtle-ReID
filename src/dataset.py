import os
import re
from pathlib import Path
import pandas as pd
from PIL import Image
import kagglehub
import torch
from torch.utils.data import Dataset


class SeaTurtleDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.map_identity()

    def __len__(self):
        return len(self.img_annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.img_annotations.iloc[idx]["file_name"]
        )
        # text_label = self.img_annotations.iloc[idx]["identity"]
        # image = decode_image(img_path)
        image = Image.open(img_path).convert("RGB")
        label = self.img_annotations.iloc[idx]["label"]
        identity = self.img_annotations.iloc[idx]["identity"]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, identity
    
    def map_identity(self):
        self.labels = list(self.img_annotations["identity"].unique())
        
        # prog = re.compile(r'(\d+)')
        # self.labels_map = {
        #     identity: int(prog.search(identity).group(1))
        #     for identity in self.labels
        # }
        
        # self.img_annotations["label"] = self.img_annotations["identity"].map(
        #     self.labels_map
        # ).astype(int)
        
        self.labels_map = {
            identity: self.labels.index(identity)
            for identity in self.labels
        }

        self.img_annotations["label"] = self.img_annotations["identity"].map(
            self.labels_map
        ).astype(int)

def download_dataset():
    path = kagglehub.dataset_download('wildlifedatasets/seaturtleid2022')

    print("Dataset downloaded and extracted to:", path)
    
    return {
        'path': path,
        'images_path': f'{path}/turtles-data/data',
        'annotations_path': f'{path}/turtles-data/data/annotations.json',
        'metadata': f'{path}/turtles-data/data/metadata.csv',
        'metadata_splits': f'{path}/turtles-data/data/metadata_splits.csv'
    }


def get_subset_data(metadata_splits_path
                    , n_individuals, min_encounters, out_dir, seed=42):
    df = pd.read_csv(metadata_splits_path)

    df_identity = df.groupby(['identity'])['date']\
        .nunique().sort_values(ascending=False)

    identity_ids = df_identity[df_identity >= min_encounters]\
        .sample(n=n_individuals, random_state=seed).index.tolist()

    df = df[df['identity'].isin(identity_ids)]
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df.to_csv(f'{out_dir}/metadata.csv', index=False)
    
    

    return df

def closed_set_spliting(df, out_dir, train_encounters=2
                        , valid_encounters=1, test_encounters=1):

    df = df.groupby('identity')\
        .apply(lambda g: _select_encounters_per_identity(
            g, train_encounters=train_encounters,
            valid_encounters=valid_encounters, test_encounters=test_encounters))\
        .reset_index(drop=True)\
        .drop(columns=['split_closed_random', 'split_open', 'year'])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{out_dir}/metadata_closed_set_split.csv', index=False)
    
    for set_name in ['train', 'valid', 'test']:
        df[df['split_closed'] == set_name].to_csv(
            f'{out_dir}/metadata_closed_set_splits_{set_name}.csv', index=False)
    
    return df

# Alternative approach with more explicit date selection
def _select_encounters_per_identity(group, train_encounters=2, 
                                    valid_encounters=1, test_encounters=1):
    result_frames = []

    result_frames.append(_get_sets(group, 'train', train_encounters))
    result_frames.append(_get_sets(group, 'valid', valid_encounters))
    result_frames.append(_get_sets(group, 'test', test_encounters))

    return pd.concat(result_frames, ignore_index=True)

def _get_sets(group, set, n_encounters):
    set_group = group[group['split_closed'] == set]
    if not set_group.empty:
        set_dates = set_group['date'].unique()[:n_encounters]
        set_images = set_group[set_group['date'].isin(set_dates)]
        return set_images
