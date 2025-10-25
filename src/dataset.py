import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import kagglehub
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class SeaTurtleDataset(Dataset):
    def __init__(
            self, metadata_path, annotations_path, img_dir,
            coco=None, transform=None
        ):
        self.metadata = pd.read_csv(metadata_path)
        self.img_dir = img_dir
        self.transform = transform
        self.coco = COCO(annotations_path)

        self.map_identity()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.metadata.iloc[idx]["file_name"]
        )
        
        image = Image.open(img_path)
        image_id = self.metadata.iloc[idx]["id"]
        label = self.metadata.iloc[idx]["label"]
        identity = self.metadata.iloc[idx]["identity"]
        
        mask = None
        if self.coco is not None:
            mask = self.get_image_mask(image_id, self.coco)  # Shape: (H, W)
            # if self.transform is not None:
            #     mask = self.transform(mask).long()

        

        # segment body, head, flippers by coco mask
        body_arr = image * (mask == 1)[:, :, None]
        flipper_arr = image * (mask == 2)[:, :, None]
        head_arr = image * (mask == 3)[:, :, None]

        if self.transform:
            image = self.transform(image)
            body_arr = self.transform(body_arr)
            head_arr = self.transform(head_arr)
            flipper_arr = self.transform(flipper_arr)

        # return image, label, identity, mask_tensor
        return {
            "image_id": image_id,
            "label": label,
            "identity": identity,
            "image": image,
            "body_arr": body_arr,
            "head_arr": head_arr,
            "flipper_arr": flipper_arr,
            "mask": mask,
        }
    
    def map_identity(self):
        self.labels = list(self.metadata["identity"].unique())

        self.labels_map = {
            identity: self.labels.index(identity)
            for identity in self.labels
        }

        self.metadata["label"] = self.metadata["identity"].map(
            self.labels_map
        ).astype(int)
        
    def get_image_mask(self, image_id, coco):
        cat_ids = coco.getCatIds()
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        # Initialize mask with image height/width
        img_info = coco.imgs[image_id]
        mask = np.zeros((img_info['height'], img_info['width']), dtype=int)
        for ann in anns:
            submask = coco.annToMask(ann)  # Get single annotation mask
            mask = np.maximum(mask, submask * ann['category_id'])  # Merge masks
        return mask

def metadata_path(dataset_dir):
    return os.path.join(dataset_dir, "turtles-data/data/metadata.csv")

def metadata_splits_path(dataset_dir):
    return os.path.join(dataset_dir, "turtles-data/data/metadata_splits.csv")

def annotations_path(dataset_dir):
    return os.path.join(dataset_dir, "turtles-data/data/annotations.json")

def images_path(dataset_dir):
    return os.path.join(dataset_dir, "turtles-data/data")

def download_dataset():
    path = kagglehub.dataset_download('wildlifedatasets/seaturtleid2022')

    print("Dataset downloaded and extracted to:", path)
    
    return {
        'path': path,
        'images_path': images_path(path),
        'annotations_path': annotations_path(path),
        'metadata': metadata_path(path),
        'metadata_splits': metadata_splits_path(path)
    }


def get_subset_data(path, 
                    n_individuals, min_encounters, out_dir, seed=42):

    df = pd.read_csv(metadata_splits_path(path))
    
    df_identity = df.groupby(['identity'])['date']\
        .nunique().sort_values(ascending=False)
    identity_ids = df_identity[df_identity >= min_encounters]\
        .sample(n=n_individuals, random_state=seed).index.tolist()
    df = df[df['identity'].isin(identity_ids)]
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{out_dir}/metadata.csv', index=False)

    _create_annotations_file(df, annotations_path(path), out_dir)

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

def _create_annotations_file(df, annotations_path, out_dir):
    # keys: ['licenses', 'info', 'categories', 'images', 'annotations']
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    identity_ids = df['identity'].unique().tolist()
    filtered_images = [
        img for img in annotations['images'] if img['identity'] in identity_ids
    ]

    filtered_annotations = [
        ann for ann in annotations['annotations'] 
            if ann['image_id'] in df['id'].values
    ]

    annotations['images'] = filtered_images
    
    annotations['annotations'] = filtered_annotations

    output_path = os.path.join(out_dir, "annotations.json")
    with open(output_path, "w") as f:
        json.dump(annotations, f)
