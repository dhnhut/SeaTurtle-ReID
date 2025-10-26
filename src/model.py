import torch
import torch.nn as nn
import torchvision.models as models
# from torch.utils.data import DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# from tqdm.notebook import tqdm

# from src.dataset import SeaTurtleDataset
# from src.arcface import ArcFace

class PartsSwinBModel(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        
        # Use Swin-B with standard 3 channels
        base_model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        
        # Use full Swin architecture (no modification needed for input channels)
        self.features = base_model.features
        self.norm = base_model.norm
        self.permute = base_model.permute
        self.avgpool = base_model.avgpool
        self.flatten = base_model.flatten
        
        # # Custom head for embedding
        # self.head = nn.Linear(base_model.head.in_features, embedding_size)
        # Separate heads for each part
        in_features = base_model.head.in_features
        self.body_head = nn.Linear(in_features, embedding_size)
        self.flipper_head = nn.Linear(in_features, embedding_size)
        self.head_head = nn.Linear(in_features, embedding_size)

    def combine_parts(self, body_emb, flipper_emb, head_emb):
        return body_emb + flipper_emb + head_emb

    def extract_features(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x
        
    def forward(self, parts_arr):
        # Split parts
        body = parts_arr[:, 0:3, :, :]
        flipper = parts_arr[:, 3:6, :, :]
        head = parts_arr[:, 6:9, :, :]
        
        # Extract features and get embeddings for each part
        body_emb = self.body_head(self.extract_features(body))
        flipper_emb = self.flipper_head(self.extract_features(flipper))
        head_emb = self.head_head(self.extract_features(head))
        
        # Weight each embedding dimension separately
        weighted_body = body_emb * self.part_weights[0]
        weighted_flipper = flipper_emb * self.part_weights[1]
        weighted_head = head_emb * self.part_weights[2]
        
        # Combine weighted embeddings
        return weighted_body + weighted_flipper + weighted_head
    

class WeightedPartsSwinBModel(PartsSwinBModel):
    def __init__(self, embedding_size, part_weights=[1, 1, 1]):
        super().__init__(embedding_size)
        self.part_weights = part_weights
    
    def combine_parts(self, body_emb, flipper_emb, head_emb):
        weighted_body = body_emb * self.part_weights[0]
        weighted_flipper = flipper_emb * self.part_weights[1]
        weighted_head = head_emb * self.part_weights[2]

        return weighted_body + weighted_flipper + weighted_head


class LearnableWeightedPartsSwinBModel(WeightedPartsSwinBModel):
    def __init__(self, embedding_size, num_parts=3):
        # Learnable weights for each part (body, flipper, head)
        self.part_weights = nn.Parameter(torch.ones(num_parts))

        super().__init__(embedding_size, part_weights=self.part_weights)
