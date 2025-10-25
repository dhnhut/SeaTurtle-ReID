import torch
import torch.nn as nn
import torchvision.models as models
# from torch.utils.data import DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# from tqdm.notebook import tqdm

# from src.dataset import SeaTurtleDataset
# from src.arcface import ArcFace

class WeightedPartsSwinBModel(nn.Module):
    def __init__(self, embedding_size, num_parts=3):
        super().__init__()
        # Learnable weights for each part (body, flipper, head)
        self.part_weights = nn.Parameter(torch.ones(num_parts))
        
        # Use Swin-B with standard 3 channels
        base_model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        
        # Use full Swin architecture (no modification needed for input channels)
        self.features = base_model.features
        self.norm = base_model.norm
        self.permute = base_model.permute
        self.avgpool = base_model.avgpool
        self.flatten = base_model.flatten
        
        # Custom head for embedding
        self.head = nn.Linear(base_model.head.in_features, embedding_size)
    
    def forward(self, parts_arr):
        # parts_arr shape: (B, 9, H, W)
        # Split into 3 parts: body (0:3), flipper (3:6), head (6:9)
        body = parts_arr[:, 0:3, :, :]
        flipper = parts_arr[:, 3:6, :, :]
        head = parts_arr[:, 6:9, :, :]
        
        # Apply learnable weights and combine
        # Each part contributes to the final 3-channel input
        x = body * self.part_weights[0] + flipper * self.part_weights[1] + head * self.part_weights[2]
        
        # Forward through Swin-B
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        
        return x