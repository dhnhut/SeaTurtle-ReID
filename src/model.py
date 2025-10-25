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
        
        # Use Swin-B
        base_model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        
        # 9 channels (3 parts × 3 RGB channels)
        old_conv = base_model.features[0][0]
        self.first_conv = nn.Conv2d(
            9,  # 9 input channels
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        
        # Initialize new conv weights: average pretrained weights across input channels
        with torch.no_grad():
            # Repeat weights 3 times for 9 channels
            self.first_conv.weight = nn.Parameter(
                old_conv.weight.repeat(1, 3, 1, 1) / 3
            )
            self.first_conv.bias = old_conv.bias
        
        # Use rest of Swin architecture
        self.features = nn.Sequential(*list(base_model.features.children())[1:])
        self.norm = base_model.norm
        self.permute = base_model.permute
        self.avgpool = base_model.avgpool
        self.flatten = base_model.flatten
        
        # Custom head for embedding
        self.head = nn.Linear(base_model.head.in_features, embedding_size)
    
    def forward(self, parts_arr):
        # parts_arr shape: (B, 9, H, W)
        # Split into 3 parts: body (0:3), flipper (3:6), head (6:9)
        body = parts_arr[:, 0:3, :, :] * self.part_weights[0]
        flipper = parts_arr[:, 3:6, :, :] * self.part_weights[1]
        head = parts_arr[:, 6:9, :, :] * self.part_weights[2]
        
        # Recombine weighted parts
        x = torch.cat([body, flipper, head], dim=1)
        
        # Forward through Swin-B
        x = self.first_conv(x)
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        
        return x