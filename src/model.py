import torch
import torch.nn as nn
import torchvision.models as models

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
        
        # Separate heads for each part
        in_features = base_model.head.in_features
        self.body_head = nn.Linear(in_features, embedding_size)
        self.flipper_head = nn.Linear(in_features, embedding_size)
        self.head_head = nn.Linear(in_features, embedding_size)

    def extract_features(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x
    
    def get_part_embeddings(self, parts_arr):
        """Extract embeddings for body, flipper, and head parts."""
        # Split parts
        body = parts_arr[:, 0:3, :, :]
        flipper = parts_arr[:, 3:6, :, :]
        head = parts_arr[:, 6:9, :, :]
        
        # Extract features and get embeddings for each part
        body_emb = self.body_head(self.extract_features(body))
        flipper_emb = self.flipper_head(self.extract_features(flipper))
        head_emb = self.head_head(self.extract_features(head))
        
        return body_emb, flipper_emb, head_emb
        
    def forward(self, parts_arr):
        body_emb, flipper_emb, head_emb = self.get_part_embeddings(parts_arr)
        
        # Simple addition of embeddings
        return body_emb + flipper_emb + head_emb
    

class WeightedPartsSwinBModel(PartsSwinBModel):
    def __init__(self, embedding_size, part_weights=None):
        super().__init__(embedding_size)
        # Scalar weights for each part (body, flipper, head)
        if part_weights is None:
            part_weights = [1.0, 1.0, 1.0]
        self.part_weights = torch.tensor(part_weights, dtype=torch.float32)
    
    def forward(self, parts_arr):
        body_emb, flipper_emb, head_emb = self.get_part_embeddings(parts_arr)
        
        # Apply scalar weights to each part
        weighted_body = body_emb * self.part_weights[0]
        weighted_flipper = flipper_emb * self.part_weights[1]
        weighted_head = head_emb * self.part_weights[2]

        return weighted_body + weighted_flipper + weighted_head


class LearnableWeightedPartsSwinBModel(PartsSwinBModel):
    def __init__(self, embedding_size, num_parts=3, per_dimension=True):
        super().__init__(embedding_size)
        
        # Learnable weights for each part (body, flipper, head)
        if per_dimension:
            # Per-dimension weights: shape [num_parts, embedding_size]
            # Each dimension of each part embedding can have different importance
            self.part_weights = nn.Parameter(torch.ones(num_parts, embedding_size))
        else:
            # Scalar weights: shape [num_parts]
            # One weight per part (simpler but less expressive)
            self.part_weights = nn.Parameter(torch.ones(num_parts))
        
        self.per_dimension = per_dimension
    
    def forward(self, parts_arr):
        body_emb, flipper_emb, head_emb = self.get_part_embeddings(parts_arr)
        
        # Apply learnable weights (works for both per_dimension=True/False)
        # Broadcasting handles both cases automatically
        weighted_body = body_emb * self.part_weights[0]
        weighted_flipper = flipper_emb * self.part_weights[1]
        weighted_head = head_emb * self.part_weights[2]
        
        return weighted_body + weighted_flipper + weighted_head

