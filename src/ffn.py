# Inspired by the paper: "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
# Link: https://arxiv.org/abs/2006.10739
#
# This script demonstrates the core idea of the paper by training two neural networks
# to fit an image:
# 1. A standard MLP that takes (x, y) coordinates as input.
# 2. An MLP that takes Fourier features of (x, y) coordinates as input.
#
# You'll observe that the standard MLP produces a blurry, low-frequency representation,
# while the MLP with Fourier features can capture fine details and high-frequency content.

import torch
import torch.nn as nn
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Resize, Compose

# --- 1. Fourier Feature Mapping ---
# This is the core component described in the paper (Eq. 5).
# It maps a low-dimensional input vector to a higher-dimensional space of sinusoids.
class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, scale=10.0):
        """
        Initializes the Fourier feature mapping.
        Args:
            in_features (int): Dimensionality of the input coordinates (e.g., 2 for (x, y)).
            mapping_size (int): The number of Fourier features to generate (m in the paper).
                                The output dimension will be 2 * mapping_size.
            scale (float): The standard deviation of the Gaussian distribution for the frequency vectors (B matrix).
        """
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        
        # B is the random frequency matrix, sampled from N(0, scale^2)
        # It is registered as a buffer so it's part of the model state, but not a trainable parameter.
        B = torch.randn((in_features, mapping_size)) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        """
        Applies the Fourier feature mapping to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (N, in_features), where N is the number of points.
        Returns:
            torch.Tensor: Mapped tensor of shape (N, 2 * mapping_size).
        """
        # Input shape: (batch_size, in_features)
        # B shape: (in_features, mapping_size)
        # x_proj shape: (batch_size, mapping_size)
        x_proj = x @ self.B
        
        # The mapping is [cos(2π * vB), sin(2π * vB)]
        return torch.cat([torch.sin(2 * np.pi * x_proj), torch.cos(2 * np.pi * x_proj)], dim=-1)

# --- 2. Standard MLP Model ---
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        
        layers = [nn.Linear(in_features, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_features, hidden_features), nn.ReLU()])
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)) # Sigmoid to keep outputs in [0, 1] range for colors


def get_image_tensor_and_coords(image, size=256):
    """Converts image to a tensor, and creates coordinate grid."""

    transform = Compose([Resize((size, size)), ToTensor()])
    img_tensor = transform(image)

    # Create a grid of (x, y) coordinates in the range [-1, 1]
    h, w = size, size
    y_coords, x_coords = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)
    
    # Reshape image tensor to match coordinates
    pixels = img_tensor.permute(1, 2, 0).view(-1, 3)
    
    return img_tensor, coords, pixels

def train_model(model, coords, pixels, epochs=2000, lr=1e-4):
    """Generic training loop for a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(coords)
        loss = loss_fn(prediction, pixels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    return model

def main(image, img_size=256, hidden_features=256, hidden_layers=4, mapping_size=256, scale=10.0, epochs=2000, device=None):
    

    # --- Data Preparation ---
    print(f"Loading image from {image}...")
    original_img_tensor, coords, pixels = get_image_tensor_and_coords(image, img_size)
    coords, pixels = coords.to(device), pixels.to(device)
    print(f"Using device: {device}")

    # --- Model 1: Standard MLP (No Fourier Features) ---
    print("\n--- Training Standard MLP ---")
    vanilla_mlp = MLP(in_features=2, hidden_features=hidden_features, hidden_layers=hidden_layers, out_features=3).to(device)
    train_model(vanilla_mlp, coords, pixels, epochs=2000)

    # --- Model 2: MLP with Fourier Features ---
    print("\n--- Training MLP with Fourier Features ---")
    fourier_mapper = FourierFeatures(in_features=2, mapping_size=mapping_size, scale=scale)
    # The MLP input size must be the output size of the fourier mapper
    fourier_mlp = MLP(in_features=mapping_size * 2, hidden_features=hidden_features, hidden_layers=hidden_layers, out_features=3).to(device)

    model_with_fourier_features = nn.Sequential(fourier_mapper, fourier_mlp).to(device)
    train_model(model_with_fourier_features, coords, pixels, epochs=2000)

    # --- Visualization ---
    print("\nGenerating output images...")
    with torch.no_grad():
        vanilla_output = vanilla_mlp(coords).cpu().view(img_size, img_size, 3).permute(2, 0, 1)
        fourier_output = model_with_fourier_features(coords).cpu().view(img_size, img_size, 3).permute(2, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img_tensor.permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(vanilla_output.permute(1, 2, 0))
    axes[1].set_title("Without Fourier Features")
    axes[1].axis('off')

    axes[2].imshow(fourier_output.permute(1, 2, 0))
    axes[2].set_title("With Fourier Features")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

