import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.notebook import tqdm

from src.dataset import SeaTurtleDataset
from src.arcface import ArcFace
from src.model import WeightedPartsSwinBModel

class TurtleReIdPipeline():
    def __init__(self, device, embedding_size, use_weighted_parts=False):
        self.device = device
        self.embedding_size = embedding_size
        self.use_weighted_parts = use_weighted_parts

        if use_weighted_parts:
            model = WeightedPartsSwinBModel(embedding_size=self.embedding_size)
        else:
            model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            model.head = nn.Linear(model.head.in_features, self.embedding_size)
        
        self.model = model.to(device)

    def set_transforms(self, img_size):
        self.train_transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.4),
            A.RandomBrightnessContrast(p=0.15),
            A.Rotate(limit=20),
            # swimb_normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()  # Convert to PyTorch tensor (C, H, W)
        ])
        self.test_transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            # swimb_normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()  # Convert to PyTorch tensor (C, H, W)
        ])

    def set_datasets(self, datasets_paths, img_dir, annotations_path):
        train_csv_path = datasets_paths['train_csv_path']
        eval_csv_path = datasets_paths['eval_csv_path']
        test_csv_path = datasets_paths['test_csv_path']

        self.train_dataset = SeaTurtleDataset(
            metadata_path=train_csv_path, img_dir=img_dir, annotations_path=annotations_path, transform=self.train_transform)
        self.eval_dataset = SeaTurtleDataset(
            metadata_path=eval_csv_path, img_dir=img_dir, annotations_path=annotations_path, transform=self.test_transform)
        self.test_dataset = SeaTurtleDataset(
            metadata_path=test_csv_path, img_dir=img_dir, annotations_path=annotations_path, transform=self.test_transform)
        
        self.num_classes = len(self.train_dataset.metadata['identity'].unique())

    def train(self, configs):
        target_part = f"{configs['target_part']}_arr"
        batch_size = configs['batch_size']
        learning_rate = configs['learning_rate']
        epochs = configs['epochs']
        model_save_path = configs['model_save_path']

        metric = ArcFace(
            num_classes=self.num_classes, embedding_size=self.embedding_size, scale=30.0, margin=0.50
        ).to(self.device)

        # Use k-NN voting (e.g., k=1 for nearest neighbor)
        k = configs['k']
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(
            list(self.model.parameters()) + list(metric.parameters()),
            lr=learning_rate
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(
            self.eval_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False)
        
        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for i, item in enumerate(progress_bar):
                target_arr, labels = item[target_part].to(self.device), item['label'].to(self.device)

                # Forward pass
                features = self.model(target_arr)
                output = metric(features, labels)
                loss = criterion(output, labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss / (i + 1)})

            scheduler.step()

            epoch_loss = running_loss / len(train_loader)

            # Evaluation
            self.model.eval()
            correct = 0
            total = 0

            # Build gallery from training set
            gallery_embeddings = []
            gallery_labels = []
            with torch.no_grad():
                for item in train_loader:
                    target_arr = item[target_part].to(self.device)
                    labels = item['label']
                    embeddings = self.model(target_arr)
                    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
                    gallery_embeddings.append(embeddings.cpu())
                    gallery_labels.append(labels)

            gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
            gallery_labels = torch.cat(gallery_labels, dim=0)

            # Evaluate on eval set
            with torch.no_grad():
                for item in eval_loader:
                    target_arr = item[target_part].to(self.device)
                    labels = item['label'].to(self.device)
                    embeddings = self.model(target_arr)
                    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Compute cosine similarity with gallery
                    similarities = torch.mm(embeddings,
                                            gallery_embeddings.to(self.device).t())

                    _, top_k_indices = torch.topk(similarities, k, dim=1)
                    top_k_labels = gallery_labels[top_k_indices.cpu()]
                    predicted, _ = torch.mode(top_k_labels, dim=1)
                    predicted = predicted.to(self.device)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    
                    
            epoch_acc = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Eval Accuracy: {epoch_acc:.2f}%")
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.model.state_dict(), model_save_path)
                print("Saved best model.")

        print(f"Finished Training. Best Test Accuracy: {best_acc:.2f}%")