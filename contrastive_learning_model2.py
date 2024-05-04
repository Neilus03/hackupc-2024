'''
This file contains the implementation of the contrastive learning model.
'''

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.nn import TripletMarginLoss
from torch.optim import Adam
import os
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

class GarmentTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filter to ensure we are only adding directories that contain images
        self.garments = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.garments = [g for g in self.garments if len(os.listdir(os.path.join(root_dir, g))) >= 3]

    def __len__(self):
        return len(self.garments)

    def __getitem__(self, idx):
        garment_folder = os.path.join(self.root_dir, self.garments[idx])
        views = [f for f in os.listdir(garment_folder) if not os.path.isdir(os.path.join(garment_folder, f))]
        
        print(f"Garment folder: {garment_folder}")  # Debug print
        print(f"Available views: {views}")  # Debug print

        if len(views) < 3:
            raise Exception("Not enough images to form a triplet")

        anchor_idx, positive_idx = random.sample(range(len(views)), 2)
        anchor_path = os.path.join(garment_folder, views[anchor_idx])
        positive_path = os.path.join(garment_folder, views[positive_idx])

        print(f"Anchor path: {anchor_path}")  # Debug print
        print(f"Positive path: {positive_path}")  # Debug print

        anchor = Image.open(anchor_path)
        positive = Image.open(positive_path)

        # For the negative sample, ensure a different garment is chosen
        negative_garment_idx = (idx + random.randint(1, len(self.garments) - 1)) % len(self.garments)
        negative_garment_folder = os.path.join(self.root_dir, self.garments[negative_garment_idx])
        negative_views = [f for f in os.listdir(negative_garment_folder) if not os.path.isdir(os.path.join(negative_garment_folder, f))]
        negative = Image.open(os.path.join(negative_garment_folder, negative_views[random.randint(0, len(negative_views) - 1)]))

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset and dataloader
root_dir = 'C:/Users/neild/OneDrive/Escritorio/hackupc-2024/images'  # Specify the directory where images are stored
dataset = GarmentTripletDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model setup and training (not shown here for brevity)


print(dataloader)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Create the embeddingmodel class to extract the embeddings from the model
class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        # Load a pre-trained ResNet and remove the last fully connected layer
        base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        # Add a new fully connected layer for embeddings
        self.fc = nn.Linear(base_model.fc.in_features, 256)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = EmbeddingModel().to(device)

TripletMarginLoss = TripletMarginLoss(margin=1.0)
optimizer = Adam(model.parameters(), lr=0.001)

# train the model
def train(model, dataloader, TripletMarginLoss, optimizer, epochs=10):
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        for data in dataloader:
            if None in data:
                continue # Skip batch if any of the data is None
            
            # Unpack the data and get the anchor, positive, and negative samples
            anchors, positives, negatives = data
            
            # Check if any of the data is None (failed load)
            if any(x is None for x in [*anchors, *positives, *negatives]):
                print("Skipping batch with None values")
                continue
            
            # Move tensors to the appropriate device
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            # Forward pass
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)

            # Compute loss
            loss = TripletMarginLoss(anchor_embeddings, positive_embeddings, negative_embeddings)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'best_model.pth')

train(model, dataloader, TripletMarginLoss, optimizer, epochs=10)