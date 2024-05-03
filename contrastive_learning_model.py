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
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image URLs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the URLs for the images
        img_url_anchor = self.data_frame.iloc[idx, 1]
        anchor = self.load_image(img_url_anchor)
        if anchor is None:
            return None  # Returning None for the entire tuple if any image fails to load
        img_url_positive = self.data_frame.iloc[idx, 2]
        # Find a different row for the negative sample
        negative_idx = (idx + 1) % len(self.data_frame)
        img_url_negative = self.data_frame.iloc[negative_idx, 1]  # You can randomize this more robustly

        # Load images from URLs
        anchor = self.load_image(img_url_anchor)
        positive = self.load_image(img_url_positive)
        negative = self.load_image(img_url_negative)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def load_image(self, url):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to download image from {url}, status code: {response.status_code}")
            return None
        try:
            img = Image.open(BytesIO(response.content))
            return img
        except IOError as e:
            print(f"Could not open image from {url}: {e}")
            return None



# Define transforms for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to your CSV file
csv_file = 'inditex_df.csv'

# Initialize dataset
dataset = GarmentTripletDataset(csv_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

train(model, dataloader, TripletMarginLoss, optimizer, epochs=10)