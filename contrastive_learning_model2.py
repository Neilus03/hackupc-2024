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
        """
        Args:
            root_dir (string): Directory with all the garment images organized in subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.garments = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.garments)

    def __getitem__(self, idx):
        garment_folder = os.path.join(self.root_dir, self.garments[idx])
        views = os.listdir(garment_folder)  # List the files in the garment's directory

        # Randomly select two different images for the anchor and positive, and one from another garment for the negative
        anchor_idx, positive_idx = random.sample(range(len(views)), 2)
        anchor = Image.open(os.path.join(garment_folder, views[anchor_idx]))
        positive = Image.open(os.path.join(garment_folder, views[positive_idx]))

        # Choose negative from a different garment
        negative_garment_idx = (idx + random.randint(1, len(self.garments) - 1)) % len(self.garments)
        negative_garment_folder = os.path.join(self.root_dir, self.garments[negative_garment_idx])
        negative_views = os.listdir(negative_garment_folder)
        negative = Image.open(os.path.join(negative_garment_folder, negative_views[random.randint(0, len(negative_views) - 1)]))

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

# Initialize dataset
root_dir = 'path_to_garments_directory'  # Update this to your directory path
dataset = GarmentTripletDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

'''
# Path to your CSV file
csv_file = 'inditex_df_no_nan.csv'

# Initialize dataset
dataset = GarmentTripletDataset(csv_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)'''

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