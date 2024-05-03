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

loss_fn = TripletMarginLoss(margin=1.0)
optimizer = Adam(model.parameters(), lr=0.001)

# train the model
def train(model, dataloader, loss_fn, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            # Split data into anchor, positive, negative
            # This part depends on how your data is fed into the network
            # You might need to customize your dataset class to return triplets
            anchors, positives, negatives = data

            # Move tensors to the appropriate device
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            # Forward pass
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)

            # Compute loss
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

