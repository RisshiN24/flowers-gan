import os
import scipy.io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn

# =========================
# Dataset Preparation
# =========================

def load_flower_dataset(data_dir="jpg/", label_path="imagelabels.mat", image_size=64, batch_size=64, num_workers=2):
    labels = scipy.io.loadmat(label_path)['labels'].squeeze() # 1D array
    image_paths = [
        os.path.join(data_dir, f"image_{i:05d}.jpg")
        for i in range(1, len(labels) + 1)
    ]
    label_tuples = [(path, int(label - 1)) for path, label in zip(image_paths, labels)]  # zero-index

    # Ensure image size is consistent and pixel values are in [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # [-1, 1]
    ])

    dataset = FlowerDataset(label_tuples, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

class FlowerDataset(Dataset):
    def __init__(self, image_label_list, transform=None):
        self.data = image_label_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# =========================
# Generator
# =========================

class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=102, img_channels=3, feature_map_size=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + label_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(x)

# =========================
# Discriminator
# =========================

class Discriminator(nn.Module):
    def __init__(self, label_dim=102, img_channels=3, feature_map_size=64):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Linear(label_dim, 64 * 64)

        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        batch_size = labels.size(0)
        label_map = self.label_embedding(labels).view(batch_size, 1, 64, 64)
        x = torch.cat([img, label_map], dim=1)
        return self.net(x).view(-1)