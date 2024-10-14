import os
import math
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim.optimizer import Optimizer
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CelebA

class ImageDataset(Dataset):
    def __init__(self, root_dir, datapair, transform):
        self.root_dir = root_dir
        self.datapair = datapair
        self.transform = transform

    def __getitem__(self, index):
        
        img_name = os.path.join(self.root_dir, self.datapair[index][0], 
                       self.datapair[index][1])
        img = Image.open(img_name).convert('RGB')

        img = self.transform(img)

        return img, self.datapair[index][2]

    def __len__(self):
        return len(self.datapair)
    
class UserDataset(Dataset):
    def __init__(self, images, labels, type_="mnist"):
        """Construct a user train_dataset and convert ndarray 
        """
        if min(labels) < 0:
            labels = (labels).reshape((-1,1)).astype(np.float32)
        else:
            labels = (labels).astype(np.int64)

        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return image, label

def fetch_trainloader(config, shuffle=True):
    """Loads dataset and returns corresponding data loader."""

    total_batch_size = config.batch_size*config.num_clients
    transform = T.Compose([
        T.Resize(config.sample_size),
        T.ToTensor(),
    ])
    
    with open(os.path.join(config.train_data_dir, "datapair.dat"), "rb") as fp:
        record = pickle.load(fp)

    datapair = record["data_pair"]
    root_dir = record["root"]
    dataset = ImageDataset(root_dir, datapair, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=total_batch_size, shuffle=shuffle, num_workers=1, drop_last=True)

    return data_loader

def fetch_mnist_loader(batch_size, train=False, shuffle=True):
    """
    Creates a DataLoader for the MNIST dataset.
    
    Args:
        batch_size (int): Number of samples per batch.
        train (bool): If True, loads the training set. If False, loads the test set.
        shuffle (bool): Whether to shuffle the dataset.
        
    Returns:
        DataLoader: A DataLoader for the MNIST dataset.
    """
    # Define the transformations to apply to the data (normalize and convert to tensor)
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),  # Convert image to PyTorch tensor
        T.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
    ])
    
    # Load the MNIST dataset
    mnist_dataset = MNIST(root='./data', train=train, download=True, transform=transform)
    
    # Create a DataLoader for the dataset
    data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    
    return data_loader