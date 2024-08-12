import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from loader import TuftsDataset
from augmentation import get_transorms
new_shape = (256, 512)
bright_range = (0.8, 1.2)
rotation_range = (-np.pi/36, np.pi/36)
scale_range = (0.8, 1.2)
jfile = json.load(open("data.json"))
class_names = jfile["class_names"]
num_classes = len(class_names)
class_weights = torch.tensor(list(jfile["class_weights"].values()), dtype=torch.float32)
train_transform = get_transorms(
    new_shape, 
    bright_range=bright_range, 
    rotation_range=rotation_range, 
    scale_range=scale_range, 
    num_classes=num_classes
)
valid_transform = get_transorms(
    new_shape, 
    num_classes=num_classes
)
train_ds = TuftsDataset(jfile["train"], masking=True, transform=train_transform)
for i in train_ds:
    print(i)