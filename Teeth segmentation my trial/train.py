import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act

from loader import TuftsDataset
from augmentation import get_transorms
from metric import MeanDiceScore
from loss import MeanDiceLoss
from engine import train_one_epoch, test_one_epoch


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Teeth model training script", add_help=False)
    parser.add_argument("-md", "--model_dir", type=str, help="model directory")
    parser.add_argument("-d", "--device", default="mps", type=str, help="GPU-ID position")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="GPU-ID position")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("-lr", "--learning_rate", default=1.e-4, type=float, help="learning rate")
    parser.add_argument("-ne", "--num_epochs", default=100, type=int, help="number of epochs")
    args = parser.parse_args()
    # load data file
    jfile = json.load(open("data.json"))
    class_names = jfile["class_names"]
    num_classes = len(class_names)
    #print(list(jfile["class_weights"].values()))
    class_weights = torch.tensor(list(jfile["class_weights"].values()), dtype=torch.float32)
    
    new_shape = (256, 512)
    bright_range = (0.8, 1.2)
    rotation_range = (-np.pi/36, np.pi/36)
    scale_range = (0.8, 1.2)
    train_transform = get_transorms(
        new_shape, 
        bright_range=bright_range, 
        rotation_range=rotation_range, 
        scale_range=scale_range, 
        num_classes=num_classes
    )
    print(train_transform)
