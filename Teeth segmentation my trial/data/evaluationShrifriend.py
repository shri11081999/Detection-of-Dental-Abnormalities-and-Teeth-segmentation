import os
import argparse
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act

from loaderShriFriend import TuftsDataset
from augmentation import get_transorms
from metric import MeanDiceScore
from loss import MeanDiceLoss

def evaluate(device, model, data_loader, criterion, metric):
    len_dl = len(data_loader)
    Loss, Dice = [], []
    with torch.no_grad():
        model.eval()
        for batch_data in tqdm.tqdm(data_loader, total=len_dl):
            inputs = batch_data["img"].to(device)
            targets = batch_data["seg"].to(device)
            outputs = nn.Softmax(dim=1)(model(inputs))
            loss = criterion(outputs, targets)
            dice = metric(outputs, targets)
            Loss.append(loss.cpu().numpy())
            Dice.append(dice.cpu().numpy())
    return Loss, Dice

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation script", add_help=False)
    parser.add_argument("-md", "--model_dir", type=str, help="model directory")
    parser.add_argument("-d", "--device", default="mps", type=str, help="device type")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="GPU-ID position")
    parser.add_argument("-bs", "--batch_size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.pt")
    assert os.path.exists(model_path), "Model not found"

    with open("data.json") as f:
        jfile = json.load(f)
    class_names = jfile["class_names"]
    num_classes = len(class_names)
    class_weights = torch.tensor(list(jfile["class_weights"].values()), dtype=torch.float32)

    device = torch.device("cuda" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device.")

    new_shape = (256, 512)
    valid_transform = get_transorms(new_shape, num_classes=num_classes)

    train_ds = TuftsDataset(jfile["train"], masking=True, transform=valid_transform)
    valid_ds = TuftsDataset(jfile["valid"], masking=True, transform=valid_transform)
    test_ds = TuftsDataset(jfile["test"], masking=True, transform=valid_transform)

    batch_
