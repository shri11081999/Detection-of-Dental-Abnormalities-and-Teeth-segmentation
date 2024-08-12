import os
import argparse
import json
import tqdm
import numpy as np
import pandas as pd
from PIL import ImageTk,Image  
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act

from loader import TuftsDataset
from augmentation import get_transorms
from metric import MeanDiceScore
from loss import MeanDiceLoss
from monai.visualize.utils import blend_images
from mymetric import MyDiceScore

def evaluate(device, model, data_loader, criterion, metric,newmymetric):

    len_dl = len(data_loader)
    Loss, Dice , MyDice= [], [],[]
    
    with torch.no_grad():
        model.eval()
        for batch_data in tqdm.tqdm(data_loader, total=len_dl):
            #print("before to device (img):",batch_data["img"].shape)
            #print("before to device (seg):",batch_data["seg"].shape)
            inputs = batch_data["img"].to(device)
            targets = batch_data["seg"].to(device)
            outputs = model(inputs.unsqueeze(0))
            outputs = nn.Softmax(dim=1)(outputs)
            

            #my addition for confidence score:
            ##outputs[0][0]=torch.where(outputs[0][0]<0.01,0,1)
            ##outputs[0][1]=torch.where(outputs[0][1]>0.99,1,0)


            loss = criterion(outputs, targets)
            dice = metric(outputs, targets)
            mydice=newmymetric(outputs,targets)

            Loss.append(loss.cpu().numpy())
            Dice.append(dice.cpu().numpy())
            MyDice.append(mydice.cpu().numpy())
            #break

    return Loss, Dice, MyDice

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation script", add_help=False)
    parser.add_argument("-md", "--model_dir", type=str, help="model directory")
    parser.add_argument("-d", "--device", default="mps", type=str, help="device type")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="GPU-ID position")
    parser.add_argument("-bs", "--batch_size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.pt")
    assert os.path.exists(model_path) == True

    jfile = json.load(open("data.json"))
    class_names = jfile["class_names"]
    num_classes = len(class_names)
    class_weights = torch.tensor(list(jfile["class_weights"].values()), dtype=torch.float32)

    # set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = args.device + ":" + str(args.gpu_id)
    else:
        device = "cpu"
    print(f"Using {device} device.")

    # create datasets
    new_shape = (256, 512)

    valid_transform = get_transorms(
        new_shape, 
        num_classes=num_classes
    )
    print(valid_transform)
    train_ds = TuftsDataset(jfile["train"], masking=True, transform=valid_transform)
    valid_ds = TuftsDataset(jfile["valid"], masking=True, transform=valid_transform)
    test_ds = TuftsDataset(jfile["test"], masking=True, transform=valid_transform)

    # create dataloaders
    '''batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)'''

    # build model
    model = UNet(
        spatial_dims = 2,
        in_channels = 1,
        out_channels = num_classes,
        channels = (32, 64, 128, 256, 512),
        strides = (2, 2, 2, 2),
        num_res_units = 2,
        norm = Norm.BATCH,
        act = Act.LEAKYRELU
    ).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded.")

    # loss function and dice metric
    metric = MeanDiceScore(softmax=False, weights=None, epsilon=0.)
    criterion = MeanDiceLoss(softmax=False, weights=class_weights)
    newmymetric= MyDiceScore(softmax=False, weights=None, epsilon=0.)
    # evaluate model
    test_loss, test_dice, test_mydice = evaluate(device, model, test_ds, criterion, metric,newmymetric)
    

    print(f"Test: {np.mean(test_loss, 0):.4f} loss, {np.nanmean(test_dice, 0):.4f} dice, {np.nanmean(test_mydice, 0):.4f} MY dice mean.")
    
    out = {
        "file_name": [],
        "set_name": ["test"]*len(jfile["test"]),
        "loss": test_loss,
        "dice": test_dice,
        "My Dice": test_mydice
    }
    #print(train_loss)
    for set_name in ["test"]:
        for i, data in enumerate(jfile[set_name]):
            out["file_name"].append(data["img"])

    df = pd.DataFrame(out)
    df.sort_values(by=["file_name"])
    df.to_csv("evaluation_results.csv")