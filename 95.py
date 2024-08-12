import torch
from PIL import Image 
from torchvision import transforms
import torchvision
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score as sk_accuracy
from torch.utils.data import Dataset
from torchvision import transforms
import glob 
import os 
from torch.utils.data import DataLoader 
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50 
from torch.utils.data import random_split
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import json
from torch.utils.data import random_split
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from PIL import ImageTk,Image  
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import tqdm
import os
import gc


# Create Dataset to easily load data from files. 
# Radiograph transforms -- extra transforms to apply to radiograph tensor
# mask_transform -- extra transforms to apply to mask tensor 
class MaskDataset(Dataset):
  def __init__(self, radiograph_transform = None, mask_transform = None, path = 'C:/Users/dixit/Desktop/teeth codes/tuft dental database/'):
    if radiograph_transform is not None: 
      self.radiograph_transform = transforms.Compose([
           transforms.ToTensor(),
           radiograph_transform
      ])
    else: 
      self.radiograph_transform = transforms.Compose((
          transforms.ToTensor(),
          transforms.Resize(img_size)
      ))

    if mask_transform is not None:
      self.mask_transform = transforms.Compose([
          transforms.ToTensor(),
          mask_transform
      ])
    else:
      self.mask_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize(img_size)
      ])

    
    self.path = path
    files = os.listdir(self.path + 'Radiographs/')
    self.files = files
    '''files2 = glob.glob(self.path + 'Radiographs/*')
    files2 = [(x.split('/')[2]).lower() for x in files2]
    files2 = [(x.split('\\')[-1]).lower() for x in files2]
    print(files2)'''
    #print(self.files)
    with open(self.path + 'Expert/expert.json', 'r') as f:
      metadata = json.load(f)
    self.metadata = {}
    for img_metadata in metadata:
      # Make it easy to query by ID 
      key = img_metadata['External ID']
      self.metadata[key] = img_metadata
      # Add tag if it has abnormality or not 
      objects = img_metadata['Label']['objects']
      # TODO: Remove if we want objects 
      del img_metadata['Label']
      self.metadata[key]['abnormality_exists'] = not (len(objects) == 1 and objects[0]['title'] == "None")

  def _to_deeplab(self, mask):
    mask = self.mask_transform(mask)[0].to(torch.int64)
    return torch.nn.functional.one_hot(mask, num_classes = 2).permute((2, 0, 1)).float()

  def __getitem__(self, idx):
    radiograph = Image.open(self.path + 'Radiographs/' + self.files[idx].upper())
    teeth_mask = Image.open(self.path + 'Segmentation/teeth_mask/' + self.files[idx].lower())
    abnormality_mask = Image.open(self.path + 'Expert/mask/' + self.files[idx].upper())

    radiograph = self.radiograph_transform(radiograph)

    # Format deeplab expects
    # All mask channels are the same
    teeth_mask = self._to_deeplab(teeth_mask)
    abnormality_mask = self._to_deeplab(abnormality_mask)

    img_metadata = self.metadata[self.files[idx].upper()]

    label = torch.tensor(img_metadata['abnormality_exists']).to(torch.int64)
    label = torch.nn.functional.one_hot(label, num_classes = 2)

    # NOTE: These both have three channels, but all channels have same value (greyscale)
    return {'radiograph' : radiograph, 'teeth_mask' : teeth_mask,
            'abnormality_mask': abnormality_mask,
            'metadata' : img_metadata,
            'label': label}
    
  
  def __len__(self):
    return len(self.files)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

"""Helper function to create deeplabv3 model with custom classifier head and specified freezed layers"""



def segmentation_metrics(val_dataloader, model):
  """ Expects input in N x CL x H x W, CL is number of classes"""
  total_iou = 0
  total_accuracy = 0
  total_f1 = 0
  total_samples = 0
  for batch in tqdm.tqdm(val_dataloader):
    radiograph = batch['radiograph'].cuda()
    num_samples = radiograph.shape[0]
    mask = batch['teeth_mask']
    predicted_mask = model(radiograph)['out']
    predicted_mask = torch.flatten(predicted_mask.argmax(axis = -3)).detach().cpu().numpy()
    ground_truth_mask = torch.flatten(mask.argmax(axis = -3)).detach().cpu().numpy()
    iou = jaccard_score(predicted_mask, ground_truth_mask, average='weighted')
    accuracy = sk_accuracy(predicted_mask, ground_truth_mask)
    f1 = f1_score(predicted_mask, ground_truth_mask)

    total_iou += num_samples * iou
    total_accuracy += num_samples * accuracy
    total_f1 += num_samples * f1
    total_samples += num_samples
    del radiograph

  iou = total_iou / total_samples
  accuracy = total_accuracy / total_samples 
  f1 = total_f1 / total_samples

  print("Validation -- IOU: ",  iou, " ACCURACY:", accuracy , " F1: ", f1)



def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and (p.grad!=None):
            layers.append(n)
            #index = output.cpu().data.numpy().argmax()
            ave_grads.append(p.grad.cpu().abs().mean())
    #ave_grads.cpu()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def plot_grad_flow2(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        print(n)


""" Helper function to train  amodel with given hyperparameters """
def training_loop(train_dataset, val_dataset, model, optim = None, num_epochs = 15, lr=2e-4, reg=1e-5, batch_size=4, print_every = 100):
  losses = []
  losses2 = []
  train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
  segmentation_loss_fn = DiceLoss()
  class_loss_fn = nn.CrossEntropyLoss()
  myloss = nn.BCELoss()
  # Configure optimizer 
  '''
  model = MultiHeadedModel().cuda()
  checkpoint = torch.load("./checkpoint/model_teeth_ori_10epochs_nofreeze_260510_segonly_2batch.pt")
  model.load_state_dict(checkpoint["model_state_dict"])  
  optim = torch.optim.Adam(model.parameters(), lr = 2e-4, weight_decay = 1e-5)
  optim.load_state_dict(checkpoint["optimizer_state_dict"])'''


  if optim == None:
    optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = reg)
    scheduler = StepLR(optim, step_size=4, gamma=0.1)
  model.train()
  
  for i in range(num_epochs):
    print("EPOCH", i)
    scheduler.step()
    print('Epoch-{0} lr: {1}'.format(i, optim.param_groups[0]['lr']))
    print()
    total_loss = 0
    total_imgs = 0
    total_batches = 0
    
    for batch in tqdm.tqdm(train_dataloader):
      # Data
      radiograph = batch['radiograph'].cuda()
      mask = batch['teeth_mask'].cuda()
      #to forget previous back prop calculations 
      optim.zero_grad() # Zero gradient 
      # Run model
      y_hat = model(radiograph)
      y_hat_mask=y_hat['out']
      # Compute loss 
      loss1 = segmentation_loss_fn(y_hat_mask, mask.float())
      # Backpropogate loss
      loss1.backward()
      plot_grad_flow(model.named_parameters())
      # Update gradients / gradient discent step
      optim.step()
      # Random Bookkeeping 
      total_loss += loss1.data * radiograph.shape[0]
      total_imgs += radiograph.shape[0]
      total_batches += 1

      if total_batches % print_every == 0:
        print('Loss', (total_loss / total_imgs).item())
      losses.append((total_loss / total_imgs).item())
      #break
    #break
    segmentation_metrics(val_dataloader, model)
  #return
  model.eval()
  segmentation_metrics(val_dataloader, model)
  dict_to_save = {
      "epoch": 0,
      "model_state_dict": None,
      "optimizer_state_dict": None,
      "history": None,
      "losses": None,
      "losses2": None
  }
  dict_to_save["model_state_dict"] = model.state_dict()
  dict_to_save["optimizer_state_dict"] = optim.state_dict()
  dict_to_save["losses"] = losses
  dict_to_save["losses2"] = None
  checkpoint_dir="./checkpoint"
  if os.path.exists(checkpoint_dir) == False:
      os.makedirs(checkpoint_dir)
  #model_path = os.path.join(checkpoint_dir, "trial.pt")
  model_path = os.path.join(checkpoint_dir, "model_teeth_ori_15epochs_nofreeze_260510_segonly_4batch.pt")
  torch.save(dict_to_save, model_path)
  return losses

if __name__ == '__main__':
    torch.cuda.set_device('cuda:0')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6, max_split_size_mb:128"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    img_size = (260, 510)
    # Data augmentation for radiographs
    # Gaussian smoothing, random adjust sharpness and random autocontrast
    radiograph_transforms = transforms.Compose([
       transforms.Resize(img_size),                                      
      #  transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.5)),
      #  transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=0.5),
      #  transforms.RandomAutocontrast(p=0.5),
      transforms.RandomHorizontalFlip(p=0.5),
       # transforms.RandomCrop((220, 360))
       ]
    )
    dataset = MaskDataset(radiograph_transform=radiograph_transforms)
    from torch.utils.data import random_split

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=1245)
    #splits = [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    #train_dataset, val_dataset = random_split(dataset, splits)
    for i in val_dataset:
      print(i['metadata'])
    
    count = 0
    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                    progress=True, 
                                                    aux_loss=True,
                                                   pretrained_backbone=True)
    
    model.classifier = DeepLabHead(2048, 2)
    model = model.cuda()
    # Set the model in training mode
    model.train()
    #print(model)
    losses=training_loop(train_dataset, val_dataset, model)
    plt.show()
