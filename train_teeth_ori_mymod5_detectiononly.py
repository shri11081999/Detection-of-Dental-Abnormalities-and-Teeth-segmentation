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
  def __init__(self, radiograph_transform = None, mask_transform = None, path = 'C:/Users/__/tuft dental database/'):
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
    files = os.listdir(self.path + '/Radiographs/Radiographs/')
    self.files = files
    #print(self.files)
    with open(self.path + 'Expert/Expert/expert.json', 'r') as f:
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
    radiograph = Image.open(self.path + '/Radiographs/Radiographs/' + self.files[idx].upper())
    teeth_mask = Image.open(self.path + 'Segmentation/Segmentation/teeth_mask/' + self.files[idx].lower()).convert("L")
    abnormality_mask = Image.open(self.path + 'Expert/Expert/mask/' + self.files[idx].upper())

    radiograph = self.radiograph_transform(radiograph)
    teeth_mask = self.radiograph_transform(teeth_mask)

    # Format deeplab expects
    # All mask channels are the same
    #teeth_mask = self._to_deeplab(teeth_mask)
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


class Myconvolution(nn.Module):
    def __init__(self):
        super(Myconvolution,self).__init__()

        #use this code in main to find/verify output size: N=batch size, C= channels, H=height, W=width
        '''input = torch.empty(N,C,H,W).random_(256)
        print("Input Size:",input.size())
        conv = nn.Conv2d(2, 3, 2, stride=2)
        output = conv(input)
        print("Output Size:",output.size())'''


        #original image=black and white teeth mask, size=1[no rgb]*260[resized width]*510[resized height]
        self.conv1=nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,stride=1,padding=0)
        #formula for width/height=  ((width/height-kernel_size+2*padding)/stride)+1
        #we get 2*258*508
        self.bn1=nn.BatchNorm2d(num_features=2)
        #shape=2*258*508
        self.relu1=nn.ReLU()
        #shape=2*258*508
        '''self.pool=nn.MaxPool2d(kernel_size=2)
        #reduce the image size by factor of 2
        #shape=2*128*253'''
        self.conv2=nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3,stride=2)
        #shape=4*128*253
        self.bn2=nn.BatchNorm2d(num_features=4)
        self.relu2=nn.ReLU()

        self.conv3=nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,stride=2)
        #shape=8*63*126
        self.bn3=nn.BatchNorm2d(num_features=8)
        self.relu3=nn.ReLU()

        self.conv4=nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,stride=2)
        #shape=8*31*62
        self.bn4=nn.BatchNorm2d(num_features=8)
        self.relu4=nn.ReLU()

        self.conv5=nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,stride=2)
        #shape=8*15*30
        self.bn5=nn.BatchNorm2d(num_features=8)
        self.relu5=nn.ReLU()

        self.fc1 = nn.Linear(3600, 1000)
        #remember to keep atleast 2 instances in batch to avoid an error as this 1d batchnorm expects at least 2
        self.bn6 = nn.BatchNorm1d(1000)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 50)
        self.bn7 = nn.BatchNorm1d(50)
        self.relu7 = nn.ReLU()
        self.fc3 = nn.Linear(50, 2)
    def forward(self, batch):
        output = self.conv1(batch)
        output = self.bn1(output)
        output = self.relu1(output)
        
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        
        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)
        
        output = self.conv5(output)
        output = self.bn5(output)
        output = self.relu5(output)
        
        output=output.view(-1,3600)
        output=self.fc1(output)
        output = self.bn6(output)
        output = self.relu6(output)

        output=self.fc2(output)
        output = self.bn7(output)
        output = self.relu7(output)

        output = self.fc3(output)

        return output
        


def classifier_evaluation_metrics(val_dataloader, model):
  """ Expects input in N x CL x H x W, CL is number of classes"""
  
  total_accuracy = 0
  total_samples = 0
  confusion_matrix = np.zeros((2, 2))
  for batch in tqdm.tqdm(val_dataloader):
    radiograph = batch['teeth_mask'].cpu()
    num_samples = radiograph.shape[0]
    label = batch['label'].detach().cpu().numpy()
    mine=model(radiograph)
    
    predicted_label = (model(radiograph) > 0.5).float().squeeze(0).detach().cpu().numpy().astype(int)
    '''print(radiograph.shape)
    print("Model prediction:")
    print(mine)
    print("Predicted label:")
    print(predicted_label)
    print("Actual label:")
    print(label)'''
    accuracy = sk_accuracy(predicted_label, label)
    '''cm = sklearn.metrics.confusion_matrix(predicted_label, label, normalize = 'all')
    confusion_matrix += cm * num_samples'''

    total_accuracy += num_samples * accuracy
    total_samples += num_samples
    del radiograph
    '''accuracy = total_accuracy / total_samples 
    print("ACCURACY", accuracy)
    return'''

  '''confusion_matrix = confusion_matrix / total_samples 
  cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix)
  cm_display.plot()
  plt.show()'''
  print("Predicted label:")
  print(mine)
  print("Actual label:")
  print(label)
  print("Metadata:")
  print(batch['metadata'])
  accuracy = total_accuracy / total_samples 
  print("ACCURACY", accuracy)


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
def training_loop(train_dataset, val_dataset, model, optim = None, num_epochs = 100, lr=2e-4, reg=1e-5, batch_size=15, print_every = 100):
  losses = []
  losses2 = []
  train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
  segmentation_loss_fn = DiceLoss()
  class_loss_fn = nn.CrossEntropyLoss()
  myloss = nn.BCELoss()
  # Configure optimizer 
  '''
  model = MultiHeadedModel().cpu()
  checkpoint = torch.load("./checkpoint/model_teeth_ori_10epochs_nofreeze_260510_segonly_2batch.pt")
  model.load_state_dict(checkpoint["model_state_dict"])  
  optim = torch.optim.Adam(model.parameters(), lr = 2e-4, weight_decay = 1e-5)
  optim.load_state_dict(checkpoint["optimizer_state_dict"])'''

  if optim == None:
    optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = reg)
    scheduler = StepLR(optim, step_size=20, gamma=0.1)
  
  print()
  print("Abnormality began")
  for i in range(num_epochs):
    model.train()
    print("EPOCH", i)
    scheduler.step()
    print('Epoch-{0} lr: {1}'.format(num_epochs, optim.param_groups[0]['lr']))
    print()
    total_loss = 0
    total_imgs = 0
    total_batches = 0
    
    # Run validation 
    #model.eval()
    #classifier_evaluation_metrics(val_dataloader, model)
    #segmentation_metrics(val_dataloader, model)
    for batch in tqdm.tqdm(train_dataloader):
      # Data
      mask = batch['teeth_mask'].cpu()
      label = batch['label'].float().cpu()
      #print(radiograph.shape)
      #to forget previous back prop calculations 
      optim.zero_grad() # Zero gradient 
      #label=array of abnormalities; label[0]=[1 if normal, 1 if abnormal]/[0 if abnormal, 0 if normal]

      # Run model
      y_hat_abnormality = model(mask)
      # Compute loss 
      loss2 = class_loss_fn(y_hat_abnormality, label)
      # Backpropogate loss
      loss2.backward()
      plot_grad_flow(model.named_parameters())
      # Update gradients / gradient discent step
      optim.step()
      #break
      # Random Bookkeeping 
      #if total_batches % print_every == 0:
        #print(loss2)
      losses2.append(loss2.item())
    #break
    
    model.eval()
    classifier_evaluation_metrics(val_dataloader, model)
  #return
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
  dict_to_save["losses"] = None
  dict_to_save["losses2"] = losses2
  checkpoint_dir="./checkpoint"
  if os.path.exists(checkpoint_dir) == False:
      os.makedirs(checkpoint_dir)
  model_path = os.path.join(checkpoint_dir, "model_teeth_minefrom_mask_100epochs_260510_abnormalityonly_15batch.pt")
  torch.save(dict_to_save, model_path)
  return losses2

if __name__ == '__main__':
    os.environ["PYTORCH_GPU_ALLOC_CONF"] = "garbage_collection_threshold:0.6, max_split_size_mb:128"
    os.environ["PYTORCH_NO_GPU_MEMORY_CACHING"] = "1"
    os.environ["GPU_LAUNCH_BLOCKING"] = "1"
    img_size = (260, 510)
    # Data augmentation for radiographs
    # Gaussian smoothing, random adjust sharpness and random autocontrast
    radiograph_transforms = transforms.Compose([
       transforms.Resize(img_size),                                      
      #  transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.5)),
      #  transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=0.5),
      #  transforms.RandomAutocontrast(p=0.5),
      #transforms.RandomHorizontalFlip(p=0.5),
       # transforms.RandomCrop((220, 360))
       ]
    )
    dataset = MaskDataset(radiograph_transform=radiograph_transforms)

    '''print(dataset[0]['metadata'])
    
    print(dataset[0]['teeth_mask'].shape)
    model = Myconvolution().cpu()
    temp=dataset[0]['teeth_mask'].unsqueeze(0).cpu()
    print(temp.shape)
    temp=model(temp)'''
    #print(temp.shape)
    from torch.utils.data import random_split

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=1245)
    #splits = [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    #train_dataset, val_dataset = random_split(dataset, splits)
    #for i in val_dataset:
    #  print(i['metadata'])
    
    count = 0
    '''model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                    progress=True, 
                                                    aux_loss=True,
                                                   pretrained_backbone=True)
    
    model.classifier = DeepLabHead(2048, 2)
    model = model.cpu()
    checkpoint = torch.load("./checkpoint/model_teeth_ori_15epochs_nofreeze_260510_segonly_4batch.pt")
    model.load_state_dict(checkpoint["model_state_dict"])  
    optim = torch.optim.Adam(model.parameters(), lr = 2e-4, weight_decay = 1e-5)
    optim.load_state_dict(checkpoint["optimizer_state_dict"])


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.cuda()'''


    model = Myconvolution().cpu()
    # Set the model in training mode
    model.train()
    #print(model)
    losses=training_loop(train_dataset, val_dataset, model)
    #plt.plot(losses)
    plt.show()
    
