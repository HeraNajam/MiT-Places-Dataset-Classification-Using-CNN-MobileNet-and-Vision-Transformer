
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

import numpy as np
import pandas as pd

from PIL import Image

## Transformations
im_size = 128

train_transforms = T.Compose( [ T.Resize( im_size ), 
                                T.ToTensor(), ] )

val_transforms   = T.Compose( [ T.Resize( im_size ), 
                                T.ToTensor(), ] )

class CNN(nn.Module):
    def __init__(self):

        # initialize super class
        super(CNN, self).__init__()
        # define conv layers
        self.layer1  = nn.Conv2d( 3, 32,   kernel_size=3, stride=2, padding=1)
        self.layer2  = nn.ReLU()
        self.layer3  = nn.Conv2d( 32, 64,  kernel_size=3, stride=2, padding=1)
        self.layer4  = nn.ReLU()
        self.layer5  = nn.Conv2d( 64, 128, kernel_size=3, stride=2, padding=1)
        self.layer6  = nn.ReLU()
        self.layer7  = nn.Conv2d( 128, 256, kernel_size=5, stride=1, padding=0)
        self.layer8  = nn.ReLU()
        self.layer9  = nn.Conv2d( 256, 512, kernel_size=5, stride=1, padding=0)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Conv2d( 512, 1024, kernel_size=5, stride=1, padding=0)
        self.layer12 = nn.ReLU()
        self.layer13 = nn.Linear( 1024 * 4 * 4, 365 )

    def forward(self, x):

        # apply convolution layers
        # 40 x 201
        x = self.layer1(x)
        # 30 x 160
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = x.view( x.shape[0], -1 ) 
        x = self.layer13(x)

        return x

# Function to load a model
def load_model(model_path):
    if torch.cuda.is_available():
        return torch.load(model_path)
    else:
        return torch.load(model_path, map_location=torch.device('cpu'))

def cnn_predict_function( label, image_name, img, model_path ):
    
    inp = val_transforms( img ).unsqueeze(0)
    gt  = torch.tensor(  [label] , dtype = torch.long )
    
    print(f'Shape of the inp: {inp.shape}')
    print(f'gt: {gt.shape}')
    print(f'gt: {gt}')
    
    # CNN Model....
    cnn = CNN()
    
    # define loss
    loss_func = nn.CrossEntropyLoss() 

    # load best network
    state = load_model( model_path )
    cnn.load_state_dict( state['net'] )
    
    
    # Set the network in evaluation mode
    cnn.eval() 
    
    with torch.no_grad():
        out = cnn(inp)
        
        # Compute loss
        loss = loss_func(out, gt)
        
        
        _, pred = torch.max(out, 1)  # Get the index of the max logit
    
    return pred, loss