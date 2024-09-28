
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


# Function to load a model
def load_model(model_path):
    if torch.cuda.is_available():
        return torch.load(model_path)
    else:
        return torch.load(model_path, map_location=torch.device('cpu'))

def mobnet_predict_function( label, image_name, img, model_path ):
    
    inp = val_transforms( img ).unsqueeze(0)
    gt  = torch.tensor(  [label] , dtype = torch.long )
    print(f'Shape of the inp: {inp.shape}')
    print(f'gt: {gt.shape}')
    print(f'gt: {gt}')
    
    # Define the model architecture (MobileNetV2)
    # Load pre-trained weights
    net = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')  
    
    # Adjust the final classification layer
    # Get the number of input features for the last layer
    num_ftrs = net.classifier[1].in_features  
    
    net.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 640),  
        nn.GELU(),  
        nn.Linear(640, 365)  
    )

    print()
    
    # Freeze pre-trained layers and unfreeze the classifier for fine-tuning
    for key, value in dict( net.named_children() ).items():
        if "classifier" in key:
            for param in value.parameters():
                param.requires_grad = True
                # Print unfrozen layers (classifier)
                # print(f"Unfreezing layer: {key}, Parameter shape: {param.shape}")  
        else:
            for param in value.parameters():
                param.requires_grad = False
                # print(param)  
    
    # define loss
    loss_func = nn.CrossEntropyLoss() 
    
    # load best network
    state = load_model( model_path )
    net.load_state_dict( state['net'] )
    
    
    # Set the network in evaluation mode
    net.eval() 
    
    with torch.no_grad():
        out = net(inp)
        
        # Compute loss
        loss = loss_func(out, gt)
        
        
        _, pred = torch.max(out, 1)  # Get the index of the max logit
    
    return pred, loss