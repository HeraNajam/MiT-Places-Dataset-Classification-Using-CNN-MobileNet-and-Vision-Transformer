
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

## ViT Model
#################################################################################################

class PatchEmbedding( nn.Module ):

    def __init__( self, im_size, patch_size, in_channels = 3, dim_emb = 768 ):
        # Create the Patch Embedding with the following attributes:
        # im_size
        # patch_size

        super().__init__()
        self.im_size    = im_size
        self.patch_size = patch_size

        # Create the embedding using Conv2d instead of linear projection (same operation)
        self.proj = nn.Conv2d(  in_channels  = in_channels,
                                out_channels = dim_emb,
                                kernel_size  = patch_size,
                                stride       = patch_size)

    def forward(self, x): # torch.Size([4, 3, 224, 224])
        # initial input shape ---> ( batch_size, in_channels, im_size, im_size )
        # Compute the the embedding, the final shape has to be (batch_size, num_patches, dim_emb)
        
        # convolution
        x = self.proj(x) 
        # ( batch_size, dim_emb, num_patches_height, num_patches_width )
        # print('\npatches after convolution:\t\t\t' + str(x.shape)) # torch.Size([4, 768, 14, 14])

        x = x.flatten(2)
        # ( batch_size, dim_emb, num_patches_height * num_patches_width )
        # print('patches after flatten on dimension 2 and 3:\t' + str(x.shape)) # torch.Size([4, 768, 196])

        x = x.transpose(1,2)
        # (batch_size, num_patches, dim_emb)
        # print('patch embeddings:\t\t\t\t' + str(x.shape)) # torch.Size([4, 196, 768])

        return x

class Attention( nn.Module ):
  
    def __init__( self, dim, n_heads = 12, qkv_bias = True, attn_p = 0., proj_p = 0. ): # dim = 768
        
        super().__init__()
        self.dim       = dim
        self.n_heads   = n_heads
        self.head_dim  = dim // n_heads # = 64
        self.scale     = self.head_dim ** - 0.5 # dk is a scaling factor to keep softmax under control
        
        # dim*3 because query, key and value are projected together and then splitted
        self.qkv       = nn.Linear( dim, dim * 3, bias = qkv_bias )
        self.attn_drop = nn.Dropout(attn_p)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        
        # input shape --> torch.Size([4, 196, 768])
        """
        x.shape = (n_samples, n_patches + 1, dim) --> n_patches + 1 because of the class token
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim: # sanity check
            raise ValueError

        # from on now n_patches = n_patches + 1
        # create q, k, and v and divide the embedding for the number of heads

        qkv = self.qkv(x) 
        # qkv --> [ n_samples, n_patches, dim*3 ] --> output --> ([4, 196, 2304])
        # print(f'This is the shape of qkv: {qkv.shape}')
        
        qkv = qkv.reshape( n_samples, n_tokens, 3, self.n_heads, self.head_dim ) 
        # divide query key and value for each token for each head 
        # [n_samples, n_patches, 3, n_heads, head_dim] output --> ([4, 196, 3, 12, 64])
        # print(f'This is the shape of qkv after reshape: {qkv.shape}')         
        
        qkv = qkv.permute(2,0,3,1,4) 
        # [3, n_samples, n_heads, n_patches, head_dim] --> ([3, 4, 12, 196, 64])
        # print(f'This is the shape of qkv after permute: {qkv.shape}')

        q, k, v =  qkv[0], qkv[1], qkv[2] 
        # divide query, key and value in different matrix
        # output [n_samples, n_heads, n_patches, head_dim] --> ([4, 12, 196, 64])
        # print(f'Shape of q: {q.shape}') 

        # Compute self-attention between q, k, and v
        # Attention = Softmax( ( Q @ K.T) / scale) @ V with @ = dot product
        k_t = k.transpose(-2, -1) 
        # [n_samples, n_heads, head_dim, n_patches] --> ([4, 12, 64, 196])
        # print(f'shape of k_t: {k_t.shape}')
        
        dp = (q @ k_t) * self.scale 
        #  [n_samples, n_heads, n_patches, n_patches] --> ([4, 12, 196, 196])
        # print(f'Shape of dp: {dp.shape}')

        attn = dp.softmax( dim = -1 ) 
        # last dimenstion --> columns
        # [n_samples, n_heads, n_patches, n_patches]  --> ([4, 12, 196, 196])
        # print(f'Shape of attn" {attn.shape}')
        
        # print(attn.shape)
        # print(attn)
        # print(torch.sum(attn,dim=2))
        # print(torch.sum(attn,dim=3))
        # print(torch.sum(attn,dim=2).shape)
        # print(torch.sum(attn,dim=3).shape)
        # print(torch.sum(attn[0,0,0,:])) # row
        # print(torch.sum(attn[0,0,:,0])) # col

        attn = self.attn_drop( attn ) ## using dropout

        weighted_avg = attn @ v 
        # [n_samples, n_heads, n_patches, head_dim] --> ([4, 12, 196, 64])
        # print(f'This is the shape of weighted average: {weighted_avg.shape}')
        # concatenate heads --> 12 heads
        
        weighted_avg = weighted_avg.transpose(1,2) 
        # (n_samples, n_patches, n_head, head_dim) --> ([4, 196, 12, 64])
        # print(f'This is the shape of weighted average: {weighted_avg.shape}')
        
        weighted_avg = weighted_avg.flatten(2) 
        # (n_samples, n_patches, dim) --> ([4, 196, 768])
        # print(f'This is the shape of weighted average: {weighted_avg.shape}')

        # project heads -- linear layer
        x = self.proj(weighted_avg)
        # (n_samples, n_patches, dim) --> ([4, 196, 768])
        # print(f'Shape of x: {x.shape}')
        
        x = self.proj_drop(x)

        return x

class MLP(nn.Module): # Multi Layer Perceptron
    
    def __init__( self, in_features, hidden_features, out_features, proj_p = 0. ):
        super().__init__()
        self.fc1  = nn.Linear( in_features, hidden_features )
        self.act  = nn.GELU()
        self.fc2  = nn.Linear( hidden_features, out_features )
        self.drop = nn.Dropout( proj_p )

    def forward(self, x): 
        # ([4, 196, 768])
        """
        x.shape() = (n_samples, n_patches + 1, in_features)
        """
        x = self.fc1(x)  
        x = self.act(x)  
        x = self.drop(x) 
        # (n_samples, n_patches + 1, hidden_features) --> ([4, 196, 384])
        # print(f'shape after fc1: {x.shape}')
        
        x = self.fc2(x)  
        x = self.drop(x) 
        # (n_samples, n_patches + 1, hidden_features) --> [4, 196, 10])
        # print(f'shape after fc2: {x.shape}')

        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads=12, mlp_ratio=4.0, qkv_bias=True, proj_p=0., attn_p=0.):
        
        super().__init__()
        self.norm1 = nn.LayerNorm( dim, eps = 1e-06 )
        self.attn  = Attention( dim, n_heads = n_heads, qkv_bias = qkv_bias, attn_p =attn_p, proj_p = proj_p )
        self.norm2 = nn.LayerNorm( dim, eps = 1e-06 )

        hidden_features = int( dim * mlp_ratio )
        # print(f'This is the hidden features: {hidden_features}')
        self.mlp        = MLP( in_features = dim, hidden_features = hidden_features, out_features = dim, proj_p = proj_p )


    def forward(self, x):
        x = x + self.attn( self.norm1(x) ) # residual block
        x = x + self.mlp(  self.norm2(x) ) # residual block

        return x

class VisionTransformer(nn.Module):

    def __init__( self, im_size = 384, patch_size = 16, in_channels = 3, n_classes = 1000, dim_emb = 768,
                  depth = 12, n_heads     = 12, mlp_ratio   = 4., qkv_bias = True, proj_p = 0., attn_p = 0.):
      
        super().__init__()
        self.patch_embed = PatchEmbedding( im_size = im_size, patch_size = patch_size, in_channels = in_channels, dim_emb = dim_emb )

        # nn.Parameters create weights in the network that will be trained together with the other layers
        self.cls_token = nn.Parameter( torch.zeros( 1, 1, dim_emb ) ) # parameter of the network, first 2 dimesions are there for convenience during the operations

        n_patches     = ( im_size // patch_size ) ** 2
        self.pos_emb  = nn.Parameter( torch.zeros( 1, 1+n_patches, dim_emb ) ) # parameter of the network --> 1+self.patch_embed.n_patches beacause it includes class token
        self.pos_drop = nn.Dropout( p = proj_p )

        self.blocks = nn.ModuleList( [ Block( dim = dim_emb, n_heads = n_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias,
                                      proj_p = proj_p, attn_p = attn_p ) for _ in range( depth )  ] )

        self.norm = nn.LayerNorm( dim_emb, eps = 1e-06 )
        self.head = nn.Linear( dim_emb, n_classes ) # Classifier


    def forward(self, x):
        n_samples = x.shape[0]
        x         = self.patch_embed(x) # patch embeddings

        cls_token = self.cls_token.expand( n_samples, -1, -1 ) # [n_samples, 1, dim]
        x         = torch.cat( ( cls_token, x ), dim = 1 ) # add the cls token to the patch embeddings

        x = x + self.pos_emb # tokens creations
        x = self.pos_drop(x)

        # blocks executions
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_final_token = x[ :, 0 ] # just the class token
        x               = self.head( cls_final_token ) #

        return x

## Set Parameters

im_size     = 128

# Transformer parameters
patch_size  = 16
in_channels = 3
dim_emb     = 192
num_heads   = 3
num_blocks  = 2
dropout     = 0.1

# Training parameter

lr          = 10e-4

model = VisionTransformer( im_size     = im_size,
                           patch_size  = patch_size,
                           in_channels = in_channels,
                           n_classes   = 365,
                           dim_emb     = dim_emb,
                           depth       = num_blocks,
                           n_heads     = num_heads,
                           mlp_ratio   = 4.,
                           qkv_bias    = True,
                           proj_p      = 0.,
                           attn_p      = 0. )

#################################################################################################

# Function to load a model
def load_model(model_path):
    if torch.cuda.is_available():
        return torch.load(model_path)
    else:
        return torch.load(model_path, map_location=torch.device('cpu'))
    
def vit_predict_function( label, image_name, img, model_path ):
    
    inp = val_transforms( img ).unsqueeze(0)
    gt  = torch.tensor(  [label] , dtype = torch.long )
    print(f'Shape of the inp: {inp.shape}')
    print(f'gt: {gt.shape}')
    print(f'gt: {gt}')
    
    ### Model
    
    
    
    
    # define loss
    loss_func = nn.CrossEntropyLoss() 
    
    # load best network
    state = load_model( model_path )
    model.load_state_dict( state['net'] )
    
    
    # Set the network in evaluation mode
    model.eval() 
    
    with torch.no_grad():
        out = model(inp)
        
        # Compute loss
        loss = loss_func(out, gt)
        
        
        _, pred = torch.max(out, 1)  # Get the index of the max logit
    
    return pred, loss