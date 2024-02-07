import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
from einops import rearrange, repeat


class DenseEncoder(nn.Module):
    def __init__(self,
                 input_dim:      int,                 
                 output_dim:     int,
                 activation:     str=None,
                 dropout:        float=0.,  # Not implemented yet
                 n_layers:       int=1,
                 n_activations:  int=0,
                 pre_activation: bool=False,
                 input_shape:    str='bld',
                 hidden_dim:     int=None):
        """
        Fully-connected network 
        """
        super().__init__()
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.input_shape   = input_shape
        
        if activation == "gelu":
            self.activation = nn.GELU()
        self.dropout        = None  # DropoutNd
        self.n_layers       = n_layers
        self.n_activations  = n_activations
        self.pre_activation = pre_activation
        
        self.initialize_layers()
        
    def initialize_layers(self):
        n_layers_to_init = self.n_layers
        n_activations_to_init = self.n_activations
        
        if self.hidden_dim is None:  # Probs not great, but implicitly handle
            self.hidden_dim = self.output_dim
            
        # Add layers
        if self.n_activations > self.n_layers or self.pre_activation:
            layers = [self.activation]
            n_activations_to_init -= 1
        else:
            layers = []
            
        while n_layers_to_init > 0 or n_activations_to_init > 0:
            if n_layers_to_init == self.n_layers:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif n_layers_to_init > 1:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            elif n_layers_to_init == 1:
                layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            
            if n_activations_to_init > 0:
                layers.append(self.activation)
            
            n_layers_to_init -= 1
            n_activations_to_init -= 1
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.input_shape == 'bdl':
            x = rearrange(x, 'b d l -> b l d')
        return self.layers(x)
    