import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
from einops import rearrange, repeat


class RepeatEncoder(nn.Module):
    def __init__(self,
                 input_dim:   int,                 
                 output_dim:  int,
                 input_shape: str='bld'):
        """
        Repeats inputs along a certain dimension D by output_dim // D
        """
        super().__init__()
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.input_shape   = input_shape
        
        self.initialize_layers()
        
    def initialize_layers(self):
        pass
        
    def forward(self, x):
        if self.input_shape == 'bdl':
            x = repeat(x, 'b d l -> b (r d) l', 
                       r=self.output_dim // x.shape[1])
            x = rearrange(x, 'b d l -> b l d')
        else:
            x = repeat(x, 'b l d -> b l (r d)', 
                       r=self.output_dim // x.shape[2])        
        return x
    