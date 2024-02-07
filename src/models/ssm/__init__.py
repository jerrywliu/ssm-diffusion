import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

from .. import OurModule

class Kernel(OurModule):
    def __init__(self,
                 d_kernel,
                 n_heads,
                 n_channels,  # 1
                 skip_connection,
                 closed_loop,
                 train,
                 inference_only=False):
        super().__init__()
        
        self.d_kernel   = d_kernel
        self.n_heads    = n_heads
        self.n_channels = n_channels
        
        self.skip_connection = skip_connection
        self.closed_loop     = closed_loop
        self.requires_grad   = train
        self.inference_only  = inference_only
        
        self._fp = (self.n_channels, self.n_heads, self.d_kernel)
        
    def fft_conv(self, u, v):
        L   = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        v_f = torch.fft.rfft(v, n=2*L) # (C H L)

        y_f = oe.contract('b h l, c h l -> b c h l', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)
        return y
    
    def fft_conv_d(self, u, v):  # Used to get hidden state
        # Can we do this more efficiently so we don't have to compute D x D hidden-state?
        L   = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2*L, dim=2).unsqueeze(-1) # (B H L 1)
        v_f = torch.fft.rfft(v, n=2*L, dim=3).unsqueeze(-1) # (C H D L 1)

        y_f = oe.contract('b h l i, c h d l i -> b c h l d', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L, dim=3)[:, :, :, :L, :] # (B C H L D)
        return y
    
    def init_weights(self):
        raise NotImplementedError('Call in child class')
    
    def forward(self, u):
        raise NotImplementedError('Call in child class')
    