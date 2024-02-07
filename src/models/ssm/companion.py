import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
from einops import rearrange, repeat

from . import Kernel
from util import krylov

class CompanionKernel(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize trainable parameters
        self.init_weights()
        
        # Shift matrix initialization
        self.shift_matrix = torch.zeros(self.n_channels, 
                                        self.n_heads, 
                                        self.d_kernel, 
                                        self.d_kernel)
        self.shift_matrix[:, :, 1:, :-1] = torch.eye(self.d_kernel - 1)
        self.a_padding = torch.zeros(*self._fp)
        self.a_padding[:, :, -1] = 1.
        
    def init_weights(self):
        # A, B, C, D matrices are trainable by default
        trainable = True if self.requires_grad is True else False
        
        # A matrix
        a = torch.randn(*self._fp)
        self.register("a", a, trainable, lr=None, wd=None)
        
        # B matrix
        b = torch.randn(*self._fp)  # learn_b is True 
        self.register("b", b, trainable, lr=None, wd=None)
        
        # C matrix
        c = torch.randn(*self._fp)  # learn_c is True
        self.register("c", c, trainable, lr=None, wd=None)
        
        # D matrix (skip connection) - learn_d is True
        # if self.skip_connection:
        d = torch.randn(self.n_channels, self.n_heads)
        self.register("d", d, trainable, lr=None, wd=None)
    
    def norm(self, x, ord=1):
        # x.shape = C x H x D
        x_norm = torch.linalg.norm(x, ord=ord, dim=2, keepdim=True)
        x = x / x_norm  # if x_norm[:, 0].item() != 0 else x 
        return x
    
    def matrix_power(self, l, c, b, a):
        ch, h, d = b.shape 
        # Construct companion matrix
        A = self.shift_matrix.to(a.device) + oe.contract(
            'c h i, c h j -> c h j i',  self.a_padding.to(a.device), a
        )
        # Use repeated squares
        g = krylov(l, A, b, c)  # Need to import
        return g
    
    def get_hidden_state(self, l, u):
        # a, b, c.shape is C x H x D
        a = self.norm(self.a, ord=self.norm_ord)
        k = self.matrix_power(l, None, self.b, a) # C x H x D x L
        x = self.fft_conv_d(u, k)  # B x C x H x L x D
        return x
    
    def get_kernel(self, u, c, l):
        a = self.norm(self.a, ord=1)
        k = self.matrix_power(l, c, self.b, a).to(u.device)
        return k
    
    def convolve(self, u, c):
        l = u.size(-1)
        f = self.get_kernel(u, c, l)
        y = self.fft_conv(u, f)
        return y
    
    def forward(self, u, n_hidden_state: int=0):
        # Input shape is (B, H, L)
        # n_hidden_state is ignored
        y = self.convolve(u, c=self.c)
        if self.skip_connection:
            y = y + oe.contract('b h l , c h -> b c h l', u, self.d) 
        # We now output multiple outputs
        return rearrange(y, 'b c h l -> b (c h) l'), None  
    