import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
from einops import rearrange, repeat

from .companion import CompanionKernel
from .util import krylov

class ShiftKernel(CompanionKernel):
    def __init__(self, n_hidden_state, **kwargs):
        self.n_hidden_state = n_hidden_state
        self.target_length = None
        super().__init__(**kwargs)
        
    def init_weights(self):
        # A matrix ix fixed shift matrix
        A = torch.zeros(self.n_channels, self.n_heads, 
                        self.d_kernel, self.d_kernel)
        A[:, :, 1:, :-1] = torch.eye(self.d_kernel - 1)
        self.register("A", A, trainable=False, lr=None, wd=None)
        
        # B matrix is fixed
        b    =  torch.zeros(self.d_kernel).float()
        b[0] = 1.
        b    = repeat(b, 'd -> c h d', 
                      c=self.n_channels, h=self.n_heads).clone().contiguous()
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C, K, D matrices are trainable by default
        trainable = True if self.requires_grad is True else False
        
        c = torch.randn(*self._fp)
        self.register("c", c, trainable, lr=None, wd=None)
        
        if self.closed_loop:
            k = torch.randn(*self._fp)
            self.register("k", k, trainable, lr=None, wd=None)
        
        # D matrix (skip connection) is trainable
        d = torch.randn(self.n_channels, self.n_heads)
        self.register("d", d, trainable, lr=None, wd=None)
        
    def matrix_power(self, l, c, b, a=None):
        ch, h, d = b.shape 
        # Use repeated squares
        g = krylov(l, self.A, b, c)  # Need to import
        return g
        
    def get_hidden_state(self, l, u):
        """
        Currently not called
        """
        # u.shape is C x H x D
        k = self.matrix_power(l, None, self.b, self.a) # C x H x D x L
        x = self.fft_conv_d(u, k)  # B x C x H x L x D
        return x
        
    def get_kernel(self, u, c=None, l=None):
        l = u.size(-1) if l is None else l
        c = self.c if c is None else c
        k = self.matrix_power(l, c, self.b, self.a).to(u.device)
        return k
    
    def convolve(self, u, A, l, c=None):
        c = self.c if c is None else c
        f = krylov(l, A, self.b, c)
        y = self.fft_conv(u, f)
        y = rearrange(y, 'b c h l -> b (c h) l')
        return y
    
    def recurse_hidden_state(self, u, A, l, n_hidden_state):
        # Compute hidden state
        f = krylov(n_hidden_state, A, self.b, None)
        x  = self.fft_conv_d(u, f)
        BK = oe.contract('c h i, c h j -> c h i j', self.b, self.k)
        x  = krylov(L=l - n_hidden_state,  
                    A=A + BK, 
                    b=x[:, :, :, n_hidden_state - 1, :], 
                    c=None)
        return x
    
    def norm(self, x, ord=1):
        # x.shape = C x H x D
        x_norm = torch.linalg.norm(x, ord=ord, dim=2, keepdim=True)
        x = x / x_norm
        return x
    
    def forward(self, u):
        ch, h, d = self.b.shape
        b,  d, l = u.shape
        
        # Inference
        if self.target_length is not None: l = self.target_length
        
        # Outputs
        y, y_, u_ = None, None, None
        
        if self.closed_loop:  
            # Can also compute from just first hidden_state = B * e, e ~ N(0, I)
            # - Then also only initialize the A + BK:
            #   - Once for each sample during training
            #   - Once for all samples during test  (Faster)
            
            # Compute A = (A + BK)^i for i in [0, ..., n - 1]
            
            # Hacks for stabilization right now
            k = self.norm(self.k.clamp(1 / 16, 1), ord=1)
            BK = oe.contract('c h i, c h j -> c h i j', self.b, k)
            # Compute 1st hidden state: x_1 = Bu_0
            x1 = oe.contract('b h l, c h l -> b c h l', u[:, :, 0].view(b, d, 1), self.b)
            x1 = rearrange(x1, 'b c h l -> b (c h) l')
            # Compute CBu, C(A + BK)x_1, ..., C(A + BK)^(l-1)x_1
            y  = krylov(l, self.norm(self.A + BK), x1, c=self.c) 
            
            # # Old way to do this (memory-wise more expensive because initializes hidden-state x?
            # # But twice as slow bc does computation twice
            # # Also repeats the original hidden_state 
            # x = self.recurse_hidden_state(u, A, l, self.n_hidden_state)
            # y = torch.einsum('...nl, ...n -> ...l', x, self.c).contiguous()
            # y = rearrange(y, 'b c h l -> b (c h) l')
            
            # During training, compute closed-loop outputs and open-loop outputs
            if not self.inference_only:
                # u_ = torch.einsum('...nl, ...n -> ...l', x, self.k).contiguous()
                # u_ = rearrange(u_, 'b c h l -> b l (c h)')  # Save 1 extra rearrange
                u_ = krylov(l, self.norm(self.A + BK), x1, c=self.k)  # B H L
                
                # Compute open-loop
                y_ = self.convolve(u, self.A, l)  # .cpu()  # Move back to gpu before loss
        else:
            y = self.convolve(u, self.A, l)
            
        # Return:
        # 1. Closed-loop output prediction (if closed-loop, else open-loop)
        # 2. 2nd-to-last layer prediction
        # 3. Open-loop output prediction
        return y, u_, y_  # y, z[0], z[1]
            
