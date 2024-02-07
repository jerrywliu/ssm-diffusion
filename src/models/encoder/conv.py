import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
from einops import rearrange, repeat

from . import OurModule

class ConvEncoder(OurModule):
    """
    Preserves the input dimensions
    """
    def __init__(self, 
                 n_kernels:     int,  # L or seq len of input, if smaller than L, when initializing the kernel we repeat
                 d_kernel:      int,  # D or number of heads of prior SSM output, "global" conv / full kernel
                 bias:          bool=False,
                 bidirectional: bool=True,
                 input_shape:   str='bdl'):
        super().__init__()
        
        self.n_kernels     = n_kernels
        self.d_kernel      = d_kernel  # kernel size
        self.bias          = bias
        self.bidirectional = bidirectional
        self.input_shape   = input_shape
        
        self._fp = (1, self.n_kernels, self.d_kernel)  # C L D of prior input
        
        self.init_weights(_id='_a')
        if self.bidirectional:
            self.init_weights(_id='_b')
        
    def init_weights(self, _id: str='_a'):
        weight = torch.randn(*self._fp)
        self.register(f'weight{_id}', weight, train=True, lr=None, wd=None)
        if self.bias:
            bias = torch.randn(1, self.n_kernels, 1)
        else:
            bias = torch.zeros(1, self.n_kernels, 1)
        self.register(f'bias{_id}', bias, train=self.bias, lr=None, wd=None)
        
    def get_kernel(self, 
                   u: torch.FloatTensor, 
                   kernel_weights: torch.FloatTensor,
                   l: int=None) -> torch.FloatTensor:
        """
        Initialize kernel
        - u: input, assume shape is B x D x L
        """
        b, h, _l = u.shape
        l = _l if l is None else l
        weights = repeat(kernel_weights, 'c n d -> c (r n) d', r = h // self.n_kernels)
        k = F.pad(u, pad=(0, l - weights.shape[-1], 0, 0), value=0)
        return k
                            
    def fft_conv(self, u, v):
        """Convolution via FFT from S4"""
        L   = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2 * L) # B x H x L
        v_f = torch.fft.rfft(v, n=2 * L) # C x H x L

        y_f = oe.contract('bhl, chl -> bchl', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2 * L)[..., :L] # B x C x H x L
        return y
                  
    def forward(self, u):
        # Input u shape is (B, H, L)
        b, h, l = u.shape
        k = self.get_kernel(u, self.weights_a)
        bias = repeat(self.bias_a, 'c n d -> c (r n) (l d)', 
                      r=h // self.n_kernels, l=l)
        y = self.fft_conv(u, k) + bias
        
        if self.bidirectional:
            _k = self.get_kernel(u, self.weight_b)
            _bias = repeat(self.bias_b, 'c n d -> c (r n) (l d)', 
                           r=h // self.n_kernels, l=l)
            y = self.fft_conv(torch.flip(u, [-1]), _k) + _bias
            
        return y
        