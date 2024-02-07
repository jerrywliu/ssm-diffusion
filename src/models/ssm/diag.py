import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
from einops import rearrange, repeat

from . import Kernel

class DiagKernel(Kernel):
    """
    Numerical quadrature integration with trapezoid rule
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learn_a     = self.requires_grad
        self.kernel_train = self.requires_grad

        # Settings from DSS: hardcode to default
        self.use_initial = False
        self.trap_rule   = True
        self.learn_theta = False
        self.theta_scale = False
        self.learn_dt    = False
        self.dt_min      = 0.001
        self.dt_max      = 0.1
        self.constrain_a = True

        self.kernel_weights = None
        self.lr = None

        # Init weights
        self.init_weights()
        
    def init_weights(self):
        # Set skip connection
        if self.kernel_weights is not None:  
            # lr and wd as None means they're the same as model lr and wd
            self.register('k', self.kernel_weights, trainable=True, lr=None, wd=None)
        
        skip = torch.randn(self.n_heads)
        self.register('skip', skip, trainable=True, lr=None, wd=None)
        
        # Only learn half the elements due to complex conjugacy
        self.param_shape = (self.n_heads, self.d_kernel // 2)
        
        # For quadrature
        self.order2const = {1 : [], 2: [1/2], 3: [5/12,13/12], 4: [3/8,7/6,23/24]}
        
        # Initialize b, c, x[0]
        if self.use_initial:
            self.b  = nn.Parameter(torch.randn(*self.param_shape))
            self.c  = nn.Parameter(torch.randn(*self.param_shape))
            self.x0 = nn.Parameter(torch.randn(*self.param_shape))
        else:
            self.q  = nn.Parameter(torch.randn(*self.param_shape))
            
        # Initialize a and theta weights (Chebyshev initialization)
        h_scale  = torch.exp(torch.arange(self.n_heads) / self.n_heads * 
                             math.log(self.dt_max / self.dt_min))
        angles   = torch.arange(self.d_kernel // 2) * math.pi
        t_scale  = h_scale if self.theta_scale else torch.ones(self.n_heads)
        theta    = oe.contract('n, d -> nd', t_scale, angles)
        a        = -repeat(h_scale, 'nk -> nk kd', 
                           kd=self.d_kernel // 2).clone().contiguous()
        
        self.register("theta", theta, self.learn_theta, lr=self.lr, wd=None)
        self.register("a", a, self.learn_a, lr=self.lr, wd=None) 
        
        if self.learn_dt:
            log_T = torch.rand(self.n_heads) * (
                math.log(self.dt_max) - math.log(self.dt_min)
            ) + math.log(self.dt_min)
            self.register("log_T", log_T, True, lr=self.lr, wd=None)
            
            
    def numerical_quadrature(self, f, g, order=3):
        # From HazyResearch:
        # this is old C-numerical recipe here looks like
        # http://www.foo.be/docs-free/Numerical_Recipe_In_C/c4-1.pdf
        # int_a^b = T*[1/2*f_1  + f_2 + ... + f_{n-1} + 1/2*f_n]
        # order 3 = T*[5/12*f_1 + 12/13f_2 + f_3 + ... f_{n-2} + 12/13*f_{n-1} + 5/12*f_n]
        # order 4 = T*[3/8*f_1  + 7/6*f_2 + 23/24*f_3 ... + f_{n-3} + f_{n-2}*23/24 + 7/6*f_{n-1}+3/8*f_n]
        # These formulas differ are manipulated so that for all but the endpoints, it's just adding them up!
        # Compare with typical simpson's composite rule that requires multiplying intermediate values.
        #
        # BE WARNED. The encapsulation on this is terrible, we rely on orders of f and g -- and T is premultiplied into f
        # it needs a serious refactor and it caused pain.
        #
        # Order here refers to the error term being of order say O(N^{-3}) for order 3
        #
        y = self.fft_conv(g, f)
        # g.shape is batch x model_dim (= n_heads) x len
        
        # y[k] = T*sum_{j} g[k-j]f[j] = T*sum_{j} h_k[j]
        # NB: F is pre-multiplied with T?
        def _roll(h,j): return h[..., :-j] if j > 0 else h

        for i, c in enumerate(self.order2const[order]):
            # roughly what we want is:
            # y[i:] += T*(c-1)*(h[i] + h[:-i]) where -0 is understood to mean h itself, which is not python
            # so the indexing here is we want
            # g[k-i]f[i] which means we need to shift g up by i positions.
            # term = _roll(g,i)*f[...,i] + g[...,i]*_roll(f,i)
            term  = oe.contract('h, b h l -> b h l', f[..., i], _roll(g, i)) 
            term += oe.contract('h l, b h -> b h l', _roll(f, i), g[..., i])
            #y[...,i:] += T*(c-1)*term
            y[..., i:] += (c - 1) * term # Note: f is premultiplied with T.
        return y
        
    def fft_conv(self, u_input: torch.tensor, v_kernel: torch.tensor, 
                 L: int=None):
        # Convolve u with v in O(n log n) time with FFT (n = len(u))
        L   = u_input.shape[-1] if L is None else L # Assume u is input
        u_f = torch.fft.rfft(u_input, n=2*L) # (B H L)
        v_f = torch.fft.rfft(v_kernel[:, :L], n=2*L) # (H L)

        y_f = oe.contract('b h l, h l -> b h l', u_f, v_f) 
        y   = torch.fft.irfft(y_f, n=2*L)[..., :L]  # (B H L)
        return y
        
    def set_weights(self, name, weights, trainable, lr=None, wd=None):
        w = getattr(self, name)
        assert w.shape == weights.shape
        self.register(name, weights, trainable, lr, wd)
    
    def get_kernel(self, u, l=None):
        # Input u should be (B, n_kernels, L)
        l = u.size(-1) if l is None else l
        # Step size
        if self.learn_dt:
            T = torch.exp(self.log_T).view(-1, 1, 1)
        else:
            T = 1 / (l - 1)  
            
        zk = T * torch.arange(l, device=u.device).view(1, -1, 1)  # zk.shape is (1, L, 1)
        _a = -self.a.abs() if self.constrain_a else self.a        # _a.shape is (nk, kd // 2)
        base_term = (2 * T * torch.exp(_a.unsqueeze(1) * zk) * 
                     torch.cos(self.theta.unsqueeze(1) * zk))     # base_term.shape is (nk, L, kd // 2)
        q  = self.b * self.c if self.use_initial else self.q      # q.shape is (nk, kd // 2)
        f  = (q.unsqueeze(1) * base_term).sum(-1)                 # f.shape is (nk, L)
        return f
    
    def forward(self, u, l=None):
        # For the sake of usage in SSDLayer, assume u is input as B X D (=H) x L
        # u = rearrange(u, 'b l d -> b d l')  # Assume u is B x L x D
        k = self.get_kernel(u, l)
        y = self.numerical_quadrature(k, u, order=2 if self.trap_rule else 1)
        
        # Add in the skip connection with per-channel D matrix
        if self.skip_connection:
            y = y + oe.contract('b h l, h -> b h l', u, self.skip)
        # Add back the initial state
        if self.use_initial:  
            y = y + (2 * (self.c * self.x0).unsqueeze(2) * base_term).sum(-1)
        # Again, for sake of usage in SSDLayer, assume output is B X D X L
        return y, None
        # return rearrange(y, 'b d l -> b l d')
