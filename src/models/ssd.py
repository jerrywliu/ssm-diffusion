import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe
from einops import rearrange, repeat

from .ssm.companion import CompanionKernel
from .ssm.shift import ShiftKernel
from .ssm.diag import DiagKernel
from .encoder.repeat import RepeatEncoder
from .encoder.dense import DenseEncoder
from .encoder.convolution import ConvEncoder
from .encoder.identity import IdentityEncoder

class SSDLayer(nn.Module):
    def __init__(self, 
                 kernel: dict,
                 decoder: dict, 
                 skip_connection: bool,
                 closed_loop: bool):
        super().__init__()
        self.kernel_args     = kernel
        self.decoder_args    = decoder
        self.skip_connection = skip_connection
        self.closed_loop     = closed_loop
        
        self.kernel  = self.get_kernel()
        self.decoder = self.get_decoder()
        
        self.inference_only = False
        
        # Bug with shape matching
        # assert self.skip_connection is False
    
    # Kernel
    def get_kernel(self):
        if self.kernel_args['type'] == 'companion':
            kernel_class = CompanionKernel
        elif self.kernel_args['type'] == 'shift':
            kernel_class = ShiftKernel
        elif self.kernel_args['type'] == 'diag':
            kernel_class = DiagKernel
        else:
            raise NotImplementedError(f"Error: {self.kernel_args['type']} not implemented")
        return kernel_class(**self.kernel_args['kwargs'])
    
    # Decoder
    def get_decoder(self):
        if self.decoder_args['type'] == 'repeat':
            decoder_class = RepeatEncoder
        elif self.decoder_args['type'] == 'dense':
            decoder_class = DenseEncoder
        elif self.decoder_args['type'] == 'convolution':
            decoder_class = ConvEncoder
        elif self.decoder_args['type'] == 'identity':
            decoder_class = IdentityEncoder
        else:
            raise NotImplementedError(f"Error: {self.decoder_args['type']} not implemented")
        return decoder_class(**self.decoder_args['kwargs'])
        
    def forward(self, u):
        # Assume input shape is (B, L, H)
        v = rearrange(u, 'b l h -> b h l')
        try:
            y, *_ = self.kernel(v)
        except:
            y = self.kernel(v)  # could output multiple, so should modify this

        y = rearrange(y, 'b h l -> b l h')
        y = self.decoder(y)
        if self.skip_connection:
            y += u
        return y

class SSD(nn.Module):
    def __init__(self, config, input_config, output_config):
        super().__init__()
        self.encoder_config = config.encoder
        self.decoder_config = config.decoder
        self.rollout_config = config.rollout
        self.input_encoder_config = input_config
        self.input_decoder_config = output_config
        
        self.input_encoder, self.encoder, self.decoder, self.rollout, self.input_decoder = self.init_layers()

    def _init_encoder(self, encoder_args):
        if encoder_args['type'] == 'repeat':
            encoder_class = RepeatEncoder
        elif encoder_args['type'] == 'dense':
            encoder_class = DenseEncoder
        elif encoder_args['type'] == 'convolution':
            encoder_class = ConvEncoder
        elif encoder_args['type'] == 'identity':
            encoder_class = IdentityEncoder
        else:
            raise NotImplementedError(f"Error: {encoder_args['type']} not implemented")
        return encoder_class(**encoder_args['kwargs'])
        
    def init_layers(self):
        encoder = []
        decoder = []
        rollout = []
        
        for ix, layer_config in enumerate(self.encoder_config):
            encoder.append(SSDLayer(**layer_config.layer))
        for ix, layer_config in enumerate(self.decoder_config):
            decoder.append(SSDLayer(**layer_config.layer))
        for ix, layer_config in enumerate(self.rollout_config):
            rollout.append(SSDLayer(**layer_config.layer))
            
        encoder = nn.Sequential(*encoder)
        decoder = nn.Sequential(*decoder)
        rollout = nn.Sequential(*rollout)
        
        input_encoder = self._init_encoder(self.input_encoder_config)
        input_decoder = self._init_encoder(self.input_decoder_config)
        
        return input_encoder, encoder, decoder, rollout, input_decoder
        
    def set_inference_length(self, length: int):
        """
        Use during evaluation to rollout up to specified length
        """
        for ix in range(len(self.ssd_decoder)):
            self.rollout[ix].kernel.target_length = length
            
    def set_inference_only(self, mode: bool=False):
        """
        Use during evaluation to only go through rollout branch
        """
        self.inference_only = mode
        self.requires_grad  = not mode  # Not implemented
        
    def sample_noise(self, z, noise_schedule):
        # z is shape B x L x D
        noise = torch.randn_like(z)
        var   = repeat(noise_schedule, 'l -> (l r)', r=self.noise_stride)
        noise = oe.contract('b l d, l -> b l d', noise, var)
        return noise
        
    def compute_rollout(self, z):
        # Compute rollout with closed-loop SSM
        z = self.rollout(z)
        return z
        
    def forward(self, u):
        # u is shape B x L x D
        z = self.encoder(self.input_encoder(u))
        # Compute closed-loop rollout
        z_rollout = self.compute_rollout(z)
        # rollout is a prediction for future samples, so keep first input sample
        z_rollout = torch.cat([z[:, :1, :], z_rollout[:, :-1, :]], dim=1)
        y_rollout = self.input_decoder(self.decoder(z_rollout))
        
        if not self.inference_only:
            # During training, can also compute outputs from available inputs
            y = self.input_decoder(self.decoder(z))
        else:
            y = None
            
        return y_rollout, y, z_rollout, z
 
