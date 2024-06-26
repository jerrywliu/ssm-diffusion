# replace the encoder / decoder with mamba
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
from .encoder.conv import ConvEncoder
from .encoder.identity import IdentityEncoder
from .mamba-main. import Mamba
from .ssd import SSDLayer, SSD


class mamba_encoder(nn.Module):
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

        encoder = Mamba(
    
    d_model=128, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        ).to("cuda")

        decoder = Mamba(

    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
        ).to("cuda")
        
        rollout = []
        for ix, layer_config in enumerate(self.rollout_config):
            rollout.append(SSDLayer(**layer_config.layer))       
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
        #print(z.shape)
        # Compute closed-loop rollout
        z_rollout = self.compute_rollout(z)
        # rollout is a prediction for future samples, so keep first input sample
        z_rollout = torch.cat([z[:, :1, :], z_rollout[:, :-1, :]], dim=1)
        
        y_rollout = self.input_decoder(self.decoder(z_rollout))
        
        
        # During training, can also compute outputs from available inputs
        #y = self.input_decoder(self.decoder(z))
        y = None
            
        return y_rollout, y, z_rollout, z
 
