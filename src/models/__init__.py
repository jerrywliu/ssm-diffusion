import torch
import torch.nn as nn
import torch.nn.functional as F

import opt_einsum as oe

class OurModule(nn.Module):
    """
    Interface for Module that allows registering buffers/parameters 
    with configurable optimizer hyperparameter
    - from S4 code: https://github.com/HazyResearch/state-spaces/blob/main/src/models/s4/s4.py
    """
    def __init__(self): 
        super().__init__()

    def register(self, 
                 name: str, 
                 tensor: torch.FloatTensor,
                 trainable: bool=False, 
                 lr: float=None, 
                 wd: float=None):
        """
        Utility method: register a tensor as a buffer or trainable parameter
        """
        if trainable:
            try:
                self.register_parameter(name, nn.Parameter(tensor))
            except KeyError:
                delattr(self, name)
                self.register_parameter(name, nn.Parameter(tensor))
        else:
            
            try:
                self.register_buffer(name, tensor)
            except KeyError:
                delattr(self, name)
                self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None: optim["lr"] = lr
        if trainable and wd is not None: optim["weight_decay"] = wd
        if len(optim) > 0: setattr(getattr(self, name), "_optim", optim)
        