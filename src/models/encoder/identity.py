import torch.nn as nn

class IdentityEncoder(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.initialize_layers()
        
    def initialize_layers(self):
        self.layers = nn.Identity()
        
    def forward(self, x):
        return self.layers(x)
    