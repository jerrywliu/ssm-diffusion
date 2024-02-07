import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn import datasets as sk_datasets


# Make a dataloader which generates these trajectories

class TwoMoonsClassic(torch.utils.data.Dataset):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, seq_len=50, n_moons=4, batch_size=256, device = "cuda:0", noise_schedule = "exp"):#, transform=transforms.Compose([torch.ToTensor()])):
        self.seq_len = seq_len
        self.noise_schedule = noise_schedule
        self.rho = 1
        if self.noise_schedule == "karras":
            self.sigma_min = 0.002
            self.sigma_max = 10
            self.t = np.power(np.power(self.sigma_max, 1/self.rho) + np.linspace(0,1,self.seq_len)*(np.power(self.sigma_min, 1/self.rho) - np.power(self.sigma_max, 1/self.rho)), self.rho)
            # TODO: d_sigma not implemented for karras
        elif self.noise_schedule == "exp":
            self.sigma_min = 1e-2
            self.sigma_max = 50
            self.t = self.sigma_max * np.exp(np.linspace(0,1,self.seq_len)*np.log(self.sigma_min/self.sigma_max))
            self.d_sigma = np.log(self.sigma_min/self.sigma_max)/self.seq_len
            
        self.N = 1000 # size of dataset, chosen arbitrarily
        self.moons_size = n_moons
        self.batch_size = batch_size
        self.dim = 2 # moons are 2D
        self.moons = torch.tensor(sk_datasets.make_moons(self.moons_size)[0]) # data only
        self.seed_is_set = False
        self.device = device

        print(f"Dataset length: {self.__len__()}")

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    # Simulate trajectories from the true probability flow ODE
    def __getitem__(self, index):
        self.set_seed(index)
        
        trajectory = []
        
        data_indices = np.random.choice(self.moons_size, self.batch_size, replace=True)
        ground_truth = self.moons[data_indices, :] # (batch_size x dim)
        noise = torch.randn((self.batch_size, self.dim))
        noisy = ground_truth + self.t[0]*noise # (batch_size x dim)
        trajectory.append(noisy.clone())
        
        for i in range(self.seq_len-1):
            # Compute d_noisy
            distances = torch.pow(torch.norm(noisy[:, None, :] - self.moons[None, :, :], dim=2), 2) # (batch_size x num_moons)
            u = distances[:, :]/(-2*self.t[i]**2) # batch x num_moons
            u_max = torch.max(u, 1).values # batch
            v = torch.exp(u - u_max[:, None]) # batch x num_moons
            v = v/torch.sum(v, 1)[:, None] # batch x num_moons
            d_noisy = self.d_sigma * (noisy - oe.contract("b n, n d -> b d", v, self.moons)) # (batch_size x dim)
            
            noisy += d_noisy
            trajectory.append(noisy.clone())
            
        return torch.stack(trajectory).swapaxes(0,1).float().to(self.device) # (seq_len x batch_size x dim)
    