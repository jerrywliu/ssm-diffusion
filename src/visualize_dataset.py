import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from src.dataloaders.two_moons import TwoMoonsClassic

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Visualize true data distribution
moon_dataset_train = TwoMoonsClassic(device = device)
moons_true = moon_dataset_train.moons.numpy()
print(f"Dataset shape: {moons_true.shape}")
plt.scatter(moons_true[:, 0], moons_true[:, 1])
plt.title("True dataset distribution")
plt.savefig("../figs/true_moons.png")

# Visualize true ODE trajectory
viz_trajectories = 10

for train_traj_i, train_traj in enumerate(moon_dataset_train):
    for data_i in range(train_traj.shape[0]):
        if data_i == viz_trajectories:
            break
        plt.scatter(train_traj[data_i, :, 0].cpu(), train_traj[data_i, :, 1].cpu())
    plt.title("Probability flow ODE trajectories")
    plt.savefig("../figs/ode_trajectories.png")
    break

# Visualize a single trajectory
viz_trajectories = 1

for train_traj_i, train_traj in enumerate(moon_dataset_train):
    for data_i in range(train_traj.shape[0]):
        if data_i == viz_trajectories:
                break
        plt.scatter(train_traj[data_i, :, 0].cpu(), train_traj[data_i, :, 1].cpu(), color="blue")
        plt.scatter(train_traj[data_i, -1:, 0].cpu(), train_traj[data_i, -1:, 1].cpu(), color="red")
    plt.title("Probability flow ODE trajectories")
    plt.savefig("../figs/ode_trajectories_single.png")
    break

# Visualize ODE endpoints
plt.scatter(train_traj[:, -1, 0].cpu(), train_traj[:, -1, 1].cpu())
plt.title("Samples from probability flow ODE")
plt.savefig("../figs/ode_samples.png")
