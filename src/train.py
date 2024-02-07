import os
import sys
import argparse

import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, SGD

import opt_einsum as oe
from einops import rearrange, repeat

from tqdm import tqdm
from omegaconf import OmegaConf


from src.dataloaders.two_moons import TwoMoonsClassic
from src.models.ssd import SSD


def run_epoch(model, autoencoder, train, epoch, dataloader, 
              optimizer, criterion, beta_weight_loss, device):
    
    model.zero_grad()
    pbar = tqdm(dataloader, leave=False)
    
    mean_loss = []
    closed_loss = []
    open_loss = []
    decoder_loss = []
    
    for ix, images in enumerate(pbar):
        #print("Batch id: ", ix)
        #inputs, outputs = images
        inputs = images
        b = inputs.shape[0]
        moons_input = inputs[0,:,:-1,:] #+ 0
#         moons_input[:,:,:2] += torch.randn_like(moons_input[:,:,:2]) * 0.05

        moons_output_1 = inputs[0,:,1:,:] # Next step
    
        sigma_data = 1.
       
        y_rollout, y, z_rollout, z = model(moons_input)
        
#         pred_c, pred_o = model(moons_input)
        #pred_c = pred_c[:,:,:2]
        #pred_o = pred_o[:,:,:2]
        
        #loss_c_1 = criterion(pred_c[:,:,:2] + moons_input[:,:,:2], moons_output_1) # *((t*t + sigma_data*sigma_data)/(t*t*sigma_data*sigma_data)) # Add loss weighting 
        #loss_c_1 = criterion(pred_c[:,:,:2], moons_output_1) 
        #loss_o = criterion(pred_o, moons_output)
        #loss_c_2 = criterion(pred_c[:,:,2:], moons_output_2)
        #loss = loss_c + loss_o
        # loss = loss_c_1 #+ loss_c_2
        
        loss_c = ((y_rollout[:, :, :] - moons_output_1[:, :, :])**2).mean() # Loss: outputs of full model vs. true trajectories
        loss_o = ((z_rollout - z)**2).mean() # Loss: train closed loop to match open loop
        loss_d = ((y[:, :, :] - moons_output_1[:, :, :])**2).mean() # Loss: outputs of encoder/decoder vs. true
        loss = loss_c + loss_o + loss_d
        
#         loss_c = (((pred_c[:,:,:2] - moons_output_1)**2) * (1. / (moons_input[:,:,2:]**0.1 + 0.25))).mean()
#         loss_o = (((pred_o[:,:,:2] - moons_output_1)**2) * (1. / (moons_input[:,:,2:]**0.1 + 0.25))).mean()
#         loss = loss_c + loss_o
        
#         loss_c = (((pred_c[:,:,:2] - moons_output_1)**2) * (1. / (moons_input[:,:,2:]**0.1 + 0.25))).mean()
#         loss_o = (((pred_o[:,:,:2] - moons_output_1)**2) * (1. / (moons_input[:,:,2:]**0.1 + 0.25))).mean()
#         loss = loss_c + loss_o
        
        # loss = criterion(pred, smmnist_noise)*((t*t + 0.5*0.5)/(t*t*0.5*0.5)) # Add loss weighting 
        #if ix % 10 == 0:
        #    print(ix, t, loss)
            
        if train:
            loss.backward()
            optimizer.step()
            model.zero_grad()
        
        pbar_desc = f'Batch: {ix}/{len(dataloader)} | {loss}'
        pbar.set_description(pbar_desc)
        mean_loss.append(loss.detach())
        closed_loss.append(loss_c.detach())
        open_loss.append(loss_o.detach())
        decoder_loss.append(loss_d.detach())
    mean_loss = torch.tensor(mean_loss)
    print(f"Epoch loss: {torch.mean(mean_loss)}, closed: {torch.mean(torch.tensor(closed_loss))}, open: {torch.mean(torch.tensor(open_loss))}, decoder: {torch.mean(torch.tensor(decoder_loss))}")
    return


def train(model, autoencoder, epoch, **kwargs):
    model.train()
    model.set_inference_only(False)
    return run_epoch(model, autoencoder, True, epoch, **kwargs)
    
    
def evaluate(model, autoencoder, epoch, **kwargs):
    model.eval()
    model.set_inference_only(mode=True)
    
    with torch.no_grad():
        return run_epoch(model, autoencoder, False, epoch, **kwargs)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='two_moons', help='Folder of config file')
    args = parser.parse_args()

    # Configs
    input_encoder_config = OmegaConf.load(f'conf/{args.config}/input_encoder.yaml')
    input_decoder_config = OmegaConf.load(f'conf/{args.config}/input_decoder.yaml')
    ssd_encoder_config = OmegaConf.load(f'conf/{args.config}/ssd_encoder.yaml')
    ssd_decoder_config = OmegaConf.load(f'conf/{args.config}/ssd_decoder.yaml')
    ssd_rollout_config = OmegaConf.load(f'conf/{args.config}/ssd_rollout.yaml')
    train_config = OmegaConf.load(f'conf/{args.config}/train.yaml')

    ssd_config = OmegaConf.merge(ssd_encoder_config, ssd_decoder_config, ssd_rollout_config)

    # Device
    train_config.device = ('cuda:0' 
                    if torch.cuda.is_available() and train_config.no_gpu is False
                    else 'cpu')
    device = torch.device(train_config.device)

    # Model
    model = SSD(ssd_config, input_encoder_config, input_decoder_config).to(device)
    train_loader = torch.utils.data.DataLoader(moon_dataset_train, batch_size=batch_size, 
                                           shuffle=True)
    eval_loader = torch.utils.data.DataLoader(moon_dataset_test, batch_size=batch_size, 
                                            shuffle=False)
    

    # seed_everything(config.seed)
    criterion = nn.MSELoss(reduction='mean')

    optim_config = {
        'lr': 1e-3,   # 1e-4
        'weight_decay': 0 # 5e-5 # 0
    }
    optimizer = Adam(model.parameters(), **optim_config)

    train_config = {
        'dataloader': train_loader,  # Fill-in / update below 
        # 'scheduler': scheduler,
        'optimizer': optimizer, 
        'criterion': criterion, 
        # 'criterion_weights': args.criterion_weights,  # [1., 1., 10., 10.], 
        'beta_weight_loss': True,
        'device': device,
    }

    val_config = {
        'dataloader': eval_loader,  # Fill-in / update below 
        # 'scheduler': scheduler,
        'optimizer': optimizer, 
        'criterion': criterion, 
        # 'criterion_weights': args.criterion_weights,  # [1., 1., 10., 10.], 
        'beta_weight_loss': True,
        'device': device,
    }

    train_config['optimizer'].lr = 0.01
    train_config['optimizer'].weight_decay = 0.

    # Train model
    epoch_pbar = tqdm(range(100))
    for epoch in epoch_pbar:
        if (epoch + 1) % 100 == 0:
            train(model, None, epoch, **train_config)
            # evaluate(model, None, epoch, **val_config)
            torch.save(model.state_dict(), "moons_ssm_ode_trajs.pt") # Without robust already have model saved.
        else:
            train(model, None, epoch, **train_config)
