#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:15:52 2024

@author: brennan
"""
import sys
sys.path.append("..")
sys.path.append("../..")

import time
import torch

from torch import autograd
from dmsr.wgan import DMSRDensityCritic, DMSRCritic
from dmsr.data_tools import generate_mock_data


batch_size        = 3
density_size      = 2 * 32 + 4
displacement_size = 32
density_channels  = 16
main_channels     = 16
style_size        = 1

input_size = displacement_size
input_channels = 6
base_channels = main_channels

disp = torch.randn(
    (batch_size, 6, displacement_size, displacement_size, displacement_size)
).requires_grad_(True)

dens = torch.randn(
    (batch_size, 2, density_size, density_size, density_size)
).requires_grad_(True)

style = torch.randn((batch_size, 1))


#%% Critic Model
critic = DMSRCritic(
    input_size, input_channels, base_channels, 2
)

initial_time = time.time()
score = critic(disp)
final_time = time.time()

print("Critic Model:")
print("Score for fake data is", score)
print("Prediction took", final_time-initial_time, "seconds")


#%% Density Critic Model
density_critic = DMSRDensityCritic(
    density_size, displacement_size, density_channels, main_channels
)

initial_time = time.time()
score = density_critic(disp, dens)
final_time = time.time()

print("Density Critic Model:")
print("Score for fake data is", score)
print("Prediction took", final_time-initial_time, "seconds")


#%% Styled Critic Model
styled_critic = DMSRCritic(
    input_size, input_channels, base_channels, style_size=style_size
)

initial_time = time.time()
score = styled_critic(disp, style)
final_time = time.time()

print("Styled Critic Model:")
print("Score for fake data is", score)
print("Prediction took", final_time-initial_time, "seconds")


#%% Styled Density Critic Model
styled_density_critic = DMSRDensityCritic(
    density_size, 
    displacement_size, 
    density_channels, 
    main_channels, 
    style_size
)

initial_time = time.time()
score = styled_density_critic(disp, dens, style)
final_time = time.time()

print("Styled Density Critic Model:")
print("Score for fake data is", score)
print("Prediction took", final_time-initial_time, "seconds")


#%% Gradient Penalty

score = score.sum()
displacement_grad, density_grad = autograd.grad(
    score,
    (disp, dens),
    retain_graph=True,
    create_graph=True,
    only_inputs=True,
)

# Compute the gradient penalty term.
density_grad = density_grad.flatten(start_dim=1)
displacement_grad = displacement_grad.flatten(start_dim=1)
grad = torch.concat((displacement_grad, density_grad), dim=1)

weight = 10
penalty = weight * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
print("Gradient penalty:", penalty)