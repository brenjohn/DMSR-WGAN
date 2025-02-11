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
from dmsr.dmsr_gan import DMSRDensityCritic


batch_size = 3
density_size = 2 * 32 + 4
displacement_size = 32
density_channels = 16
main_channels = 16
critic = DMSRDensityCritic(density_size, displacement_size, density_channels, main_channels)

#%% Foward Pass
disp = torch.randn(
    (batch_size, 6, displacement_size, displacement_size, displacement_size)
).requires_grad_(True)

dens = torch.randn(
    (batch_size, 2, density_size, density_size, density_size)
).requires_grad_(True)

initial_time = time.time()
score = critic(disp, dens)
final_time = time.time()

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