#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:06:39 2024

@author: brennan
"""

import numpy as np
import matplotlib.pyplot as plt

losses = np.load('./losses.npz')

critic_loss = losses['critic_loss']
critic_batches = losses['critic_batches']

window_size = 500
moving_average = np.convolve(
    critic_loss, np.ones(window_size)/window_size, mode='valid'
)

plt.plot(
    critic_batches, critic_loss, 
    linewidth=0.5, alpha=0.5
)

plt.plot(
    critic_batches[window_size//2-1:-window_size//2], moving_average, 
    color='black'
)

# plt.ylim((-4e3, 2e3))
# plt.yscale('log')
# plt.show()
# plt.close()



#%%
generator_loss = losses['generator_loss']
generator_batches = losses['generator_batches']

window_size = 500
moving_average = np.convolve(
    generator_loss, np.ones(window_size)/window_size, mode='valid'
)

# plt.plot(
#     generator_batches, generator_loss, 
#     linewidth=0.5, alpha=0.5
# )

plt.plot(
    generator_batches[window_size//2-1:-window_size//2], moving_average, 
    # color='black'
)
plt.show()
plt.close()