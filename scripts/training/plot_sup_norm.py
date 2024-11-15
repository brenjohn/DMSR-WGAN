#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:06:03 2024

@author: brennan
"""

import numpy as np
import matplotlib.pyplot as plt

norms = np.load('./sup_norm.npz')

sup_norm = norms['sup_norm']

# window_size = 500
# moving_average = np.convolve(
#     critic_loss, np.ones(window_size)/window_size, mode='valid'
# )

plt.plot(
    sup_norm, 
    linewidth=0.5, alpha=0.5
)

# plt.plot(
#     critic_batches[window_size//2-1:-window_size//2], moving_average, 
#     color='black'
# )

# plt.ylim((-4e3, 2e3))
# plt.yscale('log')
plt.show()
plt.close()