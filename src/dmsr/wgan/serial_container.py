#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 14:11:59 2025

@author: brennan

Defined below is a SerialContainer class to be used by the DMSRWGAN class to
wrap the generator and critic models. It replaces the DistributedDataParallel
class in runs the are not distributed across multiple GPUs.
"""

class SerialContainer():
    
    def __init__(self, model):
        self.module = model
        
    def __call__(self, *args):
        return self.module(*args)
    
    def parameters(self):
        return self.module.parameters()