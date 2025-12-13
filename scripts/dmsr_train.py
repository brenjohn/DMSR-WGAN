#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:52:29 2025

@author: brennan
"""

# Uncomment if dmsr is not installed
# import sys
# sys.path.append("../../src/")

import shutil
import tomllib
import argparse
from pathlib import Path

from dmsr.training_setup import setup_environment, clean_up_environment
from dmsr.training_setup import setup_wgan, setup_dataloader, setup_monitors


def main(parameter_file):
    """
    Orchestrates the training lifecycle of a DMSR-WGAN model.

    This function sets up the training environment (DDP or single-GPU),
    loads configuration, initializes or loads the model, prepares the
    data pipeline, sets up monitoring, and executes the training loop.
    It ensures the environment is properly cleaned up upon completion or error.

    :param parameter_file: Path to the TOML file containing all training 
                           and model configuration parameters.
    :type parameter_file: pathlib.Path
    """
    # Initialise environment. 
    is_distributed, is_main_process, device = setup_environment()
    
    try:
        # Read parameter file and create output directory.
        with open(parameter_file, 'rb') as file:
            params = tomllib.load(file)
        
        output_dir = Path(params["output_dir"])
        if is_main_process:
            output_dir.mkdir(exist_ok=True)
        shutil.copyfile(parameter_file, output_dir / 'used_parameters.toml')
        
        # Create a DMSR-WGAN model.
        gan = setup_wgan(params, device, is_distributed)
    
        # Setup training data.
        dataloader, summary_stats, metadata = setup_dataloader(
            params, output_dir, is_main_process, is_distributed
        )    
        gan.set_dataset(
            dataloader, 
            params["batch_size"], 
            metadata['box_size'] / summary_stats['HR_Coordinates_std']
        )
        
        # Setup monitor objects to monitor training.
        setup_monitors(
            params, 
            gan, 
            summary_stats, 
            metadata, 
            is_main_process, 
            output_dir, 
            device
        )
    
        # Train the model.
        if 'supervised_epochs' in params:
            gan.train(
                params['supervised_epochs'], 
                train_step = gan.generator_supervised_step
            )
            gan.train(
                params['supervised_epochs'], 
                train_step = gan.critic_supervised_step
            )
        
        gan.train(params["num_epochs"])
        
    finally:
        clean_up_environment(is_distributed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a DMSR-WGAN model."
    )
    
    parser.add_argument(
        '--parameter_file', 
        type=Path, 
        default='./training_velocity_critic.toml',
        help="Path to the parameter file"
    )
    
    args = parser.parse_args()
    main(args.parameter_file)