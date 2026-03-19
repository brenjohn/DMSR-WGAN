#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:02:05 2026

@author: brennan
"""

import argparse

from pathlib import Path
from swift_tools import DatasetBuilder


def main(args):
    """Main entry point for the patch creation script.
    
    Handles setup, configuration loading, metadata creation, and orchestrates 
    the parallel patch creation for both Low and High-Resolution data streams.
    """
    print('Reading dataset config.')
    dataset_builder = DatasetBuilder(args)
    
    print('Creating dataset.')
    dataset_builder.create_dataset()
    
    print('Saving metadata.')
    dataset_builder.save_metadata()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a dataset."
    )
    parser.add_argument(
        '--config_file',
        type=Path,
        default='config.toml',
        help="Path to the TOML configuration file."
    )
    parser.add_argument(
        '--num_procs', 
        type=int, 
        default=1,
        help="Number of parallel processes to use."
    )
    args = parser.parse_args()
    main(args)