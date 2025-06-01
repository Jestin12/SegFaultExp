#!/usr/bin/env python
# coding: utf-8

"""
Dataset Setup Script
-------------------------------------------------
Script to help set up the plant leaves dataset manually
"""

import os
import shutil
import zipfile
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds

def setup_dataset(dataset_path, output_dir):
    """
    Set up the plant leaves dataset
    
    Args:
        dataset_path (str): Path to the downloaded dataset zip file
        output_dir (str): Directory to extract the dataset to
    """
    print(f"Setting up dataset from {dataset_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print("Dataset setup complete!")
    print(f"Dataset is now available at: {output_dir}")
    print("\nYou can now run the training script with:")
    print("python train_final.py --use_tensorflow_dataset")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Set up plant leaves dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the downloaded dataset zip file')
    parser.add_argument('--output_dir', type=str, 
                      default=os.path.expanduser('~/tensorflow_datasets/plant_leaves/0.1.0'),
                      help='Directory to extract the dataset to')
    args = parser.parse_args()
    
    setup_dataset(args.dataset_path, args.output_dir) 