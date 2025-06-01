#!/usr/bin/env python
# coding: utf-8

"""
Plant Leaf Dataset Handling
-------------------------------------------------
Classes and utilities for loading, processing, and preparing plant leaf datasets
"""

import pickle
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    """
    Set random seed to ensure experiment reproducibility
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set: {seed}")


class LeafDataset(Dataset):
    """
    Plant Leaf Dataset Class
    """
    def __init__(self, features, labels, transform=None, orig_transform=None, class_mapping=None):

        self.features = features
        self.labels = labels
        self.transform = transform
        self.orig_transform = orig_transform
        
        # Label name mapping
        self.class_names = {
            0: 'Healthy',
            1: 'Diseased'
        }
        
        # Update class mapping (if provided)
        if class_mapping:
            self.class_names.update(class_mapping)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]
        
        # Handle variable image sizes and ensure RGB format
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Single channel
            image = np.repeat(image, 3, axis=2)


        if image.dtype != np.uint8:
            if image.max() <= 1.0:  # If normalized to [0,1]
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to PIL image for applying transformations
        image = Image.fromarray(image)
        
        # Save original image (may apply some basic transformations)
        orig_image = image
        if self.orig_transform:
            orig_image = self.orig_transform(image)
        
        # Apply preprocessing transformations
        if self.transform:
            image = self.transform(image)
        
        return (image, orig_image, label) if self.orig_transform else (image, label)
    
    def get_class_name(self, label):
        """
        Get class name
        
        Args:
            label (int): Class label
            
        Returns:
            str: Class name
        """
        return self.class_names.get(label, f"Unknown ({label})")


class LeafProcessor:
    """
    Plant Leaf Data Processor: Responsible for data loading, filtering, analysis and visualization
    """
    def __init__(self, config=None):
        """
        Initialize data processor
        
        Args:
            config (dict, optional): Configuration parameters
        """
        # Default configuration
        self.config = {
            'balance_data': True,
            'max_samples_per_class': 1000,  # Increased for plant dataset
            'valid_classes': [0, 1],  # Binary: healthy and diseased
            'class_mapping': {0: 0, 1: 1},  # Binary mapping (will be updated by TensorFlow loader)
            'img_size': 224,  # Standard size for plant classification
            'mean': [0.485, 0.456, 0.406],  # ImageNet mean
            'std': [0.229, 0.224, 0.225],   # ImageNet std
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        }
        
        # Update configuration (if provided)
        if config:
            self.config.update(config)
        
        # Class names
        self.class_names = {
            0: 'Healthy',
            1: 'Diseased'
        }
        
        # Store loaded data
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

    
    def load_tensorflow_dataset(self, balance_data=None, max_samples_per_class=None):

        try:
            import tensorflow_datasets as tfds
            import tensorflow as tf
        except ImportError:
            raise ImportError("Please install tensorflow and tensorflow-datasets: pip install tensorflow tensorflow-datasets")
        
        if balance_data is None:
            balance_data = self.config['balance_data']
        if max_samples_per_class is None:
            max_samples_per_class = self.config['max_samples_per_class']
        
        print("Loading plant_leaves dataset from TensorFlow Datasets...")
        print("This will download ~6.6GB of data on first run...")
        
        # Load the dataset
        ds, ds_info = tfds.load(
            'plant_leaves:0.1.0',
            split='train',  # Only train split available
            with_info=True,
            as_supervised=False  # Get full feature dict
        )
        
        print(f"Dataset info: {ds_info}")
        print(f"Number of classes: {ds_info.features['label'].num_classes}")
        print(f"Total examples: {ds_info.splits['train'].num_examples}")
        
        # Get class names
        class_names = ds_info.features['label'].names
        print(f"Classes: {class_names}")
        
        # Create intelligent binary mapping based on class names
        binary_mapping = self._create_binary_mapping(class_names)
        
        # Convert TensorFlow dataset to numpy
        print("Converting TensorFlow dataset to numpy arrays...")
        all_images, all_labels = self._tf_to_numpy(ds, binary_mapping)
        
        print(f"Converted {len(all_images)} images")
        print(f"Binary class distribution: Healthy={np.sum(all_labels==0)}, Diseased={np.sum(all_labels==1)}")
        
        # Split the dataset
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_images, all_labels, 
            test_size=(self.config['val_split'] + self.config['test_split']), 
            random_state=42, 
            stratify=all_labels
        )
        
        # Split temp into validation and test
        val_size = self.config['val_split'] / (self.config['val_split'] + self.config['test_split'])
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=(1 - val_size), 
            random_state=42, 
            stratify=y_temp
        )
        
        print(f"Dataset split: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")
        
        # Balance training data if requested
        if balance_data and max_samples_per_class is not None:
            X_train, y_train = self._balance_binary_data(X_train, y_train, max_samples_per_class)
        
        # Store preprocessed data
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        
        print("Data loading complete.")
        return (X_train, y_train, X_valid, y_valid, X_test, y_test)

    def _create_binary_mapping(self, class_names):

        print("\nCreating binary classification mapping...")
        
        # Keywords that indicate healthy plants
        healthy_keywords = ['healthy', 'normal', 'good', 'fresh']
        
        # Keywords that indicate diseased plants
        disease_keywords = ['diseased', 'disease', 'sick', 'infected', 'pest', 'fungal', 
                           'bacterial', 'viral', 'blight', 'rust', 'spot', 'wilt', 'rot',
                           'leaf_spot', 'powdery_mildew', 'downy_mildew', 'canker',
                           'scab', 'mosaic', 'yellowing', 'browning', 'necrosis']
        
        mapping = {}
        
        for i, class_name in enumerate(class_names):
            class_name_lower = class_name.lower().replace('_', ' ')
            
            # Check if it contains healthy keywords
            is_healthy = any(keyword in class_name_lower for keyword in healthy_keywords)
            
            # Check if it contains disease keywords
            is_diseased = any(keyword in class_name_lower for keyword in disease_keywords)
            
            if is_healthy and not is_diseased:
                mapping[i] = 0  # Healthy
                print(f"Class {i:2d}: '{class_name}' -> Healthy")
            elif is_diseased:
                mapping[i] = 1  # Diseased
                print(f"Class {i:2d}: '{class_name}' -> Diseased")
            else:
                # Default heuristic: if name suggests a specific condition, it's probably diseased
                if any(word in class_name_lower for word in ['spot', 'burn', 'curl', 'yellow', 'brown']):
                    mapping[i] = 1  # Diseased
                    print(f"Class {i:2d}: '{class_name}' -> Diseased (heuristic)")
                else:
                    mapping[i] = 0  # Default to healthy if uncertain
                    print(f"Class {i:2d}: '{class_name}' -> Healthy (default)")
        
        # Update config with the actual mapping
        self.config['class_mapping'] = mapping
        return mapping

    def _tf_to_numpy(self, dataset, binary_mapping):
        """
        Convert TensorFlow dataset to numpy arrays with binary labels
        """
        import tensorflow as tf
        images = []
        labels = []
        print("Processing images one by one (streaming, low memory)...")
        count = 0
        target_size = (224, 224)  # Resize to standard size immediately

        for example in dataset:
            image = example['image'].numpy()
            image = tf.image.resize(image, target_size).numpy().astype(np.uint8)
            original_label = example['label'].numpy()
            binary_label = binary_mapping[original_label]
            images.append(image)
            labels.append(binary_label)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} images...")

        print(f"Completed processing {count} images")
        return np.array(images), np.array(labels)

    def _balance_binary_data(self, features, labels, max_samples_per_class):
        """
        Balance binary classification data
        
        Args:
            features (numpy.ndarray): Image features
            labels (numpy.ndarray): Binary labels (0=healthy, 1=diseased)
            max_samples_per_class (int): Maximum samples per class
            
        Returns:
            tuple: Balanced features and labels
        """
        healthy_indices = np.where(labels == 0)[0]
        diseased_indices = np.where(labels == 1)[0]
        
        # Limit each class to max_samples_per_class
        healthy_selected = np.random.choice(healthy_indices, 
                                          min(len(healthy_indices), max_samples_per_class), 
                                          replace=False)
        diseased_selected = np.random.choice(diseased_indices, 
                                           min(len(diseased_indices), max_samples_per_class), 
                                           replace=False)
        
        # Combine selected indices
        selected_indices = np.concatenate([healthy_selected, diseased_selected])
        np.random.shuffle(selected_indices)
        
        balanced_features = features[selected_indices]
        balanced_labels = labels[selected_indices]
        
        healthy_count = np.sum(balanced_labels == 0)
        diseased_count = np.sum(balanced_labels == 1)
        print(f"Balanced training data: Healthy={healthy_count}, Diseased={diseased_count}")
        
        return balanced_features, balanced_labels

    def load_data(self, training_file, validation_file=None, testing_file=None, balance_data=None, max_samples_per_class=None):

        print("Warning: Using legacy pickle file loading. Consider using load_tensorflow_dataset() for plant leaves.")
        
        if balance_data is None:
            balance_data = self.config['balance_data']
        if max_samples_per_class is None:
            max_samples_per_class = self.config['max_samples_per_class']
        
        print(f"Loading data files...")
        
        # Load training data
        try:
            with open(training_file, mode='rb') as f:
                train = pickle.load(f)
            X_train_raw, y_train_raw = train['features'], train['labels']
            
            # If validation and test files are not provided, automatically split the training set
            if validation_file is None or testing_file is None:
                # First split out the test set
                X_train_temp, X_test_raw, y_train_temp, y_test_raw = train_test_split(
                    X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw)
                
                # Split out validation set from remaining data
                X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = train_test_split(
                    X_train_temp, y_train_temp, test_size=0.25, random_state=42, stratify=y_train_temp)
                
                print(f"Data automatically split into training set ({len(X_train_raw)} samples), validation set ({len(X_valid_raw)} samples) and test set ({len(X_test_raw)} samples)")
            
            # If validation and test files are provided, load them
            else:
                if validation_file:
                    with open(validation_file, mode='rb') as f:
                        valid = pickle.load(f)
                    X_valid_raw, y_valid_raw = valid['features'], valid['labels']
                
                if testing_file:
                    with open(testing_file, mode='rb') as f:
                        test = pickle.load(f)
                    X_test_raw, y_test_raw = test['features'], test['labels']
        
        except Exception as e:
            print(f"Error loading data files: {e}")
            raise
        
        print("Data loading complete. Filtering and mapping data...")
        
        # Filter and map data
        X_filtered_train, y_filtered_train = self._filter_and_map_data(
            X_train_raw, y_train_raw, balance=balance_data, max_count=max_samples_per_class
        )
        
        X_filtered_valid, y_filtered_valid = self._filter_and_map_data(
            X_valid_raw, y_valid_raw, balance=False
        )
        
        X_filtered_test, y_filtered_test = self._filter_and_map_data(
            X_test_raw, y_test_raw, balance=False
        )
        
        # Store preprocessed data
        self.X_train = X_filtered_train
        self.y_train = y_filtered_train
        self.X_valid = X_filtered_valid
        self.y_valid = y_filtered_valid
        self.X_test = X_filtered_test
        self.y_test = y_filtered_test
        
        print("Data processing complete.")
        return (X_filtered_train, y_filtered_train, 
                X_filtered_valid, y_filtered_valid, 
                X_filtered_test, y_filtered_test)
   
    def inspect_tensorflow_dataset(self):

        try:
            import tensorflow_datasets as tfds
            import tensorflow as tf
        except ImportError:
            raise ImportError("Please install tensorflow and tensorflow-datasets: pip install tensorflow tensorflow-datasets")
        
        print("Inspecting plant_leaves dataset...")
        
        # Load dataset info only
        ds_info = tfds.builder('plant_leaves').info
        
        print(f"Number of classes: {ds_info.features['label'].num_classes}")
        print(f"Class names: {ds_info.features['label'].names}")
        print(f"Total examples: {ds_info.splits['train'].num_examples}")
        
        # Load a small sample to see actual data
        ds = tfds.load('plant_leaves', split='train[:20]', as_supervised=False)
        
        print("\nSample data:")
        for i, example in enumerate(ds.take(10)):
            filename = example['image/filename'].numpy().decode('utf-8')
            label_idx = example['label'].numpy()
            label_name = ds_info.features['label'].names[label_idx]
            print(f"{i}: {filename} -> Label {label_idx}: {label_name}")
        
        return ds_info.features['label'].names

    def _filter_and_map_data(self, features, labels, balance=False, max_count=None):

        valid_classes = self.config['valid_classes']
        class_mapping = self.config['class_mapping']
        
        filtered_features = []
        filtered_labels = []
        
        if balance and max_count is not None:
            count_per_class = [0] * len(valid_classes)
            
            # First count samples per class
            for idx in range(len(labels)):
                if labels[idx] in valid_classes:
                    mapped_label = class_mapping[labels[idx]]
                    if count_per_class[mapped_label] < max_count:
                        filtered_features.append(features[idx])
                        filtered_labels.append(mapped_label)
                        count_per_class[mapped_label] += 1
            
            print(f"Balanced class distribution: {count_per_class}")
        else:
            # Not balancing, just filter
            for idx in range(len(labels)):
                if labels[idx] in valid_classes:
                    filtered_features.append(features[idx])
                    filtered_labels.append(class_mapping[labels[idx]])
        
        return np.array(filtered_features), np.array(filtered_labels)
    
    def create_datasets(self, augment_train=True, include_original=False):
        """
        Create PyTorch dataset objects with optimized settings
        """
        if self.X_train is None or self.y_train is None:
            print("Error: Data not yet loaded. Please call load_data() method first.")
            return None, None, None
        
        print("Creating datasets...")
        img_size = self.config['img_size']
        mean = self.config['mean']
        std = self.config['std']
        
        # Define data transformations with simplified augmentation
        if augment_train:
            train_transform = transforms.Compose([
                # Basic transformations
                transforms.Resize((img_size, img_size)),
                # Simplified data augmentation
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),  # Reduced rotation
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced color jitter
                transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),  # Reduced crop range
                
                # Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            print("Training data augmentation enabled (simplified)")
        else:
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            print("Training data augmentation disabled")
        
        # For validation and test data, only basic processing
        eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Create datasets
        train_dataset = LeafDataset(
            self.X_train, self.y_train, 
            transform=train_transform, 
            orig_transform=None,  # Disabled original image storage
            class_mapping=self.class_names
        )
        
        valid_dataset = LeafDataset(
            self.X_valid, self.y_valid, 
            transform=eval_transform, 
            orig_transform=None,  # Disabled original image storage
            class_mapping=self.class_names
        ) if self.X_valid is not None else None
        
        test_dataset = LeafDataset(
            self.X_test, self.y_test, 
            transform=eval_transform, 
            orig_transform=None,  # Disabled original image storage
            class_mapping=self.class_names
        ) if self.X_test is not None else None
        
        print(f"Datasets created: {len(train_dataset)} training samples")
        if valid_dataset:
            print(f"{len(valid_dataset)} validation samples")
        if test_dataset:
            print(f"{len(test_dataset)} test samples")
        
        return train_dataset, valid_dataset, test_dataset

    def create_data_loaders(self, train_dataset, valid_dataset=None, test_dataset=None, batch_size=32, num_workers=4):
        """
        Create PyTorch data loaders with optimized settings
        """
        print(f"Creating data loaders (batch_size={batch_size}, workers={num_workers})...")
        
        # Optimize data loader settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True,  # Drop last incomplete batch
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Number of batches loaded in advance by each worker
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        valid_loader = None
        if valid_dataset:
            valid_loader = DataLoader(
                valid_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                drop_last=False,
                persistent_workers=True,
                prefetch_factor=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                drop_last=False,
                persistent_workers=True,
                prefetch_factor=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        print(f"Data loaders created: {len(train_loader)} training batches")
        if valid_loader:
            print(f"{len(valid_loader)} validation batches")
        if test_loader:
            print(f"{len(test_loader)} test batches")
        
        return train_loader, valid_loader, test_loader
    
    def save_processed_data(self, output_dir='./processed_data', filename_prefix='leaf'):
        """
        Save preprocessed data
        
        Args:
            output_dir (str): Output directory
            filename_prefix (str): Filename prefix
        """
        # Check if data is loaded
        if self.X_train is None or self.y_train is None:
            print("Error: Data not yet loaded. Please call load_data() method first.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        train_file = os.path.join(output_dir, f"{filename_prefix}_train.npz")
        np.savez(train_file, features=self.X_train, labels=self.y_train)
        
        if self.X_valid is not None and self.y_valid is not None:
            valid_file = os.path.join(output_dir, f"{filename_prefix}_valid.npz")
            np.savez(valid_file, features=self.X_valid, labels=self.y_valid)
        
        if self.X_test is not None and self.y_test is not None:
            test_file = os.path.join(output_dir, f"{filename_prefix}_test.npz")
            np.savez(test_file, features=self.X_test, labels=self.y_test)
        
        # Save configuration
        config_file = os.path.join(output_dir, f"{filename_prefix}_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        print(f"Preprocessed data saved to {output_dir}")
    
    def load_processed_data(self, data_dir, filename_prefix='leaf'):
        """
        Load preprocessed data
        
        Args:
            data_dir (str): Data directory
            filename_prefix (str): Filename prefix
            
        Returns:
            tuple: Preprocessed training, validation and test data and labels
        """
        train_file = os.path.join(data_dir, f"{filename_prefix}_train.npz")
        valid_file = os.path.join(data_dir, f"{filename_prefix}_valid.npz")
        test_file = os.path.join(data_dir, f"{filename_prefix}_test.npz")
        config_file = os.path.join(data_dir, f"{filename_prefix}_config.json")
        
        # Load training data
        try:
            train_data = np.load(train_file)
            self.X_train = train_data['features']
            self.y_train = train_data['labels']
            
            # Load validation data (if exists)
            if os.path.exists(valid_file):
                valid_data = np.load(valid_file)
                self.X_valid = valid_data['features']
                self.y_valid = valid_data['labels']
            
            # Load test data (if exists)
            if os.path.exists(test_file):
                test_data = np.load(test_file)
                self.X_test = test_data['features']
                self.y_test = test_data['labels']
            
            # Load configuration (if exists)
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            
            print(f"Preprocessed data loaded: {len(self.X_train)} training samples")
            if self.X_valid is not None:
                print(f"{len(self.X_valid)} validation samples")
            if self.X_test is not None:
                print(f"{len(self.X_test)} test samples")
            
            return (self.X_train, self.y_train, 
                    self.X_valid, self.y_valid, 
                    self.X_test, self.y_test)
        
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            raise