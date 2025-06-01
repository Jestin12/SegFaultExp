#!/usr/bin/env python
# coding: utf-8

"""
Plant Leaf Classification Training Pipeline
-------------------------------------------------
Main script for training the plant leaf classification model with ResNet18
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse
from tqdm import tqdm
import numpy as np

# Import custom modules
from network import ResNet18
from dataset import LeafProcessor, set_seed
import vis_utils

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training function with mixed precision
def train(model, train_loader, optimizer, criterion, epoch, epochs, scaler):
    """
    Train model for one epoch with mixed precision
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if isinstance(inputs, list) and len(inputs) > 1:
            inputs = inputs[0]
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Regular backpropagation
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = train_loss / len(train_loader)
    acc = 100. * correct / total
    
    return avg_loss, acc

# Validation function with mixed precision
def validate(model, val_loader, criterion):
    """
    Validate model on validation set with mixed precision
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if isinstance(inputs, list) and len(inputs) > 1:
                inputs = inputs[0]
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    
    return avg_loss, acc

# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, acc, history, filepath):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer state
        scheduler: Learning rate scheduler state
        epoch (int): Current epoch
        acc (float): Validation accuracy
        history (tuple): Training history
        filepath (str): Path to save the checkpoint
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'acc': acc,
        'epoch': epoch,
        'train_history': history
    }
    
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    # Save an additional best_model.pth for easier loading
    best_filepath = os.path.dirname(filepath) + '/best_model.pth'
    torch.save(state, best_filepath)
    print(f"Best model saved to {best_filepath}")

# Main function
def main():
    """
    Main function for training the plant leaf classifier
    """
    parser = argparse.ArgumentParser(description='Plant Leaf ResNet18 Classification')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')  # Reduced learning rate
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')  # Back to 64 for faster epochs
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')  # Reduced epochs
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer (sgd, adam, rmsprop)')
    parser.add_argument('--scheduler', default='cosine', type=str, help='Learning rate scheduler (cosine, step, none)')
    parser.add_argument('--data_path', default=None, type=str, help='Data path')
    parser.add_argument('--use_processed', action='store_true', help='Use preprocessed data')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading threads')  # Reduced workers
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')  # Increased weight decay
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--step_size', default=30, type=int, help='Learning rate step size')
    parser.add_argument('--gamma', default=0.1, type=float, help='Learning rate decay rate')
    parser.add_argument('--use_tensorflow_dataset', action='store_true', help='Use TensorFlow plant_leaves dataset')
    parser.add_argument('--inspect_dataset', action='store_true', help='Inspect dataset before loading')
    parser.add_argument('--patience', default=15, type=int, help='Early stopping patience')  # Increased patience
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout rate')  # Added dropout
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize data processor with standard image size
    processor = LeafProcessor(config={
        'img_size': 32,  # Reduced from 96 to 32
        'mean': [0.485, 0.456, 0.406],  # ImageNet mean
        'std': [0.229, 0.224, 0.225],   # ImageNet std
        'balance_data': True,
        'max_samples_per_class': 1000  # Reduced from 2000 to 1000
    })
    
    # Initialize training history and state variables
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_acc = 0
    start_epoch = 0
    
    # Load data
    if args.use_tensorflow_dataset:
        try:
            if args.inspect_dataset:
                processor.inspect_tensorflow_dataset()
            
            processor.load_tensorflow_dataset()
            print("Successfully loaded TensorFlow plant_leaves dataset")
        except Exception as e:
            print(f"Error loading TensorFlow dataset: {e}")
            print("Please install: pip install tensorflow tensorflow-datasets")
            return
    elif args.use_processed and os.path.exists('./processed_data'):
        processor.load_processed_data('./processed_data')
    else:
        print("Error: Please specify a data loading method:")
        print("  --use_tensorflow_dataset  (for TensorFlow plant_leaves dataset)")
        print("  --use_processed  (for preprocessed data)")
        return
    
    # Create datasets with strong augmentation
    train_dataset, valid_dataset, test_dataset = processor.create_datasets(
        augment_train=True,  # Enable data augmentation
        include_original=False
    )
    
    # Create data loaders with optimized settings
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        train_dataset, valid_dataset, test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.workers
    )
    
    # Create directory to save checkpoints
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    # Initialize model with dropout
    model = ResNet18()
    # Add dropout to the model
    model.linear = nn.Sequential(
        nn.Dropout(args.dropout),
        nn.Linear(512 * 1, 2)  # 1 is the expansion for BasicBlock in ResNet18
    )
    model = model.to(device)
    
    # Data parallel (if multiple GPUs)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Enable cuDNN benchmarking
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Initialize mixed precision scaler
    scaler = None  # Disabled mixed precision since we're not using CUDA
    
    # Initialize checkpoint variable
    checkpoint = None
    
    # If resuming from checkpoint
    if args.resume:
        print('Resuming from checkpoint..')
        if not os.path.isdir('checkpoint'):
            print('Error: No checkpoint directory found!')
        else:
            opt_name = args.optimizer.lower()
            lr_str = f"_lr_{args.lr}" if args.lr != 0.001 else ""
            scheduler_str = "" if args.scheduler.lower() != "none" else "_no_scheduler"
            
            checkpoint_path = f'./checkpoint/ckpt_{opt_name}{lr_str}{scheduler_str}.pth'
            
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Specified checkpoint {checkpoint_path} not found, trying to load best_model.pth")
                checkpoint_path = './checkpoint/best_model.pth'
                
                if not os.path.exists(checkpoint_path):
                    print(f"Warning: best_model.pth not found, starting training from scratch")
                else:
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model'])
                    best_acc = checkpoint['acc']
                    start_epoch = checkpoint['epoch']
                    
                    if 'train_history' in checkpoint:
                        history = checkpoint['train_history']
                        train_loss_list, train_acc_list, val_loss_list, val_acc_list = history
                    
                    print(f"Resumed from checkpoint Epoch: {start_epoch}, Accuracy: {best_acc:.2f}%")
            else:
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']
                
                if 'train_history' in checkpoint:
                    history = checkpoint['train_history']
                    train_loss_list, train_acc_list, val_loss_list, val_acc_list = history
                
                print(f"Resumed from checkpoint Epoch: {start_epoch}, Accuracy: {best_acc:.2f}%")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Set optimizer with weight decay
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Learning rate scheduler with warmup
    if args.scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # First restart epoch
            T_mult=2,  # Multiply T_0 by this factor after each restart
            eta_min=1e-6  # Minimum learning rate
        )
    elif args.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None
    
    if checkpoint is not None and scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args.epochs, scaler)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        # Check if this is the best model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        # Save history
        history = (train_loss_list, train_acc_list, val_loss_list, val_acc_list)
        
        # Build checkpoint filename
        opt_name = args.optimizer.lower()
        lr_str = f"_lr_{args.lr}" if args.lr != 0.001 else ""
        scheduler_str = "" if args.scheduler.lower() != "none" else "_no_scheduler"
        filepath = f'./checkpoint/ckpt_{opt_name}{lr_str}{scheduler_str}.pth'
        
        # Save checkpoint
        if is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, history, filepath)
    
    # Test the best model
    print("\nTesting the best model...")
    
    best_model_path = './checkpoint/best_model.pth'
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print("Warning: Best model not found, using current model for testing")
    
    # Test
    test_results = vis_utils.plot_confusion_matrix(model, test_loader, processor.class_names, device, criterion)
    
    # Visualize results
    vis_utils.visualize_training_results(
        train_loss_list, train_acc_list, val_loss_list, val_acc_list, 
        best_acc, args.optimizer, args.lr, args.batch_size, args.scheduler
    )
    
    # Visualize predictions
    vis_utils.visualize_predictions(model, test_loader, processor.class_names, device)
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%, Test accuracy: {test_results['accuracy']:.2f}%")

if __name__ == '__main__':
    main()