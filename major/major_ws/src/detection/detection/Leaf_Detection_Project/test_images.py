#!/usr/bin/env python
# coding: utf-8

"""
Test Images Classification Script
-------------------------------------------------
Script to run test images through the trained model and display leaf classifications
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from network import ResNet18
import matplotlib.pyplot as plt

def load_model(model_path):
    """
    Load the trained model
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        model: Loaded and configured model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = ResNet18()
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model, device

def process_image(image_path, model, device):
    """
    Process a single image through the model
    
    Args:
        image_path (str): Path to the image
        model: Trained model
        device: Device to run inference on
        
    Returns:
        tuple: (predicted_class, confidence, image)
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Class names
    class_names = {0: 'Healthy', 1: 'Diseased'}
    
    return predicted.item(), confidence.item(), image

def main():
    # Model path
    model_path = 'checkpoint/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load model
    model, device = load_model(model_path)
    
    # Test images directory
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"Error: Test images directory not found at {test_dir}")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        pred_class, confidence, image = process_image(image_path, model, device)
        
        # Print results
        class_names = {0: 'Healthy', 1: 'Diseased'}
        print(f"\nImage: {image_file}")
        print(f"Prediction: {class_names[pred_class]}")
        print(f"Confidence: {confidence:.2%}")
        
        # Display image with prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Prediction: {class_names[pred_class]} ({confidence:.2%})")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main() 