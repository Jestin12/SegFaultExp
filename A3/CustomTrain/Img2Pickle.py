import os
from PIL import Image
from torchvision import transforms
import torch
import pickle

image_dir = "my_images"  # path to your image folder

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = []
label_map = {
    
}
label_index = 0

for class_name in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    if class_name not in label_map:
        label_map[class_name] = label_index
        label_index += 1

    label = label_map[class_name]

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            dataset.append((img_tensor, label))
        except Exception as e:
            print(f"Skipping {img_path}: {e}")