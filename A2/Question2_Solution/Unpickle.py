import pickle
import numpy as np

'''
******************************************* Unpickle.py ***************************************

Author:         Neel, Jestin

Description:    Analyses the valid.p file which contains the training data for the machine 
                learning model. Used to extract class labels and image sizes of the training
                images

Dependencies:   pickle      numpy

***********************************************************************************************
'''

# Load the .p file
p_file_path = 'valid.p'
with open(p_file_path, 'rb') as f:
    data = pickle.load(f)

# Extract class labels
labels = data['labels']
print("Dict keys:", data.keys())  # Print available keys in the dictionary
unique_labels = np.unique(labels)

# Define class mapping (replace these with actual class names)
class_mapping = {
    0: "Stop",
    1: "Turn right",
    2: "Turn left",
    3: "Ahead only",
    4: "Roundabout mandatory",
    # Add other class labels and their corresponding names here if needed
}

# If the dataset has more labels than defined in `class_mapping`, extend the mapping
# You can add more labels if necessary.
for label in unique_labels:
    if label not in class_mapping:
        class_mapping[label] = f"Class {label}"  # Default class name if not already in mapping

# Map labels to class names
class_names = [class_mapping.get(label, "Unknown") for label in labels]

# Extract image sizes
image_sizes = data['sizes']

# Display results
print("Unique class labels:", unique_labels)
print("Class names (first 5):", class_names[:5])  # Show the first 5 mapped class names
print("Image sizes (first 5):", image_sizes[:5])  # Show the first 5 image sizes

# Optionally, display the size of the first image
height, width = image_sizes[0]
print(f"First image size: {height}x{width}")
