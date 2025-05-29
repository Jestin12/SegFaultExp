# Leaf Detection and Classification System

This system combines YOLO for leaf detection with a custom classifier for plant leaf classification. It uses the Plant Leaves dataset from TensorFlow Datasets for training the classifier.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the dataset:
```bash
python src/prepare_dataset.py
```

3. Train the classifier:
```bash
python src/train_classifier.py
```

4. Run the detection and classification system:
```bash
python src/detect_and_classify.py
```

## Running the Camera Detection Script

1. Ensure you are in the correct directory:
```bash
cd Computer_Vision_Model/YOLO/src
```

2. Activate the virtual environment:
```bash
source ../venv/bin/activate
```

3. Run the camera detection script:
```bash
python camera_detection.py
```

4. To stop the script, press 'q' in the terminal.

## System Components

1. **Dataset Preparation** (`prepare_dataset.py`):
   - Loads the Plant Leaves dataset from TensorFlow Datasets
   - Organizes images into train and validation sets
   - Creates class directories

2. **Classifier Training** (`train_classifier.py`):
   - Uses EfficientNetB0 as the base model
   - Implements data augmentation
   - Saves the trained model and class indices

3. **Detection and Classification** (`detect_and_classify.py`):
   - Uses YOLOv8 for leaf detection
   - Uses the trained classifier for leaf classification
   - Displays real-time results with bounding boxes and labels

## Usage

1. Run the detection and classification script
2. Point your camera at leaves
3. The system will:
   - Detect leaves in the frame
   - Classify each detected leaf
   - Display bounding boxes and labels
4. Press 'q' to quit

## Notes

- The system uses YOLOv8 nano model for detection
- The classifier is based on EfficientNetB0
- Real-time processing may require a GPU for optimal performance 