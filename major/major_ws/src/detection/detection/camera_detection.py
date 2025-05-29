import os
import sys

# Automatically activate the virtual environment if not already activated
venv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'venv')
activate_script = os.path.join(venv_path, 'bin', 'activate')
if not os.environ.get('VIRTUAL_ENV'):
    if os.path.exists(activate_script):
        print("Activating virtual environment...")
        activate_cmd = f"source {activate_script}"
        os.system(activate_cmd)
    else:
        print("Warning: Virtual environment not found at", venv_path)

from ultralytics import YOLO
import cv2
import time
import torch
import torchvision.transforms as transforms
from PIL import Image

# ResNet-like model definition
class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out

class LeafClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(LeafClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.linear = torch.nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.nn.functional.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def load_classification_model(model_path, num_classes):
    model = LeafClassifier(num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def classify_leaf(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return predicted.item(), confidence.item()

def main():
    print("Starting webcam detection...")
    
    # Check if model exists
    model_path = '../models/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please ensure your trained model is in the 'models' directory")
        return
    
    # Load classification model
    clf_model_path = '../models/best_model.pth'
    num_classes = 2  # Change if your model has a different number of classes
    clf_model = load_classification_model(clf_model_path, num_classes)

    # Open webcam first to make sure it works
    print("Initializing webcam...")
    cap = None
    for cam_index in range(3):
        temp_cap = cv2.VideoCapture(cam_index)
        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        temp_cap.set(cv2.CAP_PROP_FPS, 30)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                cap = temp_cap
                print(f"Webcam initialized successfully at index {cam_index}! Frame size: {frame.shape}")
                break
            temp_cap.release()
    if cap is None:
        print("Error: Could not open any webcam (tried indices 0, 1, 2)!")
        print("- Make sure your webcam is connected and not used by another application.")
        print("- On macOS, check System Preferences > Security & Privacy > Camera and ensure Python has access.")
        print("- Try rebooting your computer if the issue persists.")
        return
    
    # Load the leaf detection model
    print("Loading YOLO model...")
    try:
        if hasattr(torch.serialization, 'add_safe_globals'):
            from ultralytics.nn.tasks import DetectionModel
            from torch.nn.modules.container import Sequential
            from ultralytics.nn.modules import Conv
            torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("Model loaded successfully!")
        print(f"Model classes: {model.names}")
        print(f"Number of classes: {len(model.names)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the model path is correct")
        print("2. Make sure the model was trained with a compatible YOLO version")
        print("3. Try updating ultralytics: pip install --upgrade ultralytics")
        return

    # Set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45   # NMS IoU threshold
    model.max_det = 1000  # maximum number of detections per image

    print("\nStarting detection loop...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame!")
            break
        
        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps_end_time = time.time()
            fps = fps_frame_count / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0
            
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, verbose=False)
        result = results[0]
        
        # Count detections
        num_detections = len(result.boxes) if result.boxes is not None else 0
        
        # Draw bounding boxes and labels
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Ensure CPU numpy array
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                conf = float(box.conf[0])
                if conf < 0.75:
                    continue  # Skip detections below 75% confidence
                cls = int(box.cls[0])
                label = f"{result.names[cls]} {conf:.2f}"
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                print(f"Leaf center: ({center_x}, {center_y})")
                
                # Place a yellow dot at the center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                
                # Draw box with leaf-appropriate color (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with background for better visibility
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Display info on frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {num_detections}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Leaf Detection", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"leaf_detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main()