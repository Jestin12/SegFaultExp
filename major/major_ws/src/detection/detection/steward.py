# from camera_detection import LeafClassifier, load_classification_model, preprocess_image, classify_leaf

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


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import termios
import tty
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from .leaf_classifier import load_classification_model, classify_leaf, get_classification_label


from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

import numpy as np
import json

from ament_index_python.packages import get_package_share_directory

import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import cv2
import time


class Steward(Node):
    def __init__(self):
        super().__init__("Steward")

        self.ImgCordPub = self.create_publisher(String, '/plant_detection', 10)

        self.ImgSub = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.ImgSub_callback, 10)

        # self.LeafClassifier = LeafClassifier()

        self.ProcessedImgPub = self.create_publisher(CompressedImage, '/processed_image/compressed', 10)


        # print("Starting webcam detection...")
    
        # Check if model exists
        self.model_path = '/home/neel/SegFaultExp/major/major_ws/src/detection/detection/models/best.pt'

        

        if not os.path.exists(self.model_path):
            print(f"Error: Model file '{self.model_path}' not found!")
            print("Please ensure your trained model is in the 'models' directory")
            return
    
        # Load classification model
        self.clf_model_path = '/home/neel/SegFaultExp/major/major_ws/src/detection/detection/models/best_model.pth'
        num_classes = 2  # Change if your model has a different number of classes
        self.clf_model = load_classification_model(self.clf_model_path, num_classes)


        # Load the leaf detection model
        print("Loading YOLO model...")
            
        try:
            if hasattr(torch.serialization, 'add_safe_globals'):
                from ultralytics.nn.tasks import DetectionModel
                from torch.nn.modules.container import Sequential
                from ultralytics.nn.modules import Conv
                torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
            print(f"Model classes: {self.model.names}")
            print(f"Number of classes: {len(self.model.names)}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Ensure the model path is correct")
            print("2. Make sure the model was trained with a compatible YOLO version")
            print("3. Try updating ultralytics: pip install --upgrade ultralytics")
            return
        
        # Set model parameters
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45   # NMS IoU threshold
        self.model.max_det = 1000  # maximum number of detections per image

        # print("\nStarting detection loop...")
        # print("Press 'q' to quit, 's' to save screenshot")
        
        # # FPS calculation
        # self.fps_start_time = time.time()
        # self.fps_frame_count = 0
        # self.fps = 0
        


    def ImgSub_callback(self, msg):

        np_arr = np.frombuffer(msg.data, np.uint8)
        ImgData = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        self.get_logger().info(f"Received image, format: {msg.format}, shape: {ImgData.shape}")

        # Convert BGR to RGB (YOLOv8 expects RGB)
        image_rgb = cv2.cvtColor(ImgData, cv2.COLOR_BGR2RGB)

        # # Calculate FPS
        # fps_frame_count += 1
        # if fps_frame_count >= 30:
        #     fps_end_time = time.time()
        #     fps = fps_frame_count / (fps_end_time - fps_start_time)
        #     fps_start_time = fps_end_time
        #     fps_frame_count = 0

        # Run YOLOv8 inference on the frame
        results = self.model.predict(image_rgb, verbose=False)
        result = results[0]
        
        # self.get_logger().info(f"R1")
        
        # Count detections
        num_detections = len(result.boxes) if result.boxes is not None else 0
        

        # Count detections
        num_detections = len(result.boxes) if result.boxes is not None else 0
        
        
        # self.get_logger().info(f"R2")
                               
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
                self.get_logger().info(f"prediciton: {label}")
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # print(f"Leaf center: ({center_x}, {center_y})")
                self.get_logger().info(f"Leaf center: ({center_x}, {center_y})\n\n")
                
                # Crop the detected leaf region
                leaf_crop = ImgData[y1:y2, x1:x2]  # or image_rgb
                
                # Get classification if model is available
                classification_label = "Unknown"
                if self.clf_model is not None and leaf_crop.size > 0:
                    pred, pred_conf = classify_leaf(leaf_crop, self.clf_model)
                    classification_label = get_classification_label(pred, pred_conf)
                
                self.get_logger().info(f"Leaf is: {classification_label}")
                
                # Place a yellow dot at the center point
                cv2.circle(image_rgb, (center_x, center_y), 5, (0, 255, 255), -1)
                
                # Draw box with leaf-appropriate color (green)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with background for better visibility
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image_rgb, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
                msg = String()
                msg.data = f"{classification_label} {center_x} {center_y}"
                self.ImgCordPub.publish(msg)
        # # Display info on frame
        # # cv2.putText(image_rgb, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(image_rgb, f"Detections: {num_detections}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # # Display the frame
        # cv2.imshow("Leaf Detection", image_rgb)
    # Add detection count to the image
        cv2.putText(image_rgb, f"Detections: {num_detections}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert RGB back to BGR for encoding (IMPORTANT!)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        try:
            # Encode image as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result_encode, encimg = cv2.imencode('.jpg', image_bgr, encode_param)
            
            if result_encode:
                # Create CompressedImage message
                processed_msg = CompressedImage()
                processed_msg.header.stamp = self.get_clock().now().to_msg()
                processed_msg.header.frame_id = "camera_frame"  # Adjust frame_id as needed
                processed_msg.format = "jpeg"
                processed_msg.data = encimg.tobytes()
                
                # Publish the processed image
                self.ProcessedImgPub.publish(processed_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error publishing processed image: {str(e)}")
        

def main(args=None):
    rclpy.init(args=args)
    
    node = Steward()
    
    
    # Use a MultiThreadedExecutor to handle the keyboard input thread
    # and ROS2 callbacks
    rclpy.spin(node)
    # Clean up
    node.is_running = False
    
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()