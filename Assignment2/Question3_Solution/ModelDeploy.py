import json
import torch
from torchvision import transforms
from PIL import Image
from network import ResNet18
import os
import cv2
from pathlib import Path 
import numpy as np
import csv
from datetime import datetime

class Pedestrian:
    
    
    def __init__(self, ModelPath, SaveDirectory, InputDirectory):
        self.SaveDirectory = SaveDirectory
        self.InputDirectory = InputDirectory
        
        # For statistics tracking
        self.detection_stats = {}
        
        # Store original images and processed results for display at the end
        self.processed_results = {}
        
        # Device selection (CUDA if available, otherwise CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = ResNet18()
        self.model = self.model.to(self.device)

        # Load the model weights
        checkpoint = torch.load(ModelPath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])  # Assuming state_dict is saved under this key
        self.model.eval()  # Set the model to evaluation mode

        # Define preprocessing steps
        self.preprocess = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize the image to 32x32
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize based on ImageNet stats
        ])

        self.class_names = {
        0: 'Stop',
        1: 'Turn right',
        2: 'Turn left',
        3: 'Ahead only',
        4: 'Roundabout mandatory',
        }
        
        
    def detect_red_cylinders(self, image):
        # Convert the image to HSV (Hue, Saturation, Value) color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range of red color in HSV
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        
        # Mask the image to keep only the red colors
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # Now, define the second range of red
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        
        # Mask for the second red range
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # Combine both masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Bitwise-AND the red mask with the original image
        red_cylinders = cv2.bitwise_and(image, image, mask=red_mask)

        # Find contours of the red regions
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the detected red cylinders
        output_image = image.copy()
        detected_count = 0
        cylinder_boxes = []

        for contour in contours:
            # Approximate the contour to a polygon and get bounding box
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if cv2.contourArea(contour) > 25000:  # Consider larger contours (like the red cylinders)
                x, y, w, h = cv2.boundingRect(approx)
                detected_count += 1
                cylinder_boxes.append((x, y, w, h))

        return output_image, detected_count, cylinder_boxes


    def Crop(self, ImageName, rosbag_name):
        image_path = os.path.join(self.InputDirectory, rosbag_name, ImageName)

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Error: The file at {image_path} does not exist!")
            return []
        else:
            # Load the image
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error loading image at path: {image_path}")
                return []
            else:
                # Store the original image
                image_key = f"{rosbag_name}/{ImageName}"
                self.processed_results[image_key] = {
                    'original_image': image.copy()
                }
                
                # Detect red cylinders
                _, red_cylinder_count, cylinder_boxes = self.detect_red_cylinders(image)
                
                # Store cylinder detection results
                self.processed_results[image_key]['cylinder_boxes'] = cylinder_boxes
                
                # Store detection statistics for this image
                self.detection_stats[image_key] = {
                    "red_cylinders": red_cylinder_count,
                    "traffic_signs": 0  # Will be updated later
                }
                
                # Resize image for further processing
                image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[1]/2)))

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                inverted = cv2.bitwise_not(blurred)
                edges = cv2.Canny(inverted, 35, 150)
            
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Loop over the contours and crop the traffic signs
                cropped_signs = []
                sign_count = 0
                sign_boxes = []
                
                for contour in contours:
                    # Approximate the contour to a polygon and get bounding box
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # Check if the contour is large enough to be a traffic sign
                    if cv2.contourArea(contour) > 2000:  
                        x, y, w, h = cv2.boundingRect(approx)

                        # Crop the detected traffic sign
                        cropped_sign = image[y:y+h, x:x+w]
                        cropped_signs.append(cropped_sign)
                        sign_count += 1
                        
                        # Store the location of the sign (adjusting for the resize)
                        sign_boxes.append((x*2, y*2, w*2, h*2))

                # Store sign detection results
                self.processed_results[image_key]['sign_boxes'] = sign_boxes
                
                # Update the detection statistics
                self.detection_stats[image_key]["traffic_signs"] = sign_count
                
                # Create save directory if it doesn't exist
                save_directory = os.path.join(self.SaveDirectory, rosbag_name)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                ImagePaths = []

                # Save cropped signs
                for i, cropped_sign in enumerate(cropped_signs):
                    save_path = os.path.join(save_directory, f"cropped_{i+1}_{ImageName}")
                    cv2.imwrite(save_path, cropped_sign)
                    ImagePaths.append(os.path.join(rosbag_name, f"cropped_{i+1}_{ImageName}"))

                return ImagePaths


    def Identify(self, ImagePath):
        # Split the ImagePath to get the rosbag name and file name
        path_parts = ImagePath.split('/')
        if len(path_parts) > 1:
            rosbag_name = path_parts[0]
            file_name = path_parts[1]
        else:
            rosbag_name = ""
            file_name = ImagePath
            
        full_path = os.path.join(self.SaveDirectory, ImagePath)

        # Open the image file
        img = Image.open(full_path)
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = self.preprocess(img)  # Apply preprocessing
        img = img.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = self.model(img.to(self.device))

        _, predicted_class = torch.max(output, 1)  # Get the predicted class index
        predicted_label = self.class_names[predicted_class.item()]  # Map index to class label
        
        # Add the classification result to our stats
        image_key = f"{rosbag_name}/{file_name.replace('cropped_', '', 1).split('_', 1)[1]}"
        if image_key in self.detection_stats:
            if "classifications" not in self.detection_stats[image_key]:
                self.detection_stats[image_key]["classifications"] = []
            self.detection_stats[image_key]["classifications"].append(predicted_label)
            
            # Store the classification result with the sign boxes
            sign_idx = int(file_name.split('_')[1]) - 1  # Get the sign index from filename
            if 'sign_classifications' not in self.processed_results[image_key]:
                self.processed_results[image_key]['sign_classifications'] = {}
            self.processed_results[image_key]['sign_classifications'][sign_idx] = predicted_label
        
        return predicted_label


    def Classify(self, ImageName, rosbag_name):
        CroppedImages = self.Crop(ImageName, rosbag_name)
        
        classifications = []
        for image in CroppedImages:
            label = self.Identify(image)
            classifications.append(label)
            
        return classifications
    
    
    def display_processed_image(self, image_key):
        """Display the original image with cylinders and traffic signs labeled"""
        if image_key not in self.processed_results:
            print(f"No processed results found for {image_key}")
            return
        
        result = self.processed_results[image_key]
        img = result['original_image'].copy()
        
        # Draw bounding boxes for cylinders
        if 'cylinder_boxes' in result:
            for i, (x, y, w, h) in enumerate(result['cylinder_boxes']):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"Cylinder {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding boxes for signs with their classifications
        if 'sign_boxes' in result:
            for i, (x, y, w, h) in enumerate(result['sign_boxes']):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Add classification label if available
                label = "Unknown"
                if 'sign_classifications' in result and i in result['sign_classifications']:
                    label = result['sign_classifications'][i]
                
                cv2.putText(img, f"Sign {i+1}: {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the image
        cv2.imshow(f"Processed Image: {image_key}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    def save_detection_stats(self, output_file="detection_stats.csv"):
        """Save detection statistics to a CSV file"""
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['image_path', 'red_cylinders', 'traffic_signs', 'expected_signs','expected_cylinders', 'detected_correctly', 'classifications']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # print(self.detection_stats.items())

            correct_count = 0

            writer.writeheader()
            for image_path, stats in self.detection_stats.items():
                # We expect 2 signs per image
                expected_signs = 2

                # We expect 2 cylinders per image
                expected_cylinders = 2

                detected_correctly = stats['traffic_signs'] == expected_signs and stats['red_cylinders'] == expected_cylinders

                correct_count += 1 if detected_correctly else 0

                writer.writerow({
                    'image_path': image_path,
                    'red_cylinders': stats.get('red_cylinders', 0),
                    'traffic_signs': stats.get('traffic_signs', 0),
                    'expected_signs': expected_signs,
                    'expected_cylinders': expected_cylinders,
                    'detected_correctly': detected_correctly,
                    'classifications': ', '.join(stats.get('classifications', []))
                })
        
        print(f"Detection statistics saved to {output_file}")
        
        # Calculate and print summary statistics
        total_images = len(self.detection_stats)
        correct_signs_detections = sum(1 for stats in self.detection_stats.values() if stats.get('traffic_signs', 0) == 2)
        correct_cylinder_detections = sum(1 for stats in self.detection_stats.values() if stats.get('red_cylinders', 0) == 2)

        print(f"\nSummary Statistics:")
        print(f"Total images processed: {total_images}")
        print(f"Images with exactly 2 signs detected: {correct_signs_detections} ({correct_signs_detections/total_images*100:.1f}%)")
        print(f"Images with exactly 2 red_cylinders detected: {correct_cylinder_detections} ({correct_cylinder_detections/total_images*100:.1f}%)")
        print(f"Images with correct detections: {correct_count} ({(correct_count)/total_images*100:.1f}%)")


def process_all_rosbags(ModelPath, OutputDirectory, InputDirectory, display_first_image=True):
    detector = Pedestrian(ModelPath, OutputDirectory, InputDirectory)
    
    # Get all rosbag directories in the InputDirectory
    rosbag_dirs = [d for d in os.listdir(InputDirectory) if os.path.isdir(os.path.join(InputDirectory, d)) and "rosbag" in d]
    
    total_rosbags = len(rosbag_dirs)
    print(f"Found {total_rosbags} rosbag directories to process")
    
    first_image_key = None
    
    # Process each rosbag directory
    for i, rosbag_name in enumerate(rosbag_dirs):
        print(f"\nProcessing rosbag {i+1}/{total_rosbags}: {rosbag_name}")
        rosbag_path = os.path.join(InputDirectory, rosbag_name)
        
        # Get all image files in the rosbag directory
        image_files = [f for f in os.listdir(rosbag_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} images in {rosbag_name}")
        
        # Process each image
        for j, image_name in enumerate(image_files):
            print(f"  Processing image {j+1}/{len(image_files)}: {image_name}")
            detector.Classify(image_name, rosbag_name)
            
            # Store the first image key for display later
            if first_image_key is None:
                first_image_key = f"{rosbag_name}/{image_name}"
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    detector.save_detection_stats(f"Question3_Solution/detection_stats_{timestamp}.csv")
    
    # Display the first processed image with bounding boxes and labels
    if display_first_image and first_image_key is not None:
        print(f"\nDisplaying results for the first processed image: {first_image_key}")
        detector.display_processed_image(first_image_key)
    
    return detector.detection_stats


def main():
    InputDirectory = 'Question3_Solution/images/'  # Directory containing rosbag folders
    OutputDirectory = 'Question3_Solution/cropped_images/'  # Output directory for cropped signs
    ModelPath = "Question3_Solution/Models/best_model_final.pth"
    
    # Process all rosbags and get statistics, then display the first image
    stats = process_all_rosbags(ModelPath, OutputDirectory, InputDirectory, display_first_image=True)
    print("\nProcessing completed!")

    output = input("Should I print the results to the terminal? (Y) ")
    if output == "Y":
        TerminalOutput = json.dumps(stats, indent = 4)

        print(TerminalOutput)
    
    print("Code executed")

if __name__ == '__main__':
    main()