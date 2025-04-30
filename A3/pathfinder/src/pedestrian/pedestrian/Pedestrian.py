# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2
import json

import torch
from torchvision import transforms
from PIL import Image
from pedestrian.network import ResNet18

import os
from ament_index_python.packages import get_package_share_directory


from collections import Counter



class Pedestrian(Node):

    def __init__(self):

        # ROS Node stuff
        super().__init__('Pedestrian')
        # self.publisher_ = self.create_publisher(String, 'Sign', 10)
        self.AvgPub = self.create_publisher(String, 'ModeSign', 10)
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0

        self.ImgSub = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.ImgSub_callback, 10)
        self.processed_img_pub = self.create_publisher(CompressedImage, '/pedestrian/processed_image', 10)

        ############ Pedestrian stuff ######################################

        # Device selection (CUDA if available, otherwise CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = ResNet18()
        self.model = self.model.to(self.device)



        # self.ModelPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models', 'best_model_final.pth'))
        self.ModelPath = os.path.join( get_package_share_directory('pedestrian'), 'Models', 'best_model_final.pth' )


        # Load the model weights
        checkpoint = torch.load(self.ModelPath, map_location=self.device)
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

        self.SignRecord = []
        self.RecentStopXY = []
        self.RecentRightXY = []
        self.RecentLeftXY = []

        self.CurrOutput = None
        self.OutputCentre = [None, None]


        # # Initialize the window once
        # cv2.namedWindow('Compressed Image', cv2.WINDOW_NORMAL)
        # cv2.moveWindow('Compressed Image', 0, 0)  # Set initial position


    def ImgSub_callback(self, msg):

        self.get_logger().info('Image Received')

        ImgArray = np.frombuffer(msg.data, np.uint8)
        Image = cv2.imdecode(ImgArray, cv2.IMREAD_COLOR)

        flipped_image = cv2.flip(Image, -1)

        CroppedSign, center_x, center_y = self.Crop(flipped_image)
        Output, center_x, center_y = self.Identify(CroppedSign, center_x, center_y)

        data1 = [Output, center_x, center_y]

        self.get_logger().info(f'Identify() outputted: {data1}')

        # msg2 = String()
        # msg2.data = "responding " + " ".join(str(item) for item in data1)
        # self.publisher_.publish(msg2)

        
        match data1[0]:
            case 'Stop':
                self.RecentStopXY = [data1[1], data1[2]]

            case 'Turn right':
                self.RecentRightXY = [data1[1], data1[2]]

            case 'Turn left':
                self.RecentLeftXY = [data1[1], data1[2]]


        if (len(self.SignRecord) <= 50):
            self.SignRecord.append(data1[0])

        else:
            self.SignRecord.pop(0)
            self.SignRecord.append(data1[0])

        count = Counter(self.SignRecord)

        most_common = count.most_common(1)
        if most_common:
            Mode = most_common[0][0]
        else:
            Mode = "Unknown"  # or some default like 'Unknown'

        if Mode != self.CurrOutput:

            match Mode:
                case 'Stop':
                    
                    new_msg = String()
                    new_msg.data = " Stop " + " ".join(str(item) for item in self.RecentStopXY)
                    self.CurrOutput = Mode
                    self.AvgPub.publish(new_msg)

                case 'Turn right':
                    new_msg = String()
                    new_msg.data = " TurnRight " + " ".join(str(item) for item in self.RecentRightXY)
                    self.CurrOutput = Mode
                    self.AvgPub.publish(new_msg)

                case 'Turn left':
                    new_msg = String()
                    new_msg.data = " TurnLeft " + " ".join(str(item) for item in self.RecentLeftXY)
                    self.CurrOutput = Mode
                    self.AvgPub.publish(new_msg)

                case 'x':
                    self.get_logger().info('Mode is nothing')
                case _:
                    self.get_logger().info('Mode Error')

             
            # self.CurrOutput = Mode

            # self.AvgPub.publish(new_msg)


    def Crop(self, image):
        '''
        Extracts the images of the signs from the input image and saves them in directory
        defined by SaveDirectory

        This code is optimised to work with rosbags, or in other words it requires that input
        image be contained within a folder within the SaveDirectory, the name of the folder 
        being passed into the function as rosbag_name

        Input:
            ImageName (string):     Name of the image

            rosbag_name(string):    Name of the folder that the image is contained within

        Output:
            ImagePaths (list)(string):  A list of strings containing the file paths of the
                                        cropped images ( the images of the individual signs )
        '''



        # Resize image for further processing
        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[1]/2)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        inverted = cv2.bitwise_not(blurred)
        edges = cv2.Canny(inverted, 35, 150)
    
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center_x = 0
        center_y = 0

        # Loop over the contours and crop the traffic signs
        cropped_sign = None
        sign_count = 0
        sign_boxes = []
        centers = []
        
        for contour in contours:
            # Approximate the contour to a polygon and get bounding box
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour is large enough to be a traffic sign
            if 1000 < cv2.contourArea(contour) < 10000:  
                x, y, w, h = cv2.boundingRect(approx)
                
                # Value for the center point of each of the signs
                center_x = x + w // 2
                center_y = y + h // 2
                # print("X:",center_x,"Y:",center_y)
                # centers.append([center_x, center_y])

                # Crop the detected traffic sign
                cropped_sign = image[y:y+h, x:x+w]
                
                sign_count += 1
                
                # Store the location of the sign (adjusting for the resize)
                sign_boxes.append((x*2, y*2, w*2, h*2))

        return cropped_sign, round(center_x, 2), round(center_y, 2)

    def display_processed_image(self, image_key):
        """Publish the original image with cylinders and traffic signs labeled to a ROS topic"""
        if image_key not in self.processed_results:
            self.get_logger().warn(f"No processed results found for {image_key}")
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
                center_x = x + w // 2
                center_y = y + h // 2
                
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Add classification label if available
                label = "Unknown"
                if 'sign_classifications' in result and i in result['sign_classifications']:
                    label = result['sign_classifications'][i]
                
                cv2.putText(img, f"Sign {i+1}: {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)  # Yellow dot at center
        
                # Or display the center coordinates
                cv2.putText(img, f"({center_x}, {center_y})", (center_x + 10, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Convert the image to a ROS CompressedImage message
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
        
        # Publish the processed image
        self.processed_img_pub.publish(msg)

    def Identify(self, cropped_sign, center_x, center_y):

        '''
        Passes in images to the ML and returns the classification of the image

        Input:
            ImagePath (string): The filepath to the target image

        Output:
            predict_label (string): The classification of the target image
                                    with reference to the Predestrian class'
                                    class names dictionary
        '''
        
        if not isinstance(cropped_sign, np.ndarray):
            return ["x", center_x, center_y]

        # Convert RGBA to RGB if necessary
        img = Image.fromarray(cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2RGB))  # Convert OpenCV to PIL

        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = self.preprocess(img)  # Apply preprocessing
        img = img.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = self.model(img.to(self.device))

        _, predicted_class = torch.max(output, 1)  # Get the predicted class index

        predicted_label = self.class_names[predicted_class.item()]  # Map index to class label
            
        
        return [predicted_label, center_x, center_y]
    

def main(args=None):
    rclpy.init(args=args)

    pedestrian = Pedestrian()

    rclpy.spin(pedestrian)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pedestrian.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
