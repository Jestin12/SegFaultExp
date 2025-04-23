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

import torch
from torchvision import transforms
from PIL import Image
from pedestrian.network import ResNet18

import os
from ament_index_python.packages import get_package_share_directory

class Pedestrian(Node):

    def __init__(self):

        # ROS Node stuff
        super().__init__('Pedestrian')
        self.publisher_ = self.create_publisher(String, 'Sign', 10)
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0

        self.ImgSub = self.create_subscription(CompressedImage, 'camera/image/compressed', self.ImgSub_callback, 10)


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


        # Initialize the window once
        cv2.namedWindow('Compressed Image', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Compressed Image', 0, 0)  # Set initial position


    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

    def ImgSub_callback(self):
        # Convert compressed image data to numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Decode image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is not None:
            cv2.imshow('Compressed Image', image)

            # Put Crop image function here vvvvvv
            cropped_signs = self.Crop(image)

            predict_labels = Identify(cropped_signs)

            msg = String()

            msg.data = " ".join(predict_labels)

            self.publisher_.publish(msg)

            cv2.waitKey(1)
        else:
            self.get_logger().warn('Failed to decode image')


    def Crop(self, ImageName, rosbag_name):
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

        return cropped_signs



    def Identify(self, cropped_signs):

        '''
        Passes in images to the ML and returns the classification of the image

        Input:
            ImagePath (string): The filepath to the target image

        Output:
            predict_label (string): The classification of the target image
                                    with reference to the Predestrian class'
                                    class names dictionary
        '''
        
        predict_labels = []

        for sign in cropped_signs:

            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            img = self.preprocess(img)  # Apply preprocessing
            img = img.unsqueeze(0)  # Add batch dimension

            # Perform inference
            with torch.no_grad():
                output = self.model(img.to(self.device))

            _, predicted_class = torch.max(output, 1)  # Get the predicted class index

            predicted_label.append( self.class_names[predicted_class.item()] ) # Map index to class label
            
        
        return predicted_labels
    

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
