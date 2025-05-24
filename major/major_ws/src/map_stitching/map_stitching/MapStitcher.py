import rclpy
from rclpy.node import Node
import cv2 as cv
import os
import numpy as np
import shutil

from sensor_msgs.msg import CompressedImage

# *************************** MapStitcher.py ***************************************

# Filename:         MapStitcher.py
# Author:           Alfie

# Description:      This script creates a ROS2 node that subscribes to a compressed
#                   camera topic, saves the images to a folder in the home directory and stitches them together
#                   into a 2D map. The number of images to be stitched can be specified with the ROS parameter
#                   num_saved_imgs using the CLI or a launch file. 
#
# Dependencies:     numpy  os shutil rclpy cv2 

# ************************************************************************************

class MapStitcher(Node):
    def __init__(self):
        super().__init__('StitchMapper')
        self.ImageSub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.save_image,
            10
        )

        self.get_logger().info("Subscribed to /camera/image_raw/compressed")

        # Declare number of images to be saved
        self.declare_parameter('num_saved_imgs', 40)
        self.num_saved_imgs = self.get_parameter('num_saved_imgs').get_parameter_value().integer_value

        self.get_logger().info(f"Number of images to be stitched: {self.num_saved_imgs}.\n To change this parameter use --ros-args -p num_saved_imgs:=[INTEGER]\n")

        #Define home directory
        self.home_dir = os.environ.get("HOME")
        if self.home_dir is None:
            raise EnvironmentError("HOME environment variable is not set.")
        
        #Define save path and create directory for images
        self.save_path = os.path.join(self.home_dir, "stitch_map_images")
        try:
            os.mkdir(self.save_path)
            print(f"Directory '{self.save_path}' created successfully.")
            print("Wating for images...")
        except FileExistsError:
            print(f"Directory '{self.save_path}' already exists.")
            while True:
                delete = input(f"Would you like to overwrite {self.save_path}? (Y/N): ").strip().upper()
                if delete == 'Y':
                    shutil.rmtree(self.save_path)
                    os.mkdir(self.save_path)
                    print(f"Directory '{self.save_path}' overwritten successfully.")
                    print("Wating for images...")
                    break
                elif delete == 'N':
                    print("Attempting to stitch images in directory...")
                    self.stitch_image()
                else:
                    print(f"'{delete}' is an invalid option. Please enter 'Y' or 'N'.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{self.save_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        #Counter to track images
        self.counter = 0

    def save_image(self, msg):
        # Skip images after desired number of images are saved
        if self.counter >= self.num_saved_imgs:
            return

        # Check if image is empty
        if not msg.data:
            self.get_logger().warn("Received empty image data. Skipping...")
            return

        # Convert to OpenCV image
        ImgArray = np.frombuffer(msg.data, np.uint8)
        CVImage = cv.imdecode(ImgArray, cv.IMREAD_COLOR)

        # Check if decoding worked correctly
        if CVImage is None:
            self.get_logger().error("Failed to decode image. Skipping...")
            return

        # Save the image
        self.counter += 1
        filename = os.path.join(self.save_path, f"image_{self.counter}.jpg")
        cv.imwrite(filename, CVImage)
        self.get_logger().info(f"Saved image {filename}.")

        if self.counter == self.num_saved_imgs:
            self.stitch_image()

    def stitch_image(self):
        stitcher = cv.Stitcher_create()
        images = []

        if len(os.listdir(self.save_path)) >= 2:
            for filename in sorted(os.listdir(self.save_path)):
                if filename.endswith(".jpg") or filename.endswith(".png"):  
                    filepath = os.path.join(self.save_path, filename)
                    image = cv.imread(filepath)

                    if image is None:
                        print(f"Failed to load image: {filepath}")
                        continue
                    else:
                        images.append(image)
        else:
            print("Not enough images to perform stitch. Exiting...")
            exit(0)

        (status,stitched) = stitcher.stitch(images=images)

        if status == cv.Stitcher_OK:
            print("Stitching successful")
            cv.imwrite(os.path.join(self.save_path + "/stitched_result.jpg"), stitched)
            print(f"Image saved to {self.save_path}")
            exit(0)
        else:
            print(f"Stitching failed with status code: {status}")

def main(args=None):
    rclpy.init(args=args)

    map_stitcher = MapStitcher()

    rclpy.spin(map_stitcher)
    map_stitcher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()