import rclpy
from rclpy.node import Node
import cv2 as cv
import os
import numpy as np
import shutil

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

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

        self.VelSub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.vel_callback,
            10
        )

        self.get_logger().info("Subscribed to /camera/image_raw/compressed")

        # Declare number of images to be saved
        self.declare_parameter('num_saved_imgs', 40)
        self.num_saved_imgs = self.get_parameter('num_saved_imgs').get_parameter_value().integer_value

        # Declare how many images to skip
        self.declare_parameter('skip_imgs', 4)
        self.skip_imgs = self.get_parameter('skip_imgs').get_parameter_value().integer_value

        self.get_logger().info(f"Number of images to be stitched: {self.num_saved_imgs}.\n To change this parameter use --ros-args -p num_saved_imgs:=[INTEGER]\n")
        self.get_logger().info(f"Skipping every {self.skip_imgs} images before saving. To change this parameter use --ros-args -p skip_imgs:=[INTEGER]\n")

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
                    break
                else:
                    print(f"'{delete}' is an invalid option. Please enter 'Y' or 'N'.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{self.save_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Counter to track images
        self.counter = 0

        # Image skip counter
        self.img_skip_counter = self.skip_imgs

        # Flag to check for movement
        self.moving = False

    def save_image(self, msg):
        # Skip images after desired number of images are saved
        if self.counter >= self.num_saved_imgs:
            return
        
        # Skip specified number of images
        if self.img_skip_counter != self.skip_imgs:
            self.img_skip_counter += 1
            return
        
        # Don't save an image if the bot isn't moving
        if not self.moving:
            self.get_logger().info("Robot is  not moving so not saving any images...")
            return

        # Check if image is empty
        if not msg.data:
            self.get_logger().warn("Received empty image data. Skipping...")
            return

        # Convert to OpenCV image
        ImgArray = np.frombuffer(msg.data, np.uint8)
        CVImage = cv.imdecode(ImgArray, cv.IMREAD_COLOR)

        # Check if image has enough features
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(CVImage, None)
        if des is None and len(kp) < 10:
            self.get_logger().warn("Image did not have enough features. Skipping...")
            return
        
        # Check if decoding worked correctly
        if CVImage is None:
            self.get_logger().error("Failed to decode image. Skipping...")
            return

        # Save the image
        self.counter += 1
        self.img_skip_counter = 0
        filename = os.path.join(self.save_path, f"image_{self.counter}.jpg")
        cv.imwrite(filename, CVImage)
        self.get_logger().info(f"Saved image {filename}.")

        if self.counter == self.num_saved_imgs:
            self.stitch_image()

    # Function to stitch images
    def stitch_image(self):
        mode = cv.Stitcher_SCANS
        stitcher = cv.Stitcher.create(mode)

        # Set thresholds for feature detection
        stitcher.setWaveCorrection(True)
        stitcher.setRegistrationResol(0.6)
        stitcher.setPanoConfidenceThresh(0.05)

        images = []

        # Check if there enough images to create a stitch
        if len(os.listdir(self.save_path)) < 2:
            print("Not enough images in folder to attempt stitching.")
            return

        # Iterate through images and append valid images to array
        for filename in sorted(os.listdir(self.save_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filepath = os.path.join(self.save_path, filename)
                image = cv.imread(filepath)

                if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                    print(f"Invalid or empty image: {filepath}")
                    continue

                images.append(image)

        print(f"Stitching {len(images)} images...")

        # Stitch images
        try:
            status, stitched = stitcher.stitch(images=images)
        except Exception as e:
            print(f"Exception during stitching: {e}")
            return

        # Check for errors when stitching
        if status == cv.Stitcher_OK:
            print("Stitching successful")
            output_path = os.path.join(self.save_path, "stitched_result.jpg")
            cv.imwrite(output_path, stitched)
            print(f"Panorama saved to {output_path}")
            self.get_logger().info("Stitching completed. Shutting down node.")
            rclpy.shutdown()
        elif status == cv.Stitcher_ERR_NEED_MORE_IMGS:
            print("Stitching failed: NEED_MORE_IMGS")
        elif status == cv.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Stitching failed: HOMOGRAPHY_EST_FAIL")
        elif status == cv.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Stitching failed: CAMERA_PARAMS_ADJUST_FAIL")
        else:
            print(f"Stitching failed with unknown status code: {status}")

    # Function to disable images saving if the bot is not moving
    def vel_callback(self, msg):
        if msg.linear.x > 0:
            self.moving = True
            return

def main(args=None):
    rclpy.init(args=args)

    map_stitcher = MapStitcher()

    rclpy.spin(map_stitcher)
    map_stitcher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()