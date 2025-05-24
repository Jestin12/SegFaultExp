import rclpy
from rclpy.node import Node
import cv2 as cv
import os
import numpy as np

from sensor_msgs.msg import CompressedImage

class StitchMapper(Node):
    def __init__(self):
        super().__init__('StitchMapper')
        self.ImageSub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.save_image,
            10
        )

        #Define home directory
        self.home_dir = os.environ.get("HOME")
        if self.home_dir is None:
            raise EnvironmentError("HOME environment variable is not set.")
        
        #Define save path and create directory for images
        self.save_path = os.path.join(self.home_dir, "stitch_map_images")
        try:
            os.mkdir(self.save_path)
            print(f"Directory '{self.save_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{self.save_path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{self.save_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        #Counter to track images
        self.counter = 0

    def save_image(self, msg):
        #Increment counter
        self.counter += 1

        #Convert to opencv image
        ImgArray = np.frombuffer(msg.data, np.uint8)
        CVImage = cv.imdecode(ImgArray, cv.IMREAD_COLOR)

        #Save image
        cv.imwrite(self.save_path + f"image_{self.counter}.jpg", CVImage)

    def stitch_image():
        pass
    

def main(args=None):
    rclpy.init(args=args)

    stitch_mapper = StitchMapper()

    rclpy.spin(stitch_mapper)
    stitch_mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()