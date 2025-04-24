import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge
import cv2
import rosbag2_py
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import os

class ImageRepublisher(Node):
    def __init__(self):
        super().__init__('image_republisher')
        self.bridge = CvBridge()

        # Publishers
        self.image_pub = self.create_publisher(CompressedImage, "/camera/image_flipped", 10)
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.timer_callback, 10)

    def timer_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS image to OpenCV: {e}")
            return
        
        # Flip the image (horizontal, vertical, or both)
        self.root = cv2.flip(cv_image, -1)  # Flip x-axis (0), y-axis(1), both (-1)
        
        # Convert back to ROS Image
        flipped_msg = self.bridge.cv2_to_imgmsg(self.root, encoding="rgb8")
        flipped_msg.header = msg.header  # Keep the original timestamp
        self.image_pub.publish(flipped_msg)
        self.get_logger().info("Published flipped image")


def main(args=None):
    rclpy.init(args=args)

    image_republisher = ImageRepublisher()

    rclpy.spin(image_republisher)

    image_republisher.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
