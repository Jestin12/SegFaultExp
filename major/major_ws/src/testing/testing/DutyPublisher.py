#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import termios
import tty
import sys
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class DutyPublisherNode(Node):
    def __init__(self):
        super().__init__('DutyPublisher')
        
        # Create a publisher with the topic name 'keyboard_input'
        # Using String message type
        self.publisher = self.create_publisher(String, '/joint_angles', 10)

        self.get_logger().info('Keyboard Publisher started. Press keys to publish, ESC or Ctrl+C to exit.')
        
        # Start a thread for keyboard input
        self.is_running = True
        self.input_thread = threading.Thread(target=self.get_key_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
    
 

    # def get_key(self):
    #     """Get a single keypress from the user."""
    #     fd = sys.stdin.fileno()
    #     old_settings = termios.tcgetattr(fd)
    #     try:
    #         tty.setraw(sys.stdin.fileno())
    #         ch = sys.stdin.read(1)
    #     finally:
    #         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    #     return ch
    
    def get_key_loop(self):
        """Continuously get keypresses and publish them."""
        while self.is_running:
            duties = input("shoulder [17, 63], elbow [5, 12.5], hand [8, 12]: ")
                
            # Create and publish the message
            msg = String()
            msg.data = duties
            self.publisher.publish(msg)
            self.get_logger().info(f'Published: {duties})')



def main(args=None):
    rclpy.init(args=args)
    
    node = DutyPublisherNode()
    
    try:
        # Use a MultiThreadedExecutor to handle the keyboard input thread
        # and ROS2 callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        # Clean up
        node.is_running = False
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()