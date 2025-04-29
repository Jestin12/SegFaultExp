#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import termios
import tty
import sys
import threading

class KeyboardPublisherNode(Node):
    def __init__(self):
        super().__init__('keyboard_publisher')
        
        # Create a publisher with the topic name 'keyboard_input'
        # Using String message type
        self.publisher = self.create_publisher(String, 'keyboard_input', 10)
        
        self.get_logger().info('Keyboard Publisher started. Press keys to publish, ESC or Ctrl+C to exit.')
        
        # Start a thread for keyboard input
        self.is_running = True
        self.input_thread = threading.Thread(target=self.get_key_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
    
    def get_key(self):
        """Get a single keypress from the user."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def get_key_loop(self):
        """Continuously get keypresses and publish them."""
        while self.is_running:
            key = self.get_key()
            
            # ASCII value 27 is ESC
            if ord(key) == 27:
                self.get_logger().info('ESC pressed, shutting down node...')
                self.is_running = False
                rclpy.shutdown()
                break
                
            # Create and publish the message
            msg = String()
            msg.data = key
            self.publisher.publish(msg)
            self.get_logger().info(f'Published: "{key}" (ASCII: {ord(key)})')

def main(args=None):
    rclpy.init(args=args)
    
    node = KeyboardPublisherNode()
    
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