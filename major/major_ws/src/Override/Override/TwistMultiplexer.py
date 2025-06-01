#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class TwistMultiplexer(Node):
    def __init__(self):
        super().__init__('twist_multiplexer')
        
        # Parameters
        self.timeout_duration = 3.0  # 3 seconds timeout
        
        # Subscribers for input topics
        self.arm_subscriber = self.create_subscription(
            Twist,
            'ArmKinematicsVel',
            self.arm_callback,
            10
        )
        
        self.line_subscriber = self.create_subscription(
            Twist,
            'LineFollowerVel', 
            self.line_callback,
            10
        )
        
        # Publisher for output topic
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # State variables
        self.last_arm_msg = None
        self.last_line_msg = None
        self.last_arm_time = 0.0
        self.last_line_time = 0.0
        
        # Timer to check timeouts and publish
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        
        self.get_logger().info('Twist Multiplexer started')
        self.get_logger().info('Priority: ArmKinematicsVel > LineFollowerVel')
        self.get_logger().info('Timeout: 3.0 seconds')
        
    def arm_callback(self, msg):
        """Callback for ArmKinematicsVel topic (higher priority)"""
        self.last_arm_msg = msg
        self.last_arm_time = time.time()
        self.get_logger().debug('Received arm command')
        
    def line_callback(self, msg):
        """Callback for LineFollowerVel topic (lower priority)"""
        self.last_line_msg = msg
        self.last_line_time = time.time()
        self.get_logger().debug('Received line follower command')
        
    def timer_callback(self):
        """Main logic to decide which message to publish"""
        current_time = time.time()
        output_msg = Twist()  # Default to zero velocity
        active_source = "none"
        
        # Check if arm commands are recent (higher priority)
        arm_valid = (self.last_arm_msg is not None and 
                    (current_time - self.last_arm_time) < self.timeout_duration)
        
        # Check if line follower commands are recent (lower priority)
        line_valid = (self.last_line_msg is not None and 
                     (current_time - self.last_line_time) < self.timeout_duration)
        
        # Priority logic: Arm > Line Follower > Stop
        if arm_valid:
            output_msg = self.last_arm_msg
            active_source = "ArmKinematicsVel"
        elif line_valid:
            output_msg = self.last_line_msg
            active_source = "LineFollowerVel"
        else:
            # Both timed out, send zero velocity
            output_msg = Twist()  # Already initialized to zeros
            active_source = "timeout_stop"
            
        # Publish the selected command
        self.cmd_vel_publisher.publish(output_msg)
        
        # Log source changes
        if not hasattr(self, 'last_active_source') or self.last_active_source != active_source:
            self.get_logger().info(f'Active control source: {active_source}')
            self.last_active_source = active_source

def main(args=None):
    rclpy.init(args=args)
    
    multiplexer = TwistMultiplexer()
    
    try:
        rclpy.spin(multiplexer)
    except KeyboardInterrupt:
        pass
    finally:
        multiplexer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
