#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ScanSubscriber(Node):
    def __init__(self):
        super().__init__('scan_subscriber')
        # Define QoS profile for the subscriber
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        # Create a subscriber to the /scan topic with the custom QoS
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos
        )
        self.get_logger().info('Subscribed to /scan topic')

    def scan_callback(self, msg):
        try:
            self.get_logger().info(
                f"Angle min: {msg.angle_min:.2f} rad, Angle max: {msg.angle_max:.2f} rad, "
                f"Range at 0°: {msg.ranges[0]:.2f} m, Range at 90°: {msg.ranges[90]:.2f} m, "
                f"Range at 180°: {msg.ranges[180]:.2f} m, Range at 270°: {msg.ranges[270]:.2f} m"
            )
        except IndexError:
            self.get_logger().warn(
                f"Invalid range index. Expected ~360 samples, got {len(msg.ranges)} samples."
            )

def main(args=None):
    rclpy.init(args=args)
    scan_subscriber = ScanSubscriber()
    rclpy.spin(scan_subscriber)
    scan_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()