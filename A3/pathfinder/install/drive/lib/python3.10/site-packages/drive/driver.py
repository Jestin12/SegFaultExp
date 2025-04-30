import rclpy
from rclpy.node import Node
import math
import numpy as np
import time

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

TURN_RIGHT = 0.5 
TURN_LEFT = 1.5




class Driver(Node):
    def __init__(self):
        super().__init__("Driver")
        
        self.MovePub = self.create_publisher(Twist, '/cmd_vel', 10)

        # self.SignSub = self.create_subscription(String, 'ModeSign', self.SignSub_Callback, 10)
        # self.LidarSub = self.create_subscription(LaserScan, 'scan', self.LidarSub_Callback, 10)
        # self.OdomSub = self.create_subscription(Odometry, 'odom', self.OdomSub_Callback, 10)
        self.KeyStrokeSub = self.create_subscription(String, 'keyboard_input', self.KeyStroke_Callback, 10)


        timer_period = 0.01  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)

        self.CurrState = "going"
        self.NextState = None

        
        '''
        States can be:
            going
            turning
            stop

        '''


    def rotate_90_degrees(self):
        angular_velocity = 2.0
        angle = math.pi / 2
        time_to_rotate = angle / angular_velocity

        twist_msg = Twist()
        twist_msg.angular.z = angular_velocity

        self.MovePub.publish(twist_msg)

        time.sleep(time_to_rotate * 1.1)

        twist_msg.angular.z = 0.0
        self.MovePub.publish(twist_msg)

        self.get_logger().info('Rotation Complete')


    def KeyStroke_Callback(self, msg):
        self.get_logger().info('key trigger')

        self.rotate_90_degrees()



        


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        



    