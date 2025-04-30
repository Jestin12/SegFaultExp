import rclpy
from rclpy.node import Node
import math
import numpy as np
import time

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


TURN_RIGHT = 0.5 
TURN_LEFT = 1.5




class Driver(Node):
    def __init__(self):
        super().__init__("Driver")
        
        self.MovePub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.SignSub = self.create_subscription(String, '/pedestrian/ModeSign', self.SignSub_Callback, 10)
        # self.LidarSub = self.create_subscription(LaserScan, 'scan', self.LidarSub_Callback, 10)
        # self.OdomSub = self.create_subscription(Odometry, 'odom', self.OdomSub_Callback, 10)
        self.KeyStrokeSub = self.create_subscription(String, 'keyboard_input', self.KeyStroke_Callback, 10)
        self.GoalPose = self.create_subscription(PoseStamped, '/goal_pose', self.GoalPose_Callback,10)
        

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.CurrState = "going"
        self.NextState = None

        self.Nav2Flag = 2
        self.Action = None
        '''
        States can be:
            going
            turning
            stop

        '''

    def timer_callback(self):
        if (self.Nav2Flag == 1):
            self.get_logger().info('Nav2 finished')

            match self.Action:
                case 'Stop':
                    self.get_logger().info('Stopping')
                    self.stop()
                    self.Action = None

                case 'TurnRight':
                    self.get_logger().info('Turning Right')
                    self.rotate_90_degrees('right')
                    self.Action = None

                case 'TurnLeft':
                    self.get_logger().info('Turning Left')
                    self.rotate_90_degrees('left')
                    self.Action = None

                case None:
                    self.get_logger().info('Action not given')
                    # self.get_logger().info('')
                    self.stop()

                case _:
                    self.get_logger().info('Unknown behaviour: Action value invalid')

            self.Nav2Flag = 2
            self.Action = None

        elif (self.Nav2Flag == 0):
            self.get_logger().info('Waiting for Nav2 to finish')

        elif (self.Nav2Flag == 2):
            self.get_logger().info(f'Waiting for Pedestrian to find a sign {self.Nav2Flag}')
            # forward()

        else:
            self.get_logger().info('Nav2Flag value is undefined, undefined behaviour')
            

    def GoalPose_Callback(self):
        self.Nav2Flag = 1

    def SignSub_Callback(self, msg):
        self.Action = msg.data.split(' ')[1]
        self.get_logger().info(f'Nav2 finished, action: {self.Action}')
        
        if (self.Nav2Flag == 2):
            self.Nav2Flag = 0


    def rotate_90_degrees(self, direction):
        angular_velocity = 0.0

        match direction:
            case 'left':
                angular_velocity = 2.0

            case 'right':
                angular_velocity = -2.0


        # angular_velocity = 2.0
        angle = math.pi / 2
        time_to_rotate = angle / abs(angular_velocity)

        twist_msg = Twist()
        twist_msg.angular.z = angular_velocity

        self.MovePub.publish(twist_msg)

        time.sleep(time_to_rotate * 1.1)

        twist_msg.angular.z = 0.0
        self.MovePub.publish(twist_msg)

        self.get_logger().info('Rotation Complete')

    def stop(self):
        msg = Twist()
        msg.linear.x = 0.0
        self.MovePub.publish(msg)
    
    def forward(self):
        msg = Twist()
        msg.linear.x = 1.0
        self.MovePub.publish(msg)

    def KeyStroke_Callback(self, msg):
        self.get_logger().info('key trigger')

        self.Nav2Flag = 1



        


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        



    