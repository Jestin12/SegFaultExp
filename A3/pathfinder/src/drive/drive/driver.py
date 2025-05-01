import rclpy
from rclpy.node import Node

import math
import time
from collections import Counter

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

TURN_RIGHT = 0.5
TURN_LEFT = 1.5

class Driver(Node):
    def __init__(self):
        super().__init__("Driver")
        
        # Publishers
        self.MovePub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.GoalPosePub = self.create_publisher(PoseStamped, '/goal_pose', 10)  # Fixed: Publisher, not subscription
        self.ResetPub = self.create_publisher(String, '/reset', 10)     #Sends a reset signal

        # Subscriptions
        self.SignSub = self.create_subscription(String, '/pedestrian/ModeSign', self.SignSub_Callback, 10)
        self.KeyStrokeSub = self.create_subscription(String, 'keyboard_input', self.KeyStroke_Callback, 10)
        self.GoalPose = self.create_subscription(PoseStamped, '/goal_pose', self.GoalPose_Callback, 10)
        self.MoveSub = self.create_subscription(Twist, '/cmd_vel', self.CmdVel_Callback, 10)
        

        # Timer
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # State variables
        self.Nav2Flag = 2
        self.Action = None

        #History Variables
        self.RecentCmdVel = None
        self.rec = 0

    def timer_callback(self):
        if self.Nav2Flag == 1:
            self.get_logger().info('Nav2 finished')
            match self.Action:
                case 'Stop':
                    self.get_logger().info('Stopping')
                    self.stop()
                    self.Action = None

                    # Publish new goal pose
                    goal_pose = PoseStamped()
                    goal_pose.header.stamp = self.get_clock().now().to_msg()
                    goal_pose.header.frame_id = "map"
                    goal_pose.pose.position.x = 0.0  # Fixed: Float
                    goal_pose.pose.position.y = 0.0  # Fixed: Float
                    goal_pose.pose.position.z = 0.0

                   # Generate random yaw
                    random_yaw = random.uniform(-math.pi, math.pi)
                    quaternion = tf.quaternion_from_euler(0, 0, random_yaw)

                    goal_pose.pose.orientation.x = quaternion[0]
                    goal_pose.pose.orientation.y = quaternion[1]
                    goal_pose.pose.orientation.z = quaternion[2]
                    goal_pose.pose.orientation.w = quaternion[3]

                    # self.Nav2Flag = 3  # Reset to waiting for sign (or define Nav2Flag == 3 behavior)

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
                    self.stop()

                case _:
                    self.get_logger().info('Unknown behaviour: Action value invalid')

            self.Nav2Flag = 2
            self.Action = None

            msg = String()
            msg.data = "reset"
            self.ResetPub.publish(msg)

        elif self.Nav2Flag == 0:
            self.get_logger().info('Waiting for Nav2 to finish')

            if (self.Nav2Flag == 0):
                self.get_logger().info(f'0.0 count: {self.rec}')

            if (self.RecentCmdVel == 0.0):
                if (self.rec < 10):
                    self.rec += 1

            else:
                if (self.rec >= 0):
                    self.rec -= 1
                    

            if (self.rec >= 10):
                # self.get_logger().info('Nav2 has stopped moving to 1')
                self.Nav2Flag = 1
                self.rec = 0

        elif self.Nav2Flag == 2:
            self.get_logger().info(f'Waiting for Pedestrian to find a sign {self.Nav2Flag}')

        else:
            self.get_logger().info('Nav2Flag value is undefined, undefined behaviour')

    def findQuaternion(self, yaw):
        return Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )

    def GoalPose_Callback(self, msg):
        self.get_logger().info(f'Received goal pose: {msg.pose.position.x}, {msg.pose.position.y}')
        # Uncomment if needed: self.Nav2Flag = 1

    def SignSub_Callback(self, msg):
        self.Action = msg.data.split(' ')[1]
        self.get_logger().info(f'ModeSign Received, action: {self.Action}')
        if self.Nav2Flag == 2:
            self.Nav2Flag = 0

    def CmdVel_Callback(self, msg):
        self.RecentCmdVel = msg.linear.x
        


    def rotate_90_degrees(self, direction):
        angular_velocity = 0.0
        match direction:
            case 'left':
                angular_velocity = 2.0
            case 'right':
                angular_velocity = -2.0

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
        # if self.Nav2Flag == 0:
        #     self.Nav2Flag = 1

def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()