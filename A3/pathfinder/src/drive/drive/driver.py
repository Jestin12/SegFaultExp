import rclpy
from rclpy.node import Node
import math
import numpy as np

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

TURN_RIGHT = 0.5 
TURN_LEFT = 1.5


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w])

def quaternion_to_yaw(q):
    """Convert a quaternion [x, y, z, w] to a yaw angle (radians)"""
    x, y, z, w = q

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def are_quaternions_close_about_z(q1, q2, threshold_rad=np.deg2rad(5)):
    """Check if two quaternions are close in yaw (within threshold_rad)"""
    yaw1 = quaternion_to_yaw(q1)
    yaw2 = quaternion_to_yaw(q2)

    # Compute shortest angular difference
    diff = np.arctan2(np.sin(yaw1 - yaw2), np.cos(yaw1 - yaw2))

    return abs(diff) <= threshold_rad


class Driver(Node):
    def __init__(self):
        super().__init__("Driver")
        
        self.MovePub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.SignSub = self.create_subscription(String, 'ModeSign', self.SignSub_Callback, 10)
        self.LidarSub = self.create_subscription(LaserScan, 'scan', self.LidarSub_Callback, 10)
        self.OdomSub = self.create_subscription(Odometry, 'odom', self.OdomSub_Callback, 10)

        # To get it started
        cmd_vel = Twist()
        cmd_vel.linear.x = 10
        self.MovePub.publish(cmd_vel)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.CurrState = "going"
        self.NextState = None
        '''
        States can be:
            going
            turning
            stop

        '''


        self.PosQ = [None, None, None, None]
        self.DesQ = [None, None, None, None]
        self.LidarData = []

    def LidarSub_Callback(self, msg):



    def timer_callback(self):
        self.get_logger().info('Timer callback triggered')

        cmd_vel = Twist()

        match self.CurrState:

            case "going":
                self.get_logger().info('In going state')

                cmd_vel = Twist()
                cmd_vel.linear.x = 10.0

                self.MovePub.publish(cmd_vel)


                if '''img thresh''':


            case "turning":

                self.get_logger().info('In turning state')
                
                cmd_vel = Twist()
                cmd_vel.angular.z = 10.0

                self.MovePub.publish(cmd_vel)

                while (are_quaternions_close_about_z(PosQ, DesQ) != True):
                    
                cmd_vel = Twist()
                cmd_vel.angular.z = 0.0

                self.MovePub.publish(cmd_vel)

                self.NextState = "going"


    def OdomSub_Callback(self, msg):
        
        self.PosQ = np.array([
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
        ])

    

    def SignSub_Callback(self, msg):
        command = msg.data

        if command == "Ahead only": 
            pass

        elif command = "Turn left": 
            self.update_angular_vel(TURN_LEFT)

        elif command == "Turn right": 
            self.update_angular_vel(-1*TURN_RIGHT)

        elif command == "Roundabout mandatory":
            pass 

        elif command == "Stop": 
            self.Stop()

    def MoveForward(self): 
        self.update_cmd_vel(2.0, 0)


    def update_angular_vel(self, double angular):

        cmd_vel = Twist()

        cmd_vel.angular.z = angular

        self.MovePub.publish(cmd_vel) 


    def update_cmd_vel(self, double linear, double angular):

        cmd_vel = Twist()

        cmd_vel.linear.x = linear
        cmd_vel.angular.z = angular

        self.MovePub.publish(cmd_vel) 


    
    def Stop(self):

        cmd_vel = Twist()

        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0

        self.MovePub.publish(cmd_vel) 



def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        



    