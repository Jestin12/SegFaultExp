import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist


class Driver(Node):
    def __init__(self):
        super().__init__("Driver")
        
        self.MovePub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.SignSub = self.create_subscription(String, 'pedestrian/Signs', self.SignSub_Callback, 10)


    # def SignSub_Callback(self, msg):



        



    