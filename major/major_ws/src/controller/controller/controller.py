
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String

import os

class Controller(Node): 
	def __init__(self): 
		super().__init__("Controller")


		# Create Subscribers


		# Create Publishers 

	



def main(args=None): 
	rclpy.init(args=args)
	node = Controller()
	rclpy.spin(node)
	rclpy.shutdown()


if __name__ == "__main__": 
	main()