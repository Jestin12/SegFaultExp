
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String

import os

class Controller(Node): 
	def __init__(self): 
		super().__init__("Controller")


		# Create Subscribers
		self.arm_subcriber = self.create_subscription(String, '/arm_status', self.arm_callback)
		self.detection_subscriber = self.create_subscription(String, '/detection_status', self.detection_callback)

		# Create Publishers 
		self.drive_publisher = self.create_publisher(String, '/drive_status', 10)
		self.arm_publisher = self.create_publisher(String, '/arm_status', 10) #this trigegrs arm_callback each time so think of timing logic 


	def arm_callback(self): 
		None 
		# check process of arm 

	def detection_callback(self): 
		None 
		#check if something has been detected 

def main(args=None): 
	rclpy.init(args=args)
	node = Controller()
	rclpy.spin(node)
	rclpy.shutdown()


if __name__ == "__main__": 
	main()