import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String

import os

class Follower(Node): 
	def __init__(self): 
		super().__init__("Follower")

		# Set pin numbers for IR sensors 
		self.LEFT = 25
		#self.CENTRE = 25
		self.RIGHT = 26
		
		GPIO.setmode(GPIO.BCM)
		GPIO.setup([self.LEFT, self.RIGHT], GPIO.IN)


		# Initialise variables 
		self.movement_command = 1

		# Create publishers 
		self.vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
		self.IR_publisher = self.create_publisher(String, "/IR_value", 10)

		self.timer = self.create_timer(0.05, self.check_sensors)

		self.vel_msg_copy = Twist()

		# Create subscribers 
		self.detection_subscriber = self.create_subscription(String, "/robot_status", self.progress_detector, 10)


	def check_sensors(self): 

		# Extracting the values from the GPIO pins 
		left_value = GPIO.input(self.LEFT)
		# centre_value = GPIO.input(self.CENTRE)
		right_value = GPIO.input(self.RIGHT)

		# Publish IR values 
		ir_msg = String()
		# ir_msg.data = f"Left: {left_value}, Centre: {centre_value}, Right: {right_value}"
		ir_msg.data = f"Left: {left_value}, Right: {right_value}"
		self.IR_publisher.publish(ir_msg)

		vel_msg = Twist()
		vel_msg = self.vel_msg_copy


		# Check if turtlebot is conducting detection or arm kinematics process 
		if (self.movement_command) == 1: 

			# STRAIGHT
			if left_value == 1 and right_value == 1:
				vel_msg.linear.x = 0.05
				# self.vel_msg.angular.z = 0.0 
				

			# ROTATE RIGHT
			elif left_value == 0 and right_value == 1: 
				# self.memory = "left"
				vel_msg.linear.x = 0.05
				vel_msg.angular.z = 0.1

			# ROTATE LEFT 
			elif left_value == 1 and right_value == 0: 
				# self.memory = "right"
				vel_msg.linear.x = 0.05
				vel_msg.angular.z = -0.1
			
			else: 
				vel_msg.linear.x = 0.0
				vel_msg.angular.z = 0.0

		else: 
			vel_msg.linear.x = 0.0
			vel_msg.angular.z = 0.0 

		self.vel_msg_copy = vel_msg

		# # Check if turtlebot is conducting detection or arm kinematics process 
		# if (self.detected and self.arm_state) == 0: 

		# 	# STRAIGHT
		# 	if left_value == 0 and centre_value == 0 and right_value == 0:
		# 		vel_msg.linear.x = 0.5
		# 		vel_msg.angular.z = 0.0 

		# 	# ROTATE RIGHT
		# 	elif left_value == 1 and centre_value == 0 and right_value == 0: 
			
		# 		vel_msg.linear.x = 0.5
		# 		vel_msg.angular.z = 0.5

		# 	# ROTATE LEFT 
		# 	elif left_value == 0 and centre_value == 0 and right_value == 1: 

		# 		vel_msg.linear.x = 0.5
		# 		vel_msg.angular.z = -0.5
			
		# 	# STOP
		# 	elif left_value == 1 and centre_value == 1 and right_value == 1: 
		# 		vel_msg.linear.x = 0.0
		# 		vel_msg.angular.z = 0.0 
			

		# else: 
		# 	vel_msg.linear.x = 0.0
		# 	vel_msg.angular.z = 0.0 


		# Publish movement command
		self.vel_publisher.publish(vel_msg)


	def progress_detector(self): 
		msg = String() 

		if msg.data == "START": 
			self.detected = 1 
		else: 
			self.detected = 0
	



def main(args=None): 
	rclpy.init(args=args)
	node = Follower()
	rclpy.spin(node)
	rclpy.shutdown()


if __name__ == "__main__": 
	main()