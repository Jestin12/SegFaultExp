import RPI.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String

import os

class Follower(Node): 
	def __init__(self): 
		super().__init__("Follower")

		# Set pin numbers for IR sensors 
		self.LEFT = 16
		self.CENTRE = 25
		self.RIGHT = 26
		
		GPIO.setmode(GPIO.BCM)
		GPIO.setup([self.CENTRE, self.LEFT, self.RIGHT], GPIO.IN)

		# Create publishers 
		self.vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
		self.IR_publisher = self.create_publisher(String, "/IR_value", 10)

		self.timer = self.create_timer(0.05, self.check_sensors)


	def check_sensors(self): 

		# Extracting the values from the GPIO pins 
		left_value = GPIO.input(self.LEFT)
		centre_value = GPIO.input(self.CENTRE)
		right_value = GPIO.input(self.RIGHT)


		vel_msg = Twist()
		ir_msg = String()

		if value == 0: 

			vel_msg.linear.x = 0.5
			vel_msg.angular.z = 0.0 

		else: 
			vel_msg.linear.x = 0.0
			vel_msg.angular.z = 0.0 


		ir_msg.data = str(value)

		self.vel_publisher.publish(vel_msg)
		self.IR_publisher.publish(ir_msg)



def main(args=None): 
	rclpy.init(args=args)
	node = Follower()
	rclpy.spin(node)
	rclpy.shutdown()


if __name__ == "__main__": 
	main()