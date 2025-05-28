import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String
from sensor_msgs.msg import Imu
import math


import os

class Follower(Node): 
	def __init__(self): 
		super().__init__("Follower")

		# Set pin numbers for IR sensors 
		self.LEFT = 25
		self.RIGHT = 26
		
		GPIO.setmode(GPIO.BCM)
		GPIO.setup([self.LEFT, self.RIGHT], GPIO.IN)


		# Initialise variables 
		self.movement_command = 1
		self.current_yaw = 0.0
		self.target_yaw = None
		self.kp = 1.0
		self.memory = "straight"
		self.vel_msg_copy = Twist()

		# Create timer
		self.timer = self.create_timer(0.05, self.check_sensors)

		# Create publishers 
		self.vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
		self.IR_publisher = self.create_publisher(String, "/IR_value", 10)
		
		# Create subscribers 
		self.detection_subscriber = self.create_subscription(String, "/plant_detect", self.plant_detector, 10)
		self.imu_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback,10)


	def imu_callback(self, msg):
		q = msg.orientation
		yaw = self.quaterion_to_yaw(q)
		self.current_yaw = yaw


	def quaternion_to_yaw(self, q):
        # Convert quaternion to yaw angle
		siny_cosp = 2 * (q.w * q.z + q.x * q.y)
		cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
		yaw = math.atan2(siny_cosp, cosy_cosp)
		return yaw


	def normalize_angle(self, angle):
        # Normalize angle to [-pi, pi]
		while angle > math.pi:
			angle -= 2.0 * math.pi
		while angle < -math.pi:
			angle += 2.0 * math.pi
		return angle
	
	# def plant_detector(self, msg):
			# EDIT BASED ON HOW THE MESSAGE IS RECIEVED


	def check_sensors(self): 

		# Extracting the values from the GPIO pins 
		left_value = GPIO.input(self.LEFT)
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
				
				# Use a KP controller to keep it straight while travelling in a straight line
				if ((self.target_yaw == None) and (self.memory == "straight")):
					self.target_yaw = self.current_yaw

				self.memory = "straight"

				yaw_error = self.normalize_angle(self.current_yaw - self.target_yaw)

				# Check if the robot deviated from initial yaw and apply correction for angular velocity
				ang_correction = self.kp * yaw_error

				vel_msg.linear.x = 0.05
				vel_msg.angular.z = -ang_correction
				

			# ROTATE RIGHT
			elif left_value == 0 and right_value == 1: 
				# Reset yaw and begin 
				self.target_yaw = None
				self.memory = "left"
				vel_msg.linear.x = 0.05
				vel_msg.angular.z = 0.05

			# ROTATE LEFT 
			elif left_value == 1 and right_value == 0: 
				self.target_yaw = None
				self.memory = "right"
				vel_msg.linear.x = 0.05
				vel_msg.angular.z = -0.05

			# Stop when you detect a plant
			# elif 

			
			else: 
				vel_msg.linear.x = 0.0
				vel_msg.angular.z = 0.0

		else: 
			vel_msg.linear.x = 0.0
			vel_msg.angular.z = 0.0 

		self.vel_msg_copy = vel_msg

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