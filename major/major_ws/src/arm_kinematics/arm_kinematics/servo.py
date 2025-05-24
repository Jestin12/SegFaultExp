import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String
from piservo import Servo
import time
import numpy as np


class ServoController(Node): 
	def __init__(self): 
		super().__init__("ServoController")


		# Assign GPIO pins 
		self.BASE = 16
		self.ELBOW = 20
		self.GRIPPER = 13
		
		GPIO.setmode(GPIO.BCM)
		GPIO.setup([self.BASE, self.ELBOW, self.GRIPPER], GPIO.OUT)


		# Create a subscriber to the joint angles 
		self.AngleSub = self.create_publisher(String, '/joint_angles', self.angles_callback, 10)

		# Create a publisher to robot status 
		self.status_publisher = self.create_publisher(String, '/robot_status', 10)

		self.joint1 = Servo(self.BASE, min_value=0, max_value=180, min_pulse=1.6, max_pulse=2, frequency=333)
		self.joint2 = Servo(self.ELBOW, min_value=0, max_value=180, min_pulse=1.6, max_pulse=2, frequency=50)
		self.joint3 = Servo(self.GRIPPER, min_value=0, max_value=180, min_pulse=1.6, max_pulse=2, frequency=50)

	
	def angles_callback(self, msg):
		joint_angles = msg.data.split()

		theta1 = float(joint_angles[0]) 
		theta2 = float(joint_angles[1])
		theta3 = float(joint_angles[2])

		# Give servos angle
		self.joint1.write(theta1) 
		self.joint2.write(theta2)
		self.joint3.write(theta3)

		# Timer to return arm back to base position 
		time.sleep(10)
		self.joint1.write(0) 
		self.joint2.write(30)
		self.joint3.write(0)

		# Start line following 
		status_msg = String() 
		status_msg.data = "START"
		self.status_publisher.publish(status_msg)



def main(args=None):
    rclpy.init(args=args)
    node = ServoController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()