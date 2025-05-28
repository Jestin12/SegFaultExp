import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String
from piservo import Servo
from time import sleep
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

		# Setup PWM signals 
		self.base_pwm = GPIO.PWM(self.BASE, 333)
		self.base_pwm.start(0)  

		self.elbow_pwm = GPIO.PWM(self.ELBOW, 50)
		self.elbow_pwm.start(0)  

		self.gripper_pwm = GPIO.PWM(self.GRIPPER, 50)
		self.gripper_pwm.start(0)  


		# Create a subscriber to the joint angles 
		self.AngleSub = self.create_publisher(String, '/joint_angles', self.angles_callback, 10)

		# Create a publisher to robot status 
		self.status_publisher = self.create_publisher(String, '/robot_status', 10)

		
	
	def angles_callback(self, msg):
		joint_angles = msg.data.split()

		theta1 = float(joint_angles[0]) 
		theta2 = float(joint_angles[1])
		theta3 = float(joint_angles[2])

		# Give servos angle
		base_signal = self.find_PWM(theta1)
		elbow_signal = self.find_PWM(theta2)
		gripper_signal = self.find_PWM(theta3)

		# Set servo positions
		self.base_pwm.ChangeDutyCycle(base_signal)   
		sleep(1)

		self.elbow_pwm.ChangeDutyCycle(elbow_signal)   
		sleep(1)

		self.gripper_pwm.ChangeDutyCycle(gripper_signal)   
		sleep(1)

		# Start line following 
		status_msg = String() 
		status_msg.data = "START"
		self.status_publisher.publish(status_msg)


	def find_PWM(self, angle): 
		None

def main(args=None):
    rclpy.init(args=args)
    node = ServoController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()