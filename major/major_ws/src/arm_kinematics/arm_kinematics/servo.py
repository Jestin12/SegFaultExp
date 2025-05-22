import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String

from piservo import Servo
import time

BASE = 13
ELBOW = 6
GRIPPER = 5


class ServoController(Node): 
	def __init__(self): 
		super().__init__("ServoController")


		# Create a subscriber to the joint angles 
		self.AngleSub = self.create_publisher(String, '/joint_angles', self.angles_callback, 10)

		self.joint1 = Servo(BASE, min_value=0, max_value=180, min_pulse=1.6, max_pulse=2, frequency=50)
		self.joint2 = Servo(ELBOW, min_value=0, max_value=180, min_pulse=1.6, max_pulse=2, frequency=50)
		self.joint3 = Servo(GRIPPER, min_value=0, max_value=180, min_pulse=1.6, max_pulse=2, frequency=50)

	
	def angles_callback(self, msg):
		joint_angles = msg.data.split()

		theta1 = float(joint_angles[0]) 
		theta2 = float(joint_angles[1])
		theta3 = float(joint_angles[2])


		# Give servos angle
		self.joint1.write(theta1) 
		self.joint2.write(theta2)
		self.joint3.write(theta3)


def main(args=None):
    rclpy.init(args=args)
    node = ServoController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()