import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String, UInt32MultiArray

from sympy import symbols, Eq, solve, sqrt, atan2, acos, cos, sin, pprint, evalf
import numpy as np
import time




class ArmKinematics(Node):
	def __init__(self): 
		super().__init__("ArmKinematics")

		# Defining link lengths (cm)
		self.L1 = 15
		self.L2_open = 25
		self.L2_closed = 28.5
		
		# Creating publishers 
		# self.joint_publisher = self.create_publisher(UInt32MultiArray, '/joint_signals', 10)
		self.joint_publisher = self.create_publisher(String, '/joint_signals', 10)
		# self.MovePub = self.create_publisher(Twist, "/cmd_vel", 10)
		self.MovePub = self.create_publisher(Twist, "/ArmKinematicsVel", 10)

		# Creating Subscribers 
		self.SignSub = self.create_subscription(String, '/plant_detection', self.coordinate_callback, 10)


		# Constant distances (cm)
		self.camera_height = 13

		# Intrinsic calibration matrix
		self.K = np.array([[1234.269707, 0.000000, 250.407578],
                        [0.000000, 1232.093593, 233.484009],
                        [0.000000, 0.000000, 1.000000]])
		

		# Camera coordinates 
		self.x_cam = -6.6
		self.y_cam = -19.5
		self.z_cam = 2.5
		self.camera_coordinates = np.array([self.x_cam, self.y_cam, self.z_cam])

		# Arm base coordinates 
		self.x_arm = 0
		self.y_arm = 0
		self.z_arm = 0
		self.arm_base_coordinates = np.array([self.x_arm, self.y_arm, self.z_arm])

		self.translation_vector = self.arm_base_coordinates - self.camera_coordinates
		
		# Camera to arm base transformation matrix 
		self.CTA = np.eye(4)
		self.CTA[:3, 3] = self.translation_vector


		# Servo duty cycle parameters 
		self.base_servo = {"name": "base",
							"min_duty": 17,
							"max_duty": 63, 
							"min_angle": 10, 
							"max_angle": 138}
		
		self.elbow_servo = {"name": "elbow",
						 	"max_duty": 12, 
							"min_duty": 5,
							"max_angle": 138, 
							"min_angle": 0}
		
		self.last_detection_time = time.time() - 5


	def coordinate_callback(self, msg): 
		current_time = time.time()
		if current_time - self.last_detection_time < 5:
			self.get_logger().info("Skipping re-detection")
			return

		self.last_detection_time = current_time

		self.get_logger().info(f"Detected Plant - {msg.data.split(' ')[0]}. Coordinates: {msg.data.split(' ')[-2]}, {msg.data.split(' ')[-1]}")

		# Convert camera points to floats 
		command = msg.data.split() 

		if command[0] == "Healthy": 

			u = float(command[-2])
			v = float(command[-1])

			# Transform from 2D camera to 3D camera frame 
			pixel_point = np.vstack((np.array([[u], [v]]), [[1]])) 
			
			camera_point = np.linalg.inv(self.K) @ pixel_point
			
			# Set the height of the detected plant to be the height of the camera 
			camera_point[2] = -self.camera_height

			homogeneous_point = np.vstack((camera_point, [[1]]))

			# Transform from camera to arm frame 
			reference_point = self.CTA @ homogeneous_point

			x_point = float(reference_point[0])
			y_point = float(reference_point[1])
			z_point = float(reference_point[2])

			self.get_logger().info(f"x_point: {x_point}")
			self.driveTo_x(x_point)
			self.move_arm(y_point, z_point)
				

    

	# Function to align arm to x position of detected object
	def driveTo_x(self, x_point): 
	# 	None
		x_speed = 0.1
		msg = Twist()
		msg.linear.x = x_speed
		msg.angular.z = 0.0

		delay = x_point * 0.01 / x_speed

		self.MovePub.publish(msg)

		time.sleep(delay)

		msg = Twist()
		msg.linear.x = 0.0

		self.MovePub.publish(msg)

	# Function to check if arm can be moved 
	def move_arm(self, Ye, Ze): 


		self.get_logger().info(f"Move Arm")

		# # Define symbolic variables
		# Theta1, Theta2 = symbols('Theta1 Theta2')

		# EqY = Eq(Ye, self.L1*cos(Theta1) + self.L2_closed*cos(Theta2 - Theta1))
		# EqZ = Eq(Ze, self.L1*sin(Theta1) - self.L2_closed*sin(Theta2 - Theta1))

		# joint_angles = solve((EqY, EqZ), (Theta1, Theta2))

		r = np.sqrt(Ye**2 + Ze**2)

		# Ensure values are proper floats
		numerator = float(self.L1**2 + self.L2_closed**2 - r**2)
		denominator = float(2 * self.L1 * self.L2_closed)
		cos_angle = numerator / denominator

		Theta2 = np.pi - np.arccos(cos_angle)

		numerator1 = float(self.L1**2 - self.L2_closed**2 + r**2)
		denominator1 = float(2 * self.L1 * r)
		cos_angle1 = numerator1 / denominator1

		Theta1 = np.arccos(cos_angle1) + np.arctan2(Ze, Ye)

		# self.get_logger().info(f"Found Angles.")
		# print(joint_angles)

		
		# Iterate through each pair of angles to find a valid solution 
		# for idx, pair in enumerate(joint_angles): 

			# theta1 = np.rad2deg(float(pair[0].evalf().as_real_imag()[0]))
			# theta2 = np.rad2deg(float(pair[1].evalf().as_real_imag()[0]))

		theta1 = np.rad2deg(Theta1)
		theta2 = np.rad2deg(Theta2)

		self.get_logger().info(f"theta 1: {theta1}")
		self.get_logger().info(f"theta 2: {theta2}")

		theta1_signal = self.find_dutyCycle(theta1, self.base_servo)
		theta2_signal = self.find_dutyCycle(theta2, self.elbow_servo)

		self.get_logger().info(f"duty cycle 1: {theta1_signal}")
		self.get_logger().info(f"duty cycle 2: {theta2_signal}")

		# Check if plant is in workspace 
		if theta1_signal < self.base_servo["min_duty"] or theta1_signal > self.base_servo["max_duty"]: 
			self.get_logger().info(f"Outside of Workspace.")
			# continue 
		elif theta2_signal < self.elbow_servo["min_duty"] or theta2_signal > self.elbow_servo["max_duty"]: 
			self.get_logger().info(f"Outside of Workspace.")
			# continue 	

		
		# Check for singularities
		elif theta2 == 0: 
			self.get_logger().info(f"Singularity Detected.")
			# continue 	


		else: 
			# Publish the joint angles if they are valid 
			# joints = UInt32MultiArray()
			# joints.data = [int(np.round(theta1_signal)), int(np.round(theta2_signal))]
			# self.joint_publisher.publish(joints)
			joints = String()
			joints.data = f"{int(np.round(theta1_signal))},{int(np.round(theta2_signal))}"
			self.joint_publisher.publish(joints)
			print("Published Angles")
			# break


	# Function to find duty cycle corresponding to angles 
	def find_dutyCycle(self, angle, dict):
		step_size = (dict["max_angle"] - dict["min_angle"])/(dict["max_duty"] - dict["min_duty"])
		num_steps = (angle - dict['min_angle'])/step_size
		duty_cycle = dict["max_duty"] - num_steps 

		return duty_cycle





def main(args=None):
    rclpy.init(args=args)
    node = ArmKinematics()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

