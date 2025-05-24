import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String 

from sympy import symbols, Eq, solve, sqrt, atan2, acos, cos, sin, pprint, evalf
import numpy as np


class ArmKinematics(Node):
	def __init__(self): 
		super().__init__("ArmKinematics")

		# Defining link lengths 
		self.L1 = 15
		self.L2 = 30
		
		# Creating publishers 
		self.joint_publisher = self.create_publisher(String, '/joint_angles', 10)
		self.MovePub = self.create_publisher(Twist, "/cmd_vel", 10)
		self.status_publisher = self.create_publisher(String, '/robot_status', 10)


		# Creating Subscribers 
		self.SignSub = self.create_subscription(String, '/plant_detection', self.coordinate_callback, 10)

		# Constant distances 
		self.camera_height = 10 #cm

		# Intrinsic calibration matrix
		self.K = np.array([[1280.514822, 0.000000, 352.819379],
                        [0.000000, 1279.348064, 260.395041],
                        [0.000000, 0.000000, 1.000000]])
		
		# Camera coordinates 
		self.x_cam = -5
		self.y_cam = 0
		self.z_cam = -1 
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
	

	def coordinate_callback(self, msg): 
		self.get_logger().info(f"Detected Plant. Coordinates: {msg.data.split(' ')[-2]}, {msg.data.split(' ')[-1]}")

		# Stop line following 
		status_msg = String() 
		status_msg.data = "STOP"
		self.status_publisher.publish(status_msg)

		# Convert camera points to floats 
		command = msg.data.split() 

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

		x_point = reference_point[0]
		y_point = reference_point[1]
		z_point = reference_point[2]

		self.driveTo_x(x_point)
		self.move_arm(y_point, z_point)
    

	# Function to align arm to x position of detected object
	def driveTo_x(self, x_point): 
		None
	
	
	# Function to check if arm can be moved 
	def move_arm(self, Ye, Ze): 

		# Define symbolic variables
		Theta1, Theta2 = symbols('Theta1 Theta2')

		EqY = Eq(Ye, self.L1*cos(Theta1) + self.L2*cos(Theta1 + Theta2))
		EqZ = Eq(Ze, self.L1*sin(Theta1) + self.L2*sin(Theta1 + Theta2))

		joint_angles = solve((EqY, EqZ), (Theta1, Theta2))

		# Iterate through each pair of angles to find a valid solution 
		for idx, pair in enumerate(joint_angles): 
			theta1 = pair[Theta1]
			theta2 = pair[Theta2]

			# Check for singularities
			if theta2.evalf() == 0 or theta2.evalf() == np.pi: 
				continue
		
			# Check if angle is in workspace 
			if not self.in_workspace(theta1, theta2): 
				continue

			else: 
				# Publish the joint angles if they are valid 
				joints = String() 
				joints.data = f"Theta 1: {theta1.evalf()}, Theta 2: {theta2.evalf()}"
				self.joint_publisher.publish(joints)
				break


	def in_workspace(self, theta1, theta2): 
		return True 
		



def main(args=None):
    rclpy.init(args=args)
    node = ArmKinematics()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

