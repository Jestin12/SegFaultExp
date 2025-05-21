import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String 

from sympy import symbols, Eq, solve, sqrt, atan2, acos, cos, sin, pprint
import numpy as np


class ArmKinematics(Node):
	def __init__(self): 
		super().__init__("ArmKinematics")

		# Defining link lengths 
		self.L1 = 15
		self.L2 = 30
		
		#publishers
		self.joint_publisher = self.create_publisher(String, '/joint_angles', 10)
		self.MovePub = self.create_publisher(Twist, "/cmd_vel", 10)


		#subscribers
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

		# Publish updated waypoint 
		# pose_msg = String() 
		# pose_msg.data = f"x: {x_point}, y: {y_point}, z: {z_point}"


		# self.CoordinatePub.publish(pose_msg)

		self.move_arm(y_point, z_point)
            
	
	
	def move_arm(self, Ye, Ze): 

		# msg = String()
		# command = msg.data.split() 

		# Ye = float(command[-2])
		# Ze = float(command[-1])

		# Define symbolic variables
		Theta1, Theta2 = symbols('Theta1 Theta2')

		EqY = Eq(Ye, self.L1*cos(self.Theta1) + self.L2*cos(self.Theta1 + self.Theta2))
		EqZ = Eq(Ze, self.L1*sin(self.Theta1) + self.L2*sin(self.Theta1 + self.Theta2))

		joint_angles = solve((EqY, EqZ), (Theta1, Theta2))

		# Publish the joint angles 
		joints = String() 
		joints.data = f"Theta 1: {joint_angles[0]}, Theta 2: {joint_angles[1]}"
		self.joint_publisher.publish(joints)

		



def main(args=None):
    rclpy.init(args=args)
    node = ArmKinematics()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()