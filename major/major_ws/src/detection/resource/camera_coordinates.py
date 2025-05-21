import rclpy 
from rclpy.node import Node 
from rclpy.time import Time 

from std_msgs.msg import String 

import numpy as np 
import math 

################# TO DO #####################
# - modify K matrix 
# - measure camera position wrt base of arm 



class CoordinateFinder(Node): 
	def __init__(self): 
		super().__init__("CoordinateFinder")

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
	

		# Create publisher for end effector pose 
		self.CoordinatePub = self.create_publisher(String, '/end_effector_pose', 10)

		# Create subscriber to camera coordinates 
		self.SignSub = self.create_subscription(String, '/plant_detection', self.coordinate_callback, 10)


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
		pose_msg = String() 
		pose_msg.data = f"x: {x_point}, y: {y_point}, z: {z_point}"


		self.CoordinatePub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CoordinateFinder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()