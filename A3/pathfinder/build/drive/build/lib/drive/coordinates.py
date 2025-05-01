import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, PointStamped, Vector3Stamped, Quaternion

import tf2_ros, tf2_geometry_msgs

import numpy as np 
import math 

class CoordinateFinder(Node): 
	def __init__(self): 
		super().__init__("CoordinateFinder")  

		self.pose = np.array([[0.0], [0.0], [0.0]])
		
		self.K = np.array([[2630.071584, 0.000000, 1070.429935],
							[0.000000, 2624.894868, 911.267456],
							[0.000000, 0.000000, 1.000000]])


		self.CTL = np.array([[-0.042659, -0.036711, 0.998415, 0.137702],
					[-0.998965, 0.017348, -0.042045, -0.005538],
					[-0.015777, -0.999175, -0.037414, -0.000957],
					[0.000000, 0.000000, 0.000000, 1.000000]])
		
		self.scale = 0.00922291727968761
		
		# Publisher to Nav2 
		self.CoordinatePub = self.create_publisher(PoseStamped, '/goal_pose', 10)

		# Subscriber to Camera coordinates 
		self.SignSub = self.create_subscription(String, '/pedestrian/ModeSign', self.coordinate_callback, 10)

		# Initialise tf transform listeners 
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)



	def coordinate_callback(self, msg): 

		# Convert camera points to floats 
		command = msg.data.split()

		# self.get_logger().info(command)

		if len(command) >= 3: 

			x = float(command[-2]) 
			y = float(command[-1]) 


			# Transform from camera to LiDAR frame 
			camera_point = np.vstack((np.array([[x], [y]]), [[1]])) 
			
			transformed_point = np.linalg.inv(self.K) @ camera_point

			homogeneous_point = np.vstack((transformed_point, [[1]]))
			
			lidar_point = self.CTL @ homogeneous_point

			# Create bound around y value to constrait LiDAR values 


			# Determine x value using average LiDAR depth within bounds 


			# Change the x value 


			# Convert lidar_point into PoseStamped format 
			lidar_point_msg = PointStamped()
			lidar_point_msg.header.frame_id = 'base_scan'
			lidar_point_msg.header.stamp = self.get_clock().now().to_msg()
			lidar_point_msg.point.x = lidar_point[0, 0]
			lidar_point_msg.point.y = lidar_point[1, 0]
			lidar_point_msg.point.z = lidar_point[2, 0]


			# Find vector for movement direction 
			direction_vector = lidar_point[:3].flatten()
			direction_vector /= np.linalg.norm(direction_vector)

			movement_vector = Vector3Stamped()
			movement_vector.header.frame_id = 'base_scan'
			movement_vector.header.stamp = self.get_clock().now().to_msg()
			movement_vector.vector.x = direction_vector[0]
			movement_vector.vector.y = direction_vector[1]
			movement_vector.vector.z = direction_vector[2]




			# Transform from robot to Nav2 frame 
			map_frame = 'map'
			lidar_frame = 'base_scan'

			robot_transformation = self.tf_buffer.lookup_transform(map_frame, lidar_frame, Time())

			nav_point = tf2_geometry_msgs.do_transform_point(lidar_point_msg, robot_transformation)
			transformed_vector = tf2_geometry_msgs.do_transform_vector3(movement_vector, robot_transformation)

			# Add to current position of robot ?????????????????
			# final_point = self.pose + nav_point

			# new_pose_x = final_point[0]
			# new_pose_y = final_point[1]
			# new_pose_z = final_point[2]
			

			# Publish updated waypoint 
			goal_pose = PoseStamped()

			goal_pose.header.stamp = self.get_clock().now().to_msg()
			goal_pose.header.frame_id = "map"  


			goal_pose.pose.position.x = nav_point.point.x
			goal_pose.pose.position.y = nav_point.point.y
			goal_pose.pose.position.z = 0.0


			yaw = math.atan2(transformed_vector.vector.y, transformed_vector.vector.x)
			goal_pose.pose.orientation = self.findQuarternion(yaw) 
			
			self.CoordinatePub.publish(goal_pose)

	
	def findQuarternion(self, yaw):
		return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    ) 





def main(args=None):
    rclpy.init(args=args)
    node = CoordinateFinder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        



    