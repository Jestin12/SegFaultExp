import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped

import numpy as np 

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
		self.SignSub = self.create_subscription(String, 'ModeSign', self.coordinate_callback, 10)



	def coordinate_callback(self, msg): 
		# Convert camera points to floats 
		command = msg 



		# Transform from camera to LiDAR frame 
		x = float(msg.x)
		y = float(msg.y)
		
		camera_point = np.vstack(np.array([[x], [y]]), [1]) 
		
		transformed_point = np.linalg.inv(self.K) @ camera_point
		
		lidar_point = self.CTL @ transformed_point



		# Transform from robot to Nav2 frame 

		robot_transformation = 

		nav_point = 

		

		# Add to current position of robot ?????????????????
		# final_point = self.pose + nav_point

		# new_pose_x = final_point[0]
		# new_pose_y = final_point[1]
		# new_pose_z = final_point[2]
		

		# Publish updated waypoint 

		goal_pose = PoseStamped()

		goal_pose.pose.position.x = nav_point[0]
		goal_pose.pose.position.y = nav_point[1]
		goal_pose.pose.position.z = nav_point[2] 


		# Publish an orientation 
		


		self.CoordinatePub.publish(goal_pose)







def main(args=None):
    rclpy.init(args=args)
    node = CoordinateFinder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        



    