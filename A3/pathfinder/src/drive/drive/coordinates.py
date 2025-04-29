import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped

import numpy as np 



class CoordinateFinder(Node):
    def __init__(self):
        super().__init__("CoordinateFinder")
        
		# Publisher to Nav2 
        self.MovePub = self.create_publisher(PoseStamped, '/goal_pose', 10)

		# Subscriber to Camera coordinates 
        self.SignSub = self.create_subscription(String, 'ModeSign', self.coordinate_callback, 10)
        
		self.pose = np.array([[0.0], [0.0], [0.0]])
        
		self.K = 
        

		self.CTL = 

    

	def coordinate_callback(self, msg): 
            
        # Transform from camera to LiDAR frame 
        p_camera = 

		# Transform from robot to Nav2 frame 
        

		# Add to current position of robot 
        

		# Publish updated waypoint 
        
        


    






def main(args=None):
    rclpy.init(args=args)
    node = CoordinateFinder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        



    