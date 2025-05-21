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





def main(args=None):
    rclpy.init(args=args)
    node = CoordinateFinder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()