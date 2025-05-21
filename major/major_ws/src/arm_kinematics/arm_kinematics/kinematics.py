import rclpy 
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String 


class ArmKinematics(Node):
	def __init__(self): 
		super().__init__("ArmKinematics")

		
		self.L1 = 15
		self.L2 = 30
            
		
		# Creating subscriber to end effector pose 
		self.pose_subscriber = self.create_subscription(String, "/end_effector_pose", self.move_arm)
            
	

	def move_arm(self, msg): 
            None




def main(args=None):
    rclpy.init(args=args)
    node = ArmKinematics()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()