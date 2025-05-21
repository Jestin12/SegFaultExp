import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String


class ArmKinematics(Node): 
	def __init__(self): 
		super().__init__("ArmKinematics")


		# Create Subscribers
		self.arm_subcriber = self.create_subscription(String, '/end_effector_pose', self.move_arm)
	
	def move_arm(self): 
            None


def main(args=None):
    rclpy.init(args=args)
    node = ArmKinematics()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()