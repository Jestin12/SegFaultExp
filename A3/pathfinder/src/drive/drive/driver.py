import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist

TURN_RIGHT = 0.5 
TURN_LEFT = 1.5


class Driver(Node):
    def __init__(self):
        super().__init__("Driver")
        
        self.MovePub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.SignSub = self.create_subscription(String, 'ModeSign', self.SignSub_Callback, 10)

        self.MoveForward()

    
    def MoveForward(self): 
        self.update_cmd_vel(2.0, 0)


    def SignSub_Callback(self, msg):
        command = msg.data

        if command == "Ahead only": 
            pass

        elif command = "Turn left": 
            self.update_angular_vel(TURN_LEFT)

        elif command == "Turn right": 
            self.update_angular_vel(-1*TURN_RIGHT)

        elif command == "Roundabout mandatory":
            pass 

        elif command == "Stop": 
            self.Stop()


    def update_angular_vel(self, double angular):

        cmd_vel = Twist()

        cmd_vel.angular.z = angular

        self.MovePub.publish(cmd_vel) 


    def update_cmd_vel(self, double linear, double angular):

        cmd_vel = Twist()

        cmd_vel.linear.x = linear
        cmd_vel.angular.z = angular

        self.MovePub.publish(cmd_vel) 


    
    def Stop(self):

        cmd_vel = Twist()

        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0

        self.MovePub.publish(cmd_vel) 



def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()




        



    