import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.KeyStrokeSub = self.create_subscription(String, 'keyboard_input', self.KeyStroke_Callback, 10)
        # self.timer = self.create_timer(1.0, self.publish_cmd_vel)
        # self.state = 'MOVE'
        self.get_logger().info('CmdVel Publisher Node has been started')

    def KeyStroke_Callback(self, msg):
        
        velmsg = Twist()
        if (msg.data == "w"):
            velmsg.linear.x = 1.0
            self.publisher_.publish(velmsg)

        elif (msg.data == "s"):
            velmsg.linear.x = 0.0
            self.publisher_.publish(velmsg)

        else:
            self.get_logger().info('invalid key')



    def publish_cmd_vel(self):
        msg = Twist()
        if self.state == 'MOVE':
            # Move forward at 0.2 m/s
            msg.linear.x = 0.2
            msg.angular.z = 0.0
            self.get_logger().info('Publishing: Move forward')
            self.state = 'STOP'
        else:
            # Stop the TurtleBot
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.get_logger().info('Publishing: Stop')
            self.state = 'MOVE'
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down CmdVel Publisher Node')
    finally:
        # Ensure the TurtleBot stops when the node shuts down
        stop_msg = Twist()
        node.publisher_.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()