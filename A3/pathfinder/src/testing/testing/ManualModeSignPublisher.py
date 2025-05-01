import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class ManualModeSignPublisher(Node):
    def __init__(self):
        super().__init__('ManualModeSignPublisher')
        
        self.publisher_ = self.create_publisher(String, '/pedestrian/ModeSign', 10)

        # self.CmdVelSub = self.create_subscription(Twist, '/cmd_vel', self.CmdVel_Callback)
        self.KeyStrokeSub = self.create_subscription(String, 'keyboard_input', self.KeyStroke_Callback, 10)

        # self.timer = self.create_timer(1.0, self.publish_string)
        self.counter = 0
        self.get_logger().info('ManualModeSignPublisher Node has been started')


    def KeyStroke_Callback(self, msg):
        if (msg.data == "g"):
            self.publish_string()


    def publish_string(self):
        msg = String()
        msg.data = ' TurnRight 292 268'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        

def main(args=None):
    rclpy.init(args=args)
    node = ManualModeSignPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down ManualModeSignPublisher Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()