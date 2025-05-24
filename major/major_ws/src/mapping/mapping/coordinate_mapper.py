import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import os


# Get current position from Odom data
####### Get image coordinates from image_detector data (How to do this????) - am i being sent it or would i have to find it?
# Publish a marker to the rviz map to show where it is
####### Combine the two to make a map and save it to "plant_map" using save_map


class Coordinate_Mapper(Node):
    def __init__(self): 
        super().__init__("Coordinate_Mapper")        

        self.timer = self.create_timer(0.05, self.publish_marker)
        
        self.marker_pub = self.create_publisher(Marker, '/feature_markers', 10) 
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.classify_sub = self.create_subscription(String, "/image_classify", self.image_classify, 10)

        # Assuming the coordinates are calculated and sent 
        # If coordinates have to be found - do the transformations and get rid of this subscriber - no use 
        self.plant_pose_sub = self.create_subscription(String, "/plant_pose", self.plant_pose, 10)


        self.current_position = None
        self.plant_pose = None

        self.marker = Marker()

        self.plant_classify = None

        # Marker size = 10 cm x 10 cm x 10 cm
        self.marker.scale.x = 0.01
        self.marker.scale.y = 0.01
        self.marker.scale.z = 0.01

    
    def odom_callback(self, msg):
        # Extract robot position from odometry
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    def image_classify(self, msg):
         self.plant_classify = msg.data
         self.get_logger().info(f"Plant classification: {self.plant_classify}")

    
    def plant_pose(self, msg):
        try:
            # Parse the string to extract x and y
            pose_data = msg.data.strip()  # Remove any extra whitespace
            
            # Assumed strcture of msg is "x: [num] : y: [num]"
            if ':' in pose_data: 
                x_str, y_str = pose_data.split(',')
                x = float(x_str.split(':')[1].strip())
                y = float(y_str.split(':')[1].strip())
            else:  # Format: "x , y"
                x, y = map(float, pose_data.split(','))

            # Save the extracted coordinates
            self.plant_pose = (x, y)

            # Log the coordinates for debugging
            self.get_logger().info(f"Plant pose: x={x}, y={y}")

        except Exception as e:
            self.get_logger().error(f"Failed to parse plant pose: {msg.data}. Error: {e}")


    def publish_marker(self):
        self.marker.header.frame_id = "map" 
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD
        self.marker.header.stamp = self.get_clock().now().to_msg()

        # Position
        self.marker.pose.position.x = self.plant_pose[0]  # Example position
        self.marker.pose.position.y = self.plant_pose[1]
        self.marker.pose.position.z = 0.0
        self.marker.pose.orientation.w = 1.0

        # Decide the marker colour based on the classification
        if self.plant_classify == "healthy":
            self.marker.color.a = 1.0
            self.marker.color.r = 0.0
            self.marker.color.g = 1.0
            self.marker.color.b = 0.0

        elif self.plant_classify  == "dead":
            self.marker.color.a = 1.0
            self.marker.color.r = 0.65
            self.marker.color.g = 0.16
            self.marker.color.b = 0.16
        
        elif self.plant_classify  == "weed":
            self.marker.color.a = 1.0
            self.marker.color.r = 0.0
            self.marker.color.g = 0.0
            self.marker.color.b = 1.0
        
        
        self.marker_pub.publish(self.marker)
        self.get_logger().info('Publishing marker')




def main(args=None): 
	rclpy.init(args=args)
	node = Coordinate_Mapper()
	rclpy.spin(node)
	rclpy.shutdown()


if __name__ == "__main__": 
	main()