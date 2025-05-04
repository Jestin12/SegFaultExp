import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

from geometry_msgs.msg import Twist, PoseStamped, PointStamped, Vector3Stamped, Quaternion


import tf2_ros, tf2_geometry_msgs

import numpy as np 
import math 

ANGLE_INCREMENT = 0.0174532923847436
CYLINDER_RADIUS = 0.05

qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    depth=10
)

class CoordinateFinder(Node): 
    def __init__(self): 
        super().__init__("CoordinateFinder")  

        self.ranges = []
        self.centroids = []
        
        self.K = np.array([[1280.514822, 0.000000, 352.819379],
                        [0.000000, 1279.348064, 260.395041],
                        [0.000000, 0.000000, 1.000000]])


        self.CTL = np.array([[-0.042659, -0.036711, 0.998415, 0.137702],
                    [-0.998965, 0.017348, -0.042045, -0.005538],
                    [-0.015777, -0.999175, -0.037414, -0.000957],
                    [0.000000, 0.000000, 0.000000, 1.000000]])
        
        self.scale = 0.00922291727968761
        
        # Publisher to Nav2 
        self.CoordinatePub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.CentroidPub = self.create_publisher(PoseStamped, '/centroid_pose', 10)

        # Subscriber to Camera coordinates 
        self.SignSub = self.create_subscription(String, '/pedestrian/ModeSign2', self.coordinate_callback, 10)
        self.LaserSub = self.create_subscription(LaserScan, '/scan', self.store_laser, qos_profile=qos_profile)

        # Initialise tf transform listeners 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def store_laser(self, msg):
        scan = msg
        ranges = scan.ranges
        self.ranges = ranges


    def coordinate_callback(self, msg): 

        self.get_logger().info(f"Received Goal Pose: {msg.data.split(' ')[-2]}, {msg.data.split(' ')[-1]}")
        # Convert camera points to floats 
        command = msg.data.split()

        if len(command) >= 3: 

            u = float(command[-2]) 
            v = float(command[-1]) 

            # Transform from camera to LiDAR frame 
            pixel_point = np.vstack((np.array([[u], [v]]), [[1]])) 
            
            camera_point= np.linalg.inv(self.K) @ pixel_point

            homogeneous_point = np.vstack((camera_point, [[1]]))
            
            lidar_point = self.CTL @ homogeneous_point

            # Determine x value using average LiDAR depth within bounds 
            cylinder_x_range = []

            for i in range(len(self.ranges)):
                theta = ANGLE_INCREMENT * i
                # self.get_logger().info(f'Theta {i}:{theta}')
                r = self.ranges[i]

                y_dist = r * math.sin(theta)

                if (y_dist <= lidar_point[1,0] + CYLINDER_RADIUS) and (y_dist >= lidar_point[1,0] - CYLINDER_RADIUS) and r != 0 and ((theta >= 0 and theta <= np.radians(90)) or (theta >= np.radians(270) and theta <= np.radians(360))):
                    cylinder_x_range.append(np.abs(r*math.cos(theta)))

            average_x = np.mean(cylinder_x_range)
            goal_x_position = average_x - 2*CYLINDER_RADIUS

            lidar_point[0,0] = goal_x_position
            lidar_point[2,0] = 0.0

            # Convert lidar_point into PoseStamped format 
            lidar_point_msg = PointStamped()
            lidar_point_msg.header.frame_id = 'base_scan'
            lidar_point_msg.header.stamp = self.get_clock().now().to_msg()
            lidar_point_msg.point.x = lidar_point[0, 0]
            lidar_point_msg.point.y = lidar_point[1, 0]
            lidar_point_msg.point.z = lidar_point[2, 0]

            # Centroid point in PoseStamped
            centroid_point_msg = PointStamped()
            centroid_point_msg.header.frame_id = 'base_scan'
            centroid_point_msg.header.stamp = self.get_clock().now().to_msg()
            centroid_point_msg.point.x = lidar_point[0, 0] + 3*CYLINDER_RADIUS
            centroid_point_msg.point.y = lidar_point[1, 0]
            centroid_point_msg.point.z = lidar_point[2, 0]


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
            centroid_point = tf2_geometry_msgs.do_transform_point(centroid_point_msg, robot_transformation)
            transformed_vector = tf2_geometry_msgs.do_transform_vector3(movement_vector, robot_transformation)
            
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

            # Publish centroid pose in map frame
            centroid_pose = PoseStamped()

            centroid_pose.header.stamp = self.get_clock().now().to_msg()
            centroid_pose.header.frame_id = "map"  


            centroid_pose.pose.position.x = centroid_point.point.x
            centroid_pose.pose.position.y = centroid_point.point.y
            centroid_pose.pose.position.z = 0.0
            
            centroid_pose.pose.orientation = self.findQuarternion(yaw) 
            
            self.CentroidPub.publish(centroid_pose)
    
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