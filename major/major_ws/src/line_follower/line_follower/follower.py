import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String
from sensor_msgs.msg import Imu
import math
import time

import os

'''
Package:		line_follower
File:			follower.py

Description:	
				This 
                                
Dependencies:

'''

class Followeralt(Node): 
        def __init__(self): 
                super().__init__("Follower")

                # Set pin numbers for IR sensors 
                self.LEFT = 19
                self.RIGHT = 25

                GPIO.setmode(GPIO.BCM)
                GPIO.setup([self.LEFT, self.RIGHT], GPIO.IN)


                # Initialise variables 
                self.movement_command = 1
                self.current_yaw = 0.0
                self.target_yaw = None
                self.kp = 1.0
                self.memory = "straight"
                self.vel_msg_copy = Twist()
                self.turn_count = 0

                # Create timer
                self.timer = self.create_timer(0.05, self.check_sensors)

                # Create publishers 
                self.vel_publisher = self.create_publisher(Twist, "/LineFollowerVel", 10)
                self.IR_publisher = self.create_publisher(String, "/IR_value", 10)
                # self.move_publisher = self.create_publisher(String, "/move_forward", 10)

                # Create subscribers 
                self.detection_subscriber = self.create_subscription(String, '/plant_detection', self.plant_check, 10)
                self.servo_subscriber = self.create_subscription(String, '/robot_status', self.servo_motion, 10)
                

        def rotate_90_degrees(self):
                linear_velocity = 0.025
                angular_velocity = -0.224

                angle = math.pi / 2
                time_to_rotate = angle / abs(angular_velocity)

                vel_msg = Twist()
                vel_msg.linear.x = linear_velocity
                vel_msg.angular.z = angular_velocity
                self.vel_msg_copy = vel_msg
                self.vel_publisher.publish(vel_msg)

                time.sleep(time_to_rotate * 0.25)

                vel_msg.angular.z = 0.0
                self.vel_msg_copy = vel_msg
                self.vel_publisher.publish(vel_msg)


        def check_sensors(self): 

                # Extracting the values from the GPIO pins 
                left_value = GPIO.input(self.LEFT)
                right_value = GPIO.input(self.RIGHT)

                # Publish IR values 
                ir_msg = String()
                # ir_msg.data = f"Left: {left_value}, Centre: {centre_value}, Right: {right_value}"
                ir_msg.data = f"Left: {left_value}, Right: {right_value}"
                self.IR_publisher.publish(ir_msg)
                
                vel_msg = Twist()
                vel_msg = self.vel_msg_copy


                # Check if turtlebot is conducting detection or arm kinematics process 
                if (self.movement_command) == 1: 

                        # STRAIGHT
                        if left_value == 1 and right_value == 1:
                                vel_msg.linear.x = 0.05


                        # ROTATE RIGHT
                        elif left_value == 0 and right_value == 1: 
                                # Reset yaw and begin 
                                # change based on size of the arc
                                vel_msg.linear.x = 0.05
                                vel_msg.angular.z = -0.05

                        # ROTATE LEFT 
                        elif left_value == 1 and right_value == 0: 
                                # change based on size of the arc
                                vel_msg.linear.x = 0.05
                                vel_msg.angular.z = 0.05

                        # Stop when you detect a plant
                        elif left_value == 0 and right_value == 0:
                                self.rotate_90_degrees()

                        else: 
                                vel_msg.linear.x = 0.0
                                vel_msg.angular.z = 0.0

                else: 
                        vel_msg.linear.x = 0.0
                        vel_msg.angular.z = 0.0 


                self.vel_msg_copy = vel_msg

                # Publish movement command
                self.vel_publisher.publish(vel_msg)


        def servo_motion(self, msg): 


                if msg.data == "DONE": 
                        self.get_logger().info("Recieved DONE command")
                        self.movement_command = 1
                else: 
                        self.movement_command = 0


        def plant_check(self, msg): 

                split_message = msg.data.split(" ")

                if split_message[0] == "Unhealthy": 
                        self.movement_command = 0
                        # self.move_publisher.publish(msg)
                else: 
                        self.movement_command = 1
                        
						
def main(args=None): 
        rclpy.init(args=args)
        node = Followeralt()
        rclpy.spin(node)
        rclpy.shutdown()