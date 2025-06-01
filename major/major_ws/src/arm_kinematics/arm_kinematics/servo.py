import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String, UInt32MultiArray
# from piservo import Servo
import time
import numpy as np

'''
Package: 	arm_kinematics
File: 		servo.py
Author: 	Jestin

Description:
			This file defines and runs a ROS2 node which interacts with the
            GPIO pins of a raspberry pi 3 to control the servo motors of a 
            robotic arm. The node receives duty cycles from the /joint_signals 
            topic and writes those duty cycles to the shoulder and elbow servo 
            motors. When the position defined by those duty cycles are reached 
        	it opens and closes the hand servo (the gripper) is achieved it
            then returns the arm back to its collapsed position.
            
Dependencies:
			RPi.GPIO	rclpy	geometry_msgs	std_msgs	time	numpy
'''

class ServoController(Node): 
        def __init__(self): 
                super().__init__("ServoController")


                # Assign GPIO pins 
                self.SHOULDERpin = 16
                self.ELBOWpin = 26
                self.HANDpin = 13
                self.ENABLEpin = 17


				#	Setup GPIO pins for pwm function
                GPIO.setmode(GPIO.BCM)

                GPIO.setwarnings(False)

                GPIO.setup(self.SHOULDERpin, GPIO.OUT)
                GPIO.setup(self.ELBOWpin, GPIO.OUT)
                GPIO.setup(self.HANDpin, GPIO.OUT)
                GPIO.setup(self.ENABLEpin, GPIO.OUT)


                GPIO.output(self.ENABLEpin, GPIO.HIGH)
                self.get_logger().info(f'pin 17 set to high')
                self.shoulder = GPIO.PWM(self.SHOULDERpin, 333)
                self.elbow = GPIO.PWM(self.ELBOWpin, 50)
                self.hand = GPIO.PWM(self.HANDpin, 50)

                self.joints = [self.shoulder, self.elbow, self.hand]


				# Initialise PWM pins and set the arm to its collapsed
                # "home" position
                self.shoulder.start(0)  
                self.elbow.start(0)
                self.hand.start(0)

                self.shoulder_duty = 18
                self.elbow_duty = 5
                self.hand_duty = 8
                self.joints_duty = [self.shoulder_duty, self.elbow_duty, self.hand_duty]

                self.shoulder.ChangeDutyCycle(self.shoulder_duty)
                self.elbow.ChangeDutyCycle(self.elbow_duty)
                self.hand.ChangeDutyCycle(self.hand_duty)

                time.sleep(2)

                GPIO.output(self.ENABLEpin, GPIO.LOW)
                self.get_logger().info(f'pin 17 set to low')


                self.get_logger().info(f'joint{0}, duty-cycle: {self.joints_duty[0]})')
                self.get_logger().info(f'joint{1}, duty-cycle: {self.joints_duty[1]})')
                self.get_logger().info(f'joint{2}, duty-cycle: {self.joints_duty[2]})')

                # Create a subscriber to the joint angles 
                self.AngleSub = self.create_subscription(String, '/joint_signals', self.angles_callback, 10)

                # Create a publisher to robot status 
                self.status_publisher = self.create_publisher(String, '/robot_status', 10)


        def angles_callback(self, msg):

				
                self.get_logger().info(f'Received')

				# extracting data from topic message
                JointDutyString = msg.data.split(',')
                JointDuty = [float(x) for x in JointDutyString]

				# logic to ensure the duty cycles aren't beyond a certain range
                # as to not break the servos
                
                if isinstance(JointDuty[0], float):             
                        if JointDuty[0] > 63:
                                JointDuty[0] = 63
                        elif JointDuty[0] < 17:
                                JointDuty[0] = 17

                self.get_logger().info(f"Base duty: {JointDuty[0]}")
                if isinstance(JointDuty[1], float):
                        if JointDuty[1] > 12.5:
                                JointDuty[1] = 12.5
                        elif JointDuty[1] < 5:
                                JointDuty[1] = 5

                self.get_logger().info(f"Elbow duty: {JointDuty[1]}")


                # if isinstance(JointDuty[2], float):
                #       if JointDuty[2] > 12:
                #               JointDuty[2] = 12
                #       elif JointDuty[2] < 8:
                #               JointDuty[2] = 8

                # self.get_logger().info(f"Hand duty: {JointDuty[2]}")

				# power on the relay to then power on the servos
                GPIO.output(self.ENABLEpin, GPIO.HIGH)
                self.get_logger().info(f'pin 17 set to high')

                order = list(range(2))
                order.reverse()

                self.shoulder.start(0)  
                self.elbow.start(0)
                self.hand.start(0)

				# open the gripper
                self.joints[2].ChangeDutyCycle(12)


				# write the duty cycles to the shoulder and elbow joints; moving the arm
                for i in order:

                        self.joints[i].ChangeDutyCycle(JointDuty[i])
                        self.get_logger().info(f'joint{i}, duty-cycle: {JointDuty[i]})')

                        self.get_logger().info(f'joint{i} done')

                        self.joints_duty[i] = JointDuty[i]

                        if self.joints_duty[i] == JointDuty[i]: 
                                self.get_logger().info("no change")

                self.get_logger().info(f"Waiting for arm to move")
                time.sleep(2)
                
				# closing the gripper
                self.joints[2].ChangeDutyCycle(8)
                self.get_logger().info(f"Closing hand")
                time.sleep(2)
                
				# returning the arm to its collapsed home position
                self.shoulder.ChangeDutyCycle(18)
                self.elbow.ChangeDutyCycle(5)
                self.hand.ChangeDutyCycle(8)
                self.get_logger().info(f"Returning to home position")
                time.sleep(4)

				# turning off the relay, powering off the servos
                GPIO.output(self.ENABLEpin, GPIO.LOW)
                self.get_logger().info(f'pin 17 set to low')

				# sending message to trigger line following
                # Start line following 
                status_msg = String() 
                status_msg.data = "DONE"
                self.status_publisher.publish(status_msg)



def main(args=None):
    rclpy.init(args=args)
    node = ServoController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()