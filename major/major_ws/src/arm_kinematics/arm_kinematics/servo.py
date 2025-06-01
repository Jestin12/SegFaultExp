import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String, UInt32MultiArray
# from piservo import Servo
import time
import numpy as np



class ServoController(Node): 
        def __init__(self): 
                super().__init__("ServoController")


                # Assign GPIO pins 
                self.SHOULDERpin = 16
                self.ELBOWpin = 26
                self.HANDpin = 13
                self.ENABLEpin = 17

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

                # self.shoulder.stop() 
                # self.elbow.stop()
                # self.hand.ChangeDutyCycle(0)
                GPIO.output(self.ENABLEpin, GPIO.LOW)
                self.get_logger().info(f'pin 17 set to low')


                self.get_logger().info(f'joint{0}, duty-cycle: {self.joints_duty[0]})')
                self.get_logger().info(f'joint{1}, duty-cycle: {self.joints_duty[1]})')
                self.get_logger().info(f'joint{2}, duty-cycle: {self.joints_duty[2]})')

                # Create a subscriber to the joint angles 
                # self.AngleSub = self.create_subscription(String, '/joint_angles', self.angles_callback, 10)
                self.AngleSub = self.create_subscription(String, '/joint_signals', self.angles_callback, 10)

                # Create a publisher to robot status 
                self.status_publisher = self.create_publisher(String, '/robot_status', 10)


        def angles_callback(self, msg):

                self.get_logger().info(f'Received')

                JointDutyString = msg.data.split(',')
                JointDuty = [float(x) for x in JointDutyString]

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

                GPIO.output(self.ENABLEpin, GPIO.HIGH)
                self.get_logger().info(f'pin 17 set to high')

                order = list(range(2))
                order.reverse()

                self.shoulder.start(0)  
                self.elbow.start(0)
                self.hand.start(0)

                self.joints[2].ChangeDutyCycle(12)

                for i in order:

                        self.joints[i].ChangeDutyCycle(JointDuty[i])
                        self.get_logger().info(f'joint{i}, duty-cycle: {JointDuty[i]})')

                        self.get_logger().info(f'joint{i} done')

                        self.joints_duty[i] = JointDuty[i]

                        if self.joints_duty[i] == JointDuty[i]: 
                                self.get_logger().info("no change")

                self.get_logger().info(f"Waiting for arm to move")
                time.sleep(2)
                self.joints[2].ChangeDutyCycle(8)
                self.get_logger().info(f"Closing hand")
                time.sleep(2)

                self.shoulder.ChangeDutyCycle(18)
                self.elbow.ChangeDutyCycle(5)
                self.hand.ChangeDutyCycle(8)
                self.get_logger().info(f"Returning to home position")
                time.sleep(4)

                GPIO.output(self.ENABLEpin, GPIO.LOW)
                self.get_logger().info(f'pin 17 set to low')

                # self.shoulder.stop() 
                # self.elbow.stop()
                # self.hand.stop()
                # for i in range(3):
                #       time.sleep(1)
                #       self.get_logger().info(f"Waiting for arm to finish {i}")

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