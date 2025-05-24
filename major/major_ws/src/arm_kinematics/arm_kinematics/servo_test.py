import RPi.GPIO as GPIO 
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from std_msgs.msg import String

from piservo import Servo
import time
import numpy as np

pin = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)

joint1 = Servo(pin, min_pulse=1.0, max_pulse=2.0, frequency=50)


theta = 0
joint1.write(theta)

time.sleep(1)
GPIO.cleanup()