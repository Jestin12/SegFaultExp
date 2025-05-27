import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)

motor_pin = 13  
GPIO.setup(motor_pin, GPIO.OUT)

motor_pwm = GPIO.PWM(motor_pin, 50)
motor_pwm.start(0)  

try:
    while True:
     
        motor_pwm.ChangeDutyCycle(7.5)   
        sleep(1)

        motor_pwm.ChangeDutyCycle(10.0)  
        sleep(1)

        motor_pwm.ChangeDutyCycle(12.5) 
        sleep(1)

except KeyboardInterrupt:
    pass

motor_pwm.stop()
GPIO.cleanup()
