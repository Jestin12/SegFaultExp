import RPi.GPIO as GPIO
import time
# from cServo import cServo


class Servo:
    def __init__(self, pin, frequency, max_pulse_ms, min_pulse_ms):
        self.pin = pin
        self.frequency = frequency      # Hz
        self.max_pulse_ms = max_pulse_ms    #milliseconds
        self.min_pulse_ms = min_pulse_ms    #milliseconds
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

        self.pwm = GPIO.PWM(self.pin, self.frequency)
        self.period_ms = 1000.0 / self.frequency  # Period in milliseconds
        self.pwm.start(0)
        self.CurrAngle = None



    def set_servo_angle(self, angle):

        self.CurrAngle = angle

        duty = (self.min_pulse_ms + (angle / 180.0) * (self.max_pulse_ms - self.min_pulse_ms)) / self.period_ms * 100.0
        self.pwm.ChangeDutyCycle(duty)
