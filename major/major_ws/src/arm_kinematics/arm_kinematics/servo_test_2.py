import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

shoulder_pin =16
elbow_pin =26
hand_pin =13
GPIO.setup(shoulder_pin, GPIO.OUT)
GPIO.setup(elbow_pin, GPIO.OUT)
GPIO.setup(hand_pin, GPIO.OUT)


shoulder = GPIO.PWM(shoulder_pin,333)
elbow = GPIO.PWM(elbow_pin,50)
hand = GPIO.PWM(hand_pin,50)

shoulder.start(0)  
elbow.start(0)
hand.start(0)


try:
    while True:

        joint, duty = input("Please enter a joint and duty cycle (a,10):").split(",")
        print(joint)
        duty = float(duty)
        if (joint):
            match joint: 
                case "s":
                    print("shoulder")
                    if duty > 63:
                        duty = 63
                    elif duty < 17:
                        duty = 17
                    print("duty")
                                  
                    shoulder.ChangeDutyCycle(float(duty)) 

                case "e":
                    if duty > 12.5:
                        duty = 12.5
                    elif duty < 5:
                        duty = 5

                    elbow.ChangeDutyCycle(float(duty)) 

                case "h":
                    if duty > 12:
                        duty = 12
                    elif duty < 8:
                        duty = 8

                    hand.ChangeDutyCycle(float(duty)) 

        sleep(1)



        # .ChangeDutyCycle(float(ameline))   

        #motor_pwm.ChangeDutyCycle(7.0)  
        #sleep(1)

        #motor_pwm.ChangeDutyCycle(9.5) 
        #sleep(1)

except KeyboardInterrupt:
    pass

shoulder.stop()
elbow.stop()
hand.stop()
GPIO.cleanup()

