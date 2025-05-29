#include <Servo.h>

Servo servo2, servo3;
const int servoPin1 = 11; // 333Hz servo (Timer1, OC1A)
const int servoPin2 = 2;  // 50Hz servo (Timer3, OC3B)
const int servoPin3 = 6;  // 50Hz servo (Timer4, OC4A)

void setup() {
    // Configure Timer1 for 333Hz PWM (Servo 1)
    pinMode(servoPin1, OUTPUT);
    TCCR1A = 0;  // Clear Timer1 control registers
    TCCR1B = 0;
    TCCR1A |= (1 << WGM11);              // Fast PWM, ICR1 as TOP
    TCCR1B |= (1 << WGM12) | (1 << WGM13);
    TCCR1B |= (1 << CS11);               // Prescaler = 8
    ICR1 = 6000;                         // TOP for 333Hz (16MHz / 8 / 333 ≈ 6000)
    TCCR1A |= (1 << COM1A1);             // Non-inverted PWM on pin 11
    OCR1A = 300;                         // Initial 1.5ms pulse (~90°, 5% duty)

    // Configure Servo library for 50Hz servos
    servo2.attach(servoPin2);  // 50Hz on pin 2
    servo3.attach(servoPin3);  // 50Hz on pin 6
    servo2.write(90);         // Initial position
    servo3.write(90);

    Serial.begin(9600);  // Start serial for angle input
}

void loop() {
    if (Serial.available() > 0) {
        // Expect format: "servo_id,angle\n" (e.g., "1,90\n")
        int servoId = Serial.parseInt();
        int angle = Serial.parseInt();
        if (Serial.read() == '\n' && angle >= 0 && angle <= 180) {
            if (servoId == 1) {
                // 333Hz servo: 0.5ms–2.5ms = 300–1500 counts
                OCR1A = 300 + (angle / 180.0) * 1200;
                Serial.print("Servo 1 (333Hz) set to ");
                Serial.println(angle);
            } else if (servoId == 2) {
                servo2.write(angle);
                Serial.print("Servo 2 (50Hz) set to ");
                Serial.println(angle);
            } else if (servoId == 3) {
                servo3.write(angle);
                Serial.print("Servo 3 (50Hz) set to ");
                Serial.println(angle);
            }
        }
        while (Serial.available() > 0) {
            Serial.read();
        }
    }
}
