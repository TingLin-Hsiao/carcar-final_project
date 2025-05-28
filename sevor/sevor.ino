#include <Servo.h>
Servo myServo;
const int servo_pin = 9;
int angle_middle = 90;
int angle_up = 85;                              //need to be measured
int angle_down = 95;

void setup() {
  myServo.attach(servo_pin);
}
void loop() {
  myServo.write(angle_middle);
  delay(2000);
  myServo.write(angle_up);
  delay(2000);
  myServo.write(angle_down);
  delay(2000);
}