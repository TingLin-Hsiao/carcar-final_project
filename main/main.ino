#include <Servo.h>
Servo myServo;

const int stepPin = 3;
const int dirPin = 4;
const int servo_pin = 9;

const int return_steps = 2;
const int nail_count = 200;  

int nail_list[] = {0,40,80,120,80,120,160,0,40,80,120,80,120,160,0,40,80,120,80,120,160};
//0 8 16 24 16 24 32
int angle_up = 85;                              //need to be measured
int angle_down = 95;


void setup() {
  pinMode(stepPin,OUTPUT); 
  pinMode(dirPin,OUTPUT);
  myServo.attach(servo_pin);
}

void moveSteps(int steps, bool clockwise) {
  digitalWrite(dirPin, clockwise ? HIGH : LOW);
  for (int x = 0; x < steps; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }
}


void loop() {
  myServo.write(angle_up);
  delay(3000);
  int num = sizeof(nail_list) / sizeof(nail_list[0])-1;
  for(int i=0; i<num; i++){
    int distance =((nail_list[i+1] - nail_list[i]+1+nail_count)%nail_count)*2;

    moveSteps(distance, true);
    delay(500); 
    myServo.write(angle_down);
    delay(500);

    moveSteps(return_steps, false);
    delay(500);
    myServo.write(angle_up);
    delay(500);


    
  }
  delay(10000);
}





