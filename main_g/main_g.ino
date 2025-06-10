#include <Servo.h>
Servo myServo;

const int stepPin = 3;
const int dirPin = 4;
const int servo_pin = 9;

const int return_steps = 2;
const int nail_count = 200;  

int nail_list[] = {};
//0 8 16 24 16 24 32
int angle_up = 95-13;                              //need to be measured
int angle_down = 85-13;


void setup() {
  pinMode(stepPin,OUTPUT); 
  pinMode(dirPin,OUTPUT);
  myServo.attach(servo_pin);
}

void moveSteps(int steps, bool clockwise) {
  digitalWrite(dirPin, clockwise ? HIGH : LOW);

  int ramp_steps = min(steps / 4, 50); // 起步與減速各佔步數的 1/4，最多 50 步
  int cruise_steps = steps - ramp_steps * 2;
  
  for (int i = 0; i < steps; i++) {
    int delay_us;
    
    // Acceleration phase
    if (i < ramp_steps) {
      delay_us = 2000 - (i * (1000 / ramp_steps));  // 從2000遞減到1000
    }
    // Deceleration phase
    else if (i >= (ramp_steps + cruise_steps)) {
      delay_us = 1000 + ((i - ramp_steps - cruise_steps) * (1000 / ramp_steps)); // 從1000增加到2000
    }
    // Constant speed
    else {
      delay_us = 1000;
    }

    if(steps==return_steps) delay_us = 2000;

    digitalWrite(stepPin, HIGH);
    delayMicroseconds(delay_us);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(delay_us);
  }
}


void loop() {
  myServo.write(angle_up);
  delay(3000);
  int num = sizeof(nail_list) / sizeof(nail_list[0])-1;
  for(int i=0; i<nail_count; i++){
    int distance =((nail_list[i+1] - nail_list[i]+1+nail_count)%nail_count)*2;
    distance = (67+1)*2;
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
  exit(0);
}





