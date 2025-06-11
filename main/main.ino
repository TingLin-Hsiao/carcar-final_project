#include <Servo.h>
#include "nail_list.h"
Servo myServo;

const int stepPin = 3;
const int dirPin = 4;
const int servo_pin = 9;

const int return_steps = 2;
const int nail_count = 200;  
const int ratio=400/nail_count;

int nail=0;

int angle_up = 97-10;                              //need to be measured
int angle_down = 83-15;

void setup() {
  Serial.begin(9600);
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

  if(clockwise) nail = (nail+steps/ratio+200)%200;
  if(!clockwise) nail = (nail-steps/ratio+200)%200;
  Serial.println(nail);
}

void walk(int distance, bool clockwise) {
  moveSteps(return_steps, !clockwise);
  delay(500);
  myServo.write(angle_up);
  delay(500);
  Serial.println("up");

  moveSteps(distance, clockwise);
  delay(500); 
  myServo.write(angle_down);    
  delay(500);
  Serial.println("down");

  
}

void loop() {
  Serial.println("Loop start");
  myServo.write(angle_down);
  delay(3000);
  int next_clockwise = 1; //1=clockwise, 0=counter
  int num = sizeof(nail_list) / sizeof(nail_list[0])-1;
  for(int i=0; i<num; i++){
    int distance =((nail_list[i+1] - nail_list[i]+nail_count)%nail_count)*ratio;
    if(((nail_list[i+2] - nail_list[i+1]+nail_count)%nail_count)*ratio<200) next_clockwise = 1;
    else next_clockwise = 0;
    
    if (distance<200){
      if(next_clockwise) walk(distance+1*ratio,1);
      if(!next_clockwise) walk(distance-0*ratio,1);
  
    }else{
      distance = 400 - distance;
      if(next_clockwise) walk(distance-0*ratio,0);
      if(!next_clockwise) walk(distance+1*ratio,0);
      
    }
    if(i==num-1){
      moveSteps(return_steps, 1);
      delay(500);
      myServo.write(angle_up);
      delay(500);
    }
  }
  while(1);
}





