#include <Servo.h>
Servo myServo;

const int stepPin = 3;
const int dirPin = 4;
const int servo_pin = 9;

int nail_list[] = {0,40,80,120,80,120,160};
int angle_up = 85;                              //need to be measured
int angle_down = 95;


void setup() {
  pinMode(stepPin,OUTPUT); 
  pinMode(dirPin,OUTPUT);
  myServo.attach(servo_pin);
}
void loop() {
  myServo.write(angle_up);
  delay(3000);
  int num = sizeof(nail_list) / sizeof(nail_list[0])-1;
  for(int i=0; i<num; i++){
    int distance =((nail_list[i+1] - nail_list[i]+1+200)%200)*2;
    digitalWrite(dirPin,LOW); // counterclockwise
    for(int x = 0; x < distance; x++) {
      digitalWrite(stepPin,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(stepPin,LOW); 
      delayMicroseconds(500); 
    }
    delay(2000); 
    myServo.write(angle_down);
    delay(1000);
    digitalWrite(dirPin,HIGH); // clockwise

    digitalWrite(stepPin,HIGH); 
    delayMicroseconds(500); 
    digitalWrite(stepPin,LOW); 
    delayMicroseconds(500); 
  
    delay(2000);
    myServo.write(angle_up);
    delay(1000);


    
  }
  delay(10000);
}





