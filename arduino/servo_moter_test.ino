#include <Servo.h>


Servo servo;
int moterPinNumber = 11;
String state;

void setup() {
  servo.attach(moterPinNumber);
  Serial.begin(9600);
  Serial.print("서보 모터 초기화..");
  servo.write(90);
}

void loop() {
  if (Serial.available()) {
    state = Serial.read();  

//    while (Serial.available()) {
//      Serial.read();
//    }

    if (state = "left") {
      moveAngle(0);
      Serial.print("left...");
    } else if (state = "right") {
      moveAngle(90);
      Serial.print("right...");       
    }
  }

//  delay(100);
//  for (int i = 0; i <= 180; i++) {
//    moveAngle(i);
//  }
//  moveAngle(0);
//  moveAngle(10);
//  moveAngle(20);
//  moveAngle(30);
//  moveAngle(40);
//  moveAngle(50);
//  moveAngle(60);
//  moveAngle(70);
//  moveAngle(80);
//  moveAngle(90);
//  moveAngle(100);
//  moveAngle(110);
//  moveAngle(120);
//  moveAngle(130);
//  moveAngle(140);
//  moveAngle(150);
//  moveAngle(160);
//  moveAngle(170);
//  moveAngle(180);
//  while (Serial.available() > 0) {
//    int angle = Serial.read();
//    moveAngle(angle);
//
//  }
}

void moveAngle(int argAngle) {
  Serial.print(argAngle + "각도로 이동합니다...");
  servo.write(argAngle);
//  delay(500);
}
