#include <Servo.h>


Servo servo;
int moterPinNumber = 9;
int input_data;

void setup() {
  servo.attach(moterPinNumber);
  Serial.begin(9600);
  Serial.print("서보 모터 초기화..");
  servo.write(90);
}

void loop() {
  // put your main code here, to run repeatedly:
  while(Serial.available())
  {
    input_data = Serial.read();
  }

  // 직진
  if(input_data == '0')
  {
    moveAngle(90);
  }
  // 우회전
  else if(input_data == '1')
  {
    moveAngle(95);
  } else if (input_data == '2')
  {
    moveAngle(100);
  } else if (input_data == '3')
  {
    moveAngle(105);
  } else if (input_data == '4')
  {
    moveAngle(110);
  } else if (input_data == '5')
  {
    moveAngle(115);
  } 

}

void moveAngle(int argAngle) {
//  Serial.print(argAngle + "각도로 이동합니다...");
  servo.write(argAngle);
//  delay(500);
}
