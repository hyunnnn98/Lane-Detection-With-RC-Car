#include <Servo.h>


Servo servo;
int moterPinNumber = 11;
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

  if(input_data == '1')
  {
    moveAngle(10);
  }
  else if(input_data == '2')
  {
    moveAngle(20);
  } else if (input_data == '3')
  {
    moveAngle(30);
  } else if (input_data == '4')
  {
    moveAngle(40);
  } else if (input_data == '5')
  {
    moveAngle(50);
  } else if (input_data == '6')
  {
    moveAngle(60);
  } else if (input_data == '7')
  {
    moveAngle(70);
  } else if (input_data == '8')
  {
    moveAngle(80);
  }
}

void moveAngle(int argAngle) {
//  Serial.print(argAngle + "각도로 이동합니다...");
  servo.write(argAngle);
//  delay(500);
}
