#include <Servo.h>

Servo servo;
Servo ESC;
int ServorPinNumber = 7;
int ESCPinNumber = 9;


// 조향각 초기 값
int wheelDirection = 90;
// 속도 초기 값
int controlSpeed = 103;


void setup() {
  Serial.begin(9600);
  Serial.println("시리얼 통신 완료");
  ESC.attach(ESCPinNumber);
  servo.attach(ServorPinNumber);
  delay(1000);

  Serial.println("Enter number for control option:");
  Serial.println("0. START");
  Serial.println("1. Weak Left");
  Serial.println("2. STOP");
  Serial.println("3. Weak Right");
  Serial.println("4. Left");
  Serial.println("6. Right");
  Serial.println("+. INCREASE SPEED");
  Serial.println("-. DECREASE SPEED");
}

void loop() {
  char user_input;   
  
  while (Serial.available())
  {
    user_input = Serial.read();
    
    if(user_input == '0'){
      Serial.println("start car");
      controlSpeed = 103;
      ESC.write(controlSpeed);
    }
    else if(user_input == '1'){
      Serial.println("Weak Left");
      wheelDirection -= 5;
      moveAngle(wheelDirection);
      
    }
    else if(user_input == '2'){
      Serial.println("stop");
      controlSpeed = 100;
      ESC.write(controlSpeed);
      
    }
    else if(user_input == '3'){
      Serial.println("Weak Right");
      wheelDirection += 5;
      moveAngle(wheelDirection);
      
    }
    else if(user_input == '4'){
      Serial.println("Left");
      wheelDirection -= 10;
      moveAngle(wheelDirection);
      
    }
    else if(user_input == '6'){
      Serial.println("Right");
      wheelDirection += 10;
      moveAngle(wheelDirection);
      
    }
    else if(user_input == '+'){
      controlSpeed += 1;
      ESC.write(controlSpeed);
      Serial.println("Speed +: ");
      Serial.println(controlSpeed);
    }
    else if(user_input == '-'){
      controlSpeed -= 1;
      ESC.write(controlSpeed);
      Serial.println("Speed -: ");
      Serial.println(controlSpeed);
    }
    else
    {
      Serial.println("Invalid option entered.");
    }
  }
  
  
}


void moveAngle(int argAngle) {
//  Serial.print(argAngle + "move angle...");
  servo.write(argAngle);
  delay(1000);
}
