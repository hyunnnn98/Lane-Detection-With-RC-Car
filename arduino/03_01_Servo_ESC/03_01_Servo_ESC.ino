#include <Servo.h>

Servo servo;
Servo ESC;
int ServorPinNumber = 7;
int ESCPinNumber = 9;
bool ESC_Start = false;

void setup() {
  Serial.begin(9600);
  Serial.println("시리얼 통신 완료");
  // ESC.attach(ESCPinNumber);
  servo.attach(ServorPinNumber);
  delay(2000);
}

void loop() {
  String sig;
  char check[1];
  char text_d[3] = {0, 0, 0};
  int value_d = 0;

  // 🐢 파이썬으로 부터 값 받아오기
  while(Serial.available())
  {
    char wait = Serial.read();
    sig.concat(wait);
    Serial.println(sig);
  }
  
  sig.substring(0,1).toCharArray(check,2);
  
  //  if (check[0] == 'S'){
  //    ESC_Start = true;
  //  }
  //  if (check[0] == 'P'){
  //    ESC_Start = false;
  //  }
  
  if (check[0] == 'T' )
  {
    sig.substring(1, 4).toCharArray(text_d, 4);
    
    value_d = atoi(text_d);
    
    moveAngle(value_d);
  }

  //  if(ESC_Start == true) {
  //    ESC.write(103);
  //    Serial.println("Start");
  //  }else{
  //    ESC.write(99);
  //    Serial.println("Stop");
  //  }

  if(sizeof(sig) > 3){
    Serial.println(sig);
  }
  
   delay(300);
}


void moveAngle(int argAngle) {
  Serial.println("move angle");
  Serial.println(argAngle);
  servo.write(argAngle);
}
