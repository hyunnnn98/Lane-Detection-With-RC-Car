#include <Servo.h>

// <<-- SETUP
void setup()
{ 
  Serial.begin(9600);
  Serial.println("SETUP COMPLETE");
}
// -->>

void loop()
{  
  String sig;
  char check[1];
  
  char text_t[2] = {0, 0};
  char text_d[3] = {0, 0, 0};
  int value_t = 0;
  int value_d = 0;
  
  while(Serial.available())
  {
    char wait = Serial.read();
    Serial.print("Serial");
    Serial.println(wait);  
    sig.concat(wait);
  }

  Serial.println(sig);  
  sig.substring(0,1).toCharArray(check,2);

  if (check[0] == 'Q')
  {
    sig.substring(1, 3).toCharArray(text_t, 3);
    value_t = atoi(text_t);

    for (int i = 0; i < text_t.length(); i++)    // 배열의 요소 개수만큼 반복
    {
        printf("%d\n", text_t[i]);    // 배열의 인덱스에 반복문의 변수 i를 지정
    }

    sig.substring(3, 6).toCharArray(text_d, 4);
    value_d = atoi(text_d);

    for (int i = 0; i < text_d.length(); i++)    // 배열의 요소 개수만큼 반복
    {
        printf("%d\n", text_d[i]);    // 배열의 인덱스에 반복문의 변수 i를 지정
    }
  }
    
  delay(100);
}