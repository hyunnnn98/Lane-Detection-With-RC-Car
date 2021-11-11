#include <Servo.h>

Servo servo;
int moterPinNumber = 11;

// 제어 값
int normal_angle = 90;
int weight_val = 5;
int input_data = 0;

void setup() {
  Serial.print("🐢 Servo motor initialization...");
  servo.attach(moterPinNumber);
  Serial.begin(9600);
  servo.write(normal_angle);
}

void loop() {
  while(Serial.available())
  {
    input_val = atof(Serial.readString());
  }

  // 양수 or 음수 구분 -> bool 값
  bool is_positive_num = input_val >= 0
  int weighted_angle_val = weight_val * input_val

  // 양수이던 음수이던 해당 val + or - (가중치 * val)
  if (is_positive_num) {
    moveAngle(normal_angle + (weighted_angle_val))
  } else {
    moveAngle(normal_angle - (weighted_angle_val))
  }

}

void moveAngle(int argAngle) {
  Serial.print("🐢 Move to " + argAngle);
  servo.write(argAngle);
}
