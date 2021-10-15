"""
파이썬에서 on을 입력하면 불이 켜지고
off를 입력받으면 불이 꺼짐
"""
import serial
import time
import sys


try:
    servo = serial.Serial('COM11', 9600, timeout=1)
    time.sleep(1)
except:
    print("연결 실패...")
    sys.exit(0)

if (servo.readable()):
    # 아두이노 준비 상태 확인
    print(servo.readline().decode())


while (True):
    print("Choose: ", end='')
    state = input()

    if state == 'q' :
        break
    elif state == 'left':
        servo.write(b'left')
        print(servo.readline().decode())
    elif state == 'right':
        servo.write(b'right')
        print(servo.readline().decode())
    else:
        print("left or right 만 입력 가능함..!")

    # time.sleep(0.1)

servo.close()
