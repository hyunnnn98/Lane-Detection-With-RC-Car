"""
파이썬에서 on을 입력하면 불이 켜지고
off를 입력받으면 불이 꺼짐
"""
import serial
import time
import sys

def sendToArduino(degree):
    print("MOVE SERVO ANGLE TO ", degree)

    degree = degree.encode('utf-8')
    servo.write(val)
    time.sleep(0.5)   


try:
    servo = serial.Serial('COM12', 9600, timeout=1)
    time.sleep(1)
except:
    print("연결 실패...")
    sys.exit(0)

while True:
    if servo.readable():
        val = input()

        sendToArduino(val)
