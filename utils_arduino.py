################################################################################
######## START - 아두이노 조향각 연산을 처리하는 기능 ##########################
################################################################################

def sendToArduino(servo, steeringWheelRadius):
    # print("MOVE SERVO ANGLE TO ", degree)
    signal = str(steeringWheelRadius).encode('utf-8')
    servo.write(signal)
