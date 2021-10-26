################################################################################
######## START - 아두이노 조향각 연산을 처리하는 기능 ##########################
################################################################################

# IMPORT NECESSARY LIBRARIES
def sendToArduino(servo, steeringWheelRadius):
    # print("MOVE SERVO ANGLE TO ", degree)
    if (steeringWheelRadius < 0) :
        signal = steeringWheelRadius * -1

    signal = str(steeringWheelRadius).encode('utf-8')
    servo.write(signal)
