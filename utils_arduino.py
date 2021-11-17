################################################################################
######## START - 아두이노 조향각 연산을 처리하는 기능 ##########################
################################################################################

# 제어 값
center_angle = 90
weight_val = 5

MAX_ANGLE = 120
MIN_ANGLE = 60

def sendToEsc(servo, signal):
    return servo.write(signal.encode('utf-8'))

def encodeValue(value):
    transVal = 'T' + str(value)
    print(transVal)
    encodeStr = transVal.encode('utf-8')
    return encodeStr

def sendToArduino(servo, steeringWheelRadius):
    weighted_angle_val = weight_val * steeringWheelRadius
    angle = center_angle + (weighted_angle_val)
    
    # convert to max and min servo degree
    if angle > MAX_ANGLE:
        angle = MAX_ANGLE
    elif angle < MIN_ANGLE :
        angle = MIN_ANGLE
    
    # print("MOVE SERVO ANGLE TO ", encodeValue(angle))
    servo.write(encodeValue(angle))
