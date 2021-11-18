################################################################################
######## START - 아두이노 조향각 연산을 처리하는 기능 ##########################
################################################################################

from utils_constants import *

def sendToEsc(signal):
    return signal.encode('utf-8')

def encodeValue(value):
    transVal = 'T' + str(value)
    print(transVal)
    encodeStr = transVal.encode('utf-8')
    return encodeStr

def weightAngleValue(steeringWheelRadius):
    weighted_angle_val = WEIGHT_VAL * steeringWheelRadius
    angle = CENTER_ANGLE + (weighted_angle_val)
    
    # convert to max and min servo degree
    if angle > MAX_ANGLE:
        angle = MAX_ANGLE
    elif angle < MIN_ANGLE :
        angle = MIN_ANGLE
    
    # print("MOVE SERVO ANGLE TO ", encodeValue(angle))
    return encodeValue(angle)
