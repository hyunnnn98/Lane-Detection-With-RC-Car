# IMPORT NECESSARY LIBRARIES
import cv2
import numpy as np

swheel = cv2.imread('images/steering_wheel_image.png', 0)
swheelRows, swheelCols = swheel.shape

smoothed_angle = 0
calibrate_degrees = 5

################################################################################
#### START - Steering_GUI ######################################################
def steeringAngle(degrees):
    degrees = degrees * calibrate_degrees

    global smoothed_angle

    if degrees > 35:
        degrees = 35
    elif degrees < -35:
        degrees = -35

    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
        degrees - smoothed_angle) / abs(degrees - smoothed_angle)

    M = cv2.getRotationMatrix2D(
        (swheelCols/2, swheelRows/2), -smoothed_angle, 1)
    dst = cv2.warpAffine(swheel, M, (swheelCols, swheelRows))
    degrees = round(smoothed_angle)

    return dst, degrees


def steeringText(img, angle):
    mask = np.zeros_like(img)
    cv2.putText(img, str(round(angle)), (100, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    res = cv2.addWeighted(img, 0.7, mask, 0.3, 0)

    return res
#### END - Steering_GUI ########################################################
################################################################################
