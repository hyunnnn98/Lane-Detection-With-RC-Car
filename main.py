################################################################################
######## 차선 인식 프로젝트 ####################################################
################################################################################
# DESCRIPTION:  이 프로젝트는 차선 감지를 방법을 보여주기 위해 만들어졌습니다.
#               시스템은 전면 카메라가 장착된 RC카 자동차에서 작동합니다.
#               OPENCV 라이브러리의 도움으로 알고리즘을 설계할 수 있었습니다.
#               차선을 식별하고 앞바퀴의 조향각도를 예측할 수 있습니다.
#               RC카가 현재 차선에서 멀어지면 운전자에게 경고를 할 수 있습니다.
################################################################################

# IMPORT NECESSARY LIBRARIES
import cv2
import serial
import time

import numpy as np

# IMPORT NECESSARY UTILS
from utils_video import *
from utils_lane_deceting import *
from utils_calibration import calib, undistort
from utils_steering import steeringAngle, steeringText
from utils_arduino import sendToArduino
from utils_exception_handler import LaneFrame


def onMouse(x):
    pass

################################################################################
######## START - MAIN FUNCTION #################################################
################################################################################


DETECTION_ERR_COUNT = 0
CALIBRATION_COUNT = 0

# 🍒 Read the input image
image = readVideo()

# 🍒 camera matrix & distortion coefficient
mtx, dist = calib()

# 🍒 back up lane frame img
LaneFrame = LaneFrame()

# 🍒 Read the arduino signal
try:
    servo = serial.Serial('COM11', 9600, timeout=1)
    time.sleep(1)
except:
    print("Error timeout arduino...")

################################################################################
#### START - LOOP TO PLAY THE INPUT IMAGE ######################################
while True:

    _, frame = image.read()
    try:
        # 🐸 camera calibration 적용하기
        frame = undistort(frame, mtx, dist)

        # 🐸 birdView 적용하기
        # 1. "perspectiveWarp()" 함수를 호출하여 원근 변형 적용
        #       -> (birdView)라는 변수에 할당
        # 2- 원근 변형을 적용할 이미지(프레임)
        birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)

        # 🐸 birdView 가 적용된 이미지 불러오기
        # 1. "processImage()" 함수를 호출하여 이미지 처리 적용
        # 2. 각각의 변수(img, hls, grayscale, thresh, blur, canny)를 할당
        thresh = processImage(birdView)

        # 🐸 좌 / 우 차선 구별
        # 1. 밝기 값이 적용된 thresh 파일 불러오기
        # 2. "get_histogram()" 함수를 호출하여 히스토그램을 플롯하고 표시
        hist, leftBase, rightBase = plotHistogram(thresh)

        # 🐸 슬라이딩 윈도우 계산
        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(
            thresh, hist)

        # 🐸 차선 라인 평균값 도출
        draw_info = general_search(thresh, left_fit, right_fit)

        # 🐸 차선 곡률 반경 측정
        curveRad, curveDir = measure_lane_curvature(
            ploty, left_fitx, right_fitx)

        # 🐸 차선 인식 예외처리
        # 1. 예외 알고리즘 1 ) 오른쪽 mean - 왼쪽 mean == 250 정도.. ?
        # 2. 예외 알고리즘 2 ) 왼쪽은 100 ~ 300, 오른쪽은 1000 ~ 1200
        # or curveRad > 3000
        right_fit_x_avg = int(np.mean(right_fitx))
        left_fit_x_avg = int(np.mean(left_fitx))

        overed_lane_detected = right_fit_x_avg - left_fit_x_avg > 1000
        overed_lane_curveRad = curveRad > 3500
        left_lane_detected = left_fit_x_avg < 100 or left_fit_x_avg > 300
        right_lane_detected = right_fit_x_avg < 1000 or right_fit_x_avg > 1200

        if (left_lane_detected or right_lane_detected or overed_lane_detected or overed_lane_curveRad):
            # 🐢 차선 인식 실패에 따른 예외처리 알고리즘 시작

            if LaneFrame.checkBackedImg():
                CALIBRATION_COUNT += 1
                print("✅ 라인 보정 알고리즘 작동 : " , CALIBRATION_COUNT)
                thresh, minverse, draw_info, curveRad, curveDir = LaneFrame.loadFrameData()

            else:
                print("❌ 백업된 라인 이미지가 없음")

        else:
            # 🐢 허용 오차 범위 안의 영상 데이터 백업
            LaneFrame.saveFrameData(
                thresh, minverse, draw_info, curveRad, curveDir)
        # print(left_fit_x_avg, " : ", right_fit_x_avg)
        # if right_fit[0] > 3:
        #     print("오른쪽 차선 인식 불가!")

        # 🐸 감지된 차선 영역을 파란색으로 채우기
        meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)
        # print("편차 : ", int(meanPts[0][0][0]))
        deviation, directionDev = offCenter(meanPts, frame)

        # 🐸 차선 정보 추가
        finalImg, steeringWheelRadius = addText(
            result, curveRad, curveDir, deviation, directionDev)

        # 🐸 조향각 정보 Steering_GUI
        strDst, strDegrees = steeringAngle(steeringWheelRadius)
        steer = steeringText(strDst, strDegrees)

        # 🐸 아두이노 서보 모터로 데이터 전송
        # print(strDst, strDegrees)
        # sendToArduino(servo, strDegrees)

        # 🐸 최종 이미지 출력
        cv2.imshow("steering wheel", steer)
        cv2.imshow("Final", finalImg)

        # cv2.waitKey(1000)
    except:
        DETECTION_ERR_COUNT += 1
        print("라인 검출 에러 카운트 : ", DETECTION_ERR_COUNT)

    # Wait for the ENTER key to be pressed to stop playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#### END - LOOP TO PLAY THE INPUT IMAGE ########################################
################################################################################

# Cleanup
image.release()
cv2.destroyAllWindows()

################################################################################
######## END - MAIN FUNCTION ###################################################
################################################################################
