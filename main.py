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

from utils_video import gstreamerPipeline, readVideo, processImage, perspectiveWarp, plotHistogram
from utils_steering import steeringAngle, steeringText
from utils_lane_deceting import slide_window_search, general_search, measure_lane_curvature, draw_lane_lines, offCenter, addText
from utils_arduino import sendToArduino


################################################################################
######## START - MAIN FUNCTION #################################################
################################################################################

detection_err_count = 0

# 🍒 Read the input image
image = readVideo()

# 🍒 Read the arduino signal
try:
    servo = serial.Serial('COM12', 9600, timeout=1)
    time.sleep(1)
except:
    print("Error timeout arduino...")

################################################################################
#### START - LOOP TO PLAY THE INPUT IMAGE ######################################
while True:

    _, frame = image.read()
    try:
        # 🐸 birdView 적용하기
        # 1. "perspectiveWarp()" 함수를 호출하여 원근 변형 적용
        #       -> (birdView)라는 변수에 할당
        # 2- 원근 변형을 적용할 이미지(프레임)
        birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)

        # 🐸 birdView 가 적용된 이미지 불러오기
        # 1. "processImage()" 함수를 호출하여 이미지 처리 적용
        # 2. 각각의 변수(img, hls, grayscale, thresh, blur, canny)를 할당
        hls, grayscale, thresh, blur, canny = processImage(birdView)
        hlsL, grayscaleL, threshL, blurL, cannyL = processImage(
            birdViewL)
        hlsR, grayscaleR, threshR, blurR, cannyR = processImage(
            birdViewR)

        # 🐸 좌 / 우 차선 구별
        # 1. 밝기 값이 적용된 thresh 파일 불러오기
        # 2. "get_histogram()" 함수를 호출하여 히스토그램을 플롯하고 표시
        hist, leftBase, rightBase = plotHistogram(thresh)

        # 🐸 슬라이딩 윈도우 계산
        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(
            thresh, hist)

        draw_info = general_search(thresh, left_fit, right_fit)

        curveRad, curveDir = measure_lane_curvature(
            ploty, left_fitx, right_fitx)
        # plt.plot(hist)
        # plt.plot(left_fit)
        # plt.show()

        # 🐸 감지된 차선 영역을 파란색으로 채우기
        meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)
        deviation, directionDev = offCenter(meanPts, frame)

        # 🐸 차선 정보 추가
        finalImg, steeringWheelRadius = addText(
            result, curveRad, curveDir, deviation, directionDev)

        sendToArduino(servo, steeringWheelRadius)

        # 🐸 조향각 정보 Steering_GUI
        strDst, strDegrees = steeringAngle(steeringWheelRadius)
        steer = steeringText(strDst, strDegrees)

        # 🐸 최종 이미지 출력
        cv2.imshow("steering wheel", steer)
        cv2.imshow("Final", finalImg)
    except:
        detection_err_count += 1
        print("라인 검출 에러 카운트 : ", detection_err_count)

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
