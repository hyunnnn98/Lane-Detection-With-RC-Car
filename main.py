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

# IMPORT NECESSARY UTILS
from utils_video import *
from utils_lane_deceting import *
from utils_calibration import calib, undistort
from utils_steering import steeringAngle, steeringText
from utils_exception_handler import LaneFrame, exception_handler
from utils_arduino import sendToArduino, sendToEsc
from utils_constants import *


################################################################################
######## START - MAIN FUNCTION #################################################
################################################################################

# 🍒 Read the input image
image = readVideo()

# 🍒 camera matrix & distortion coefficient
mtx, dist = calib()

# 🍒 back up lane frame img
LaneFrame = LaneFrame()
preStrDegrees = 90

# 🍒 Read the arduino signal
try:
    servo = serial.Serial(ARDUINO_CONNECT_PORT, 9600, timeout=1)
    time.sleep(2)
except:
    print("❌ Error timeout arduino...")

################################################################################
#### START - LOOP TO PLAY THE INPUT IMAGE ######################################
while True:

    _, frame = image.read()
    try:
        # 🐸 camera calibration 적용하기
        # frame = cv2.resize(frame, (640, 360))
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
        is_error_lane_detected = exception_handler(
            left_fitx, right_fitx, curveRad)

        if CALIBRATION_MODE and is_error_lane_detected:

            # 🐢 차선 인식 실패에 따른 예외처리 알고리즘 시작
            if LaneFrame.checkBackedImg():
                CALIBRATION_COUNT += 1
                print("✅ 라인 보정 알고리즘 작동 :", CALIBRATION_COUNT)
                thresh, minverse, draw_info, curveRad, curveDir = LaneFrame.loadFrameData()

            else:
                print("❌ 백업된 라인 이미지가 없음")

        else:
            # 🐢 허용 오차 범위 안의 영상 데이터 백업
            LaneFrame.saveFrameData(
                thresh, minverse, draw_info, curveRad, curveDir)

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
        # print('🚙 조향 각도', strDegrees , '\n')

        # 🐸 아두이노 서보 모터로 데이터 전송
        try:
            if servo.readable():
                if preStrDegrees != strDegrees:
                    sendToArduino(servo, strDegrees)
                
                preStrDegrees = strDegrees
        except:
            print('❌ Arduino connection failed...')
            
            print('✅ 아두이노 연결중... 2초간 정지')
            servo = serial.Serial(ARDUINO_CONNECT_PORT, 9600, timeout=1)
            time.sleep(2)

        # 🐸 최종 이미지 출력
        cv2.imshow("steering wheel", steer)
        cv2.imshow("Final", finalImg)

        # cv2.waitKey(1000)
    except:
        DETECTION_ERR_COUNT += 1
        print("❌ 라인 검출 알고리즘 오류 :", DETECTION_ERR_COUNT)

    # Wait for the ENTER key to be pressed to stop playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(1) & 0xFF == ord('p'):
        # 🐸 stop Esc moter
        print("🐸 stop Esc moter")
        sendToEsc(servo, ESC_STOP_SIGNAL)
        
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # 🐸 start Esc moter
        print("🐸 start Esc moter")
        sendToEsc(servo, ESC_START_SIGNAL)

#### END - LOOP TO PLAY THE INPUT IMAGE ########################################
################################################################################

# Cleanup
servo.close()
image.release()
cv2.destroyAllWindows()

################################################################################
######## END - MAIN FUNCTION ###################################################
################################################################################
