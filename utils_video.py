################################################################################
######## START - FUNCTIONS TO PERFORM IMAGE PROCESSING #########################
######## START - 이미지 처리를 수행하는 기능 ###################################
################################################################################

# IMPORT NECESSARY LIBRARIES
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt, cm, colors

# Get path to the current working directory
# 현재 작업 디렉토리의 경로 가져오기
CWD_PATH = os.getcwd()

test_video = 'tracks/real_camera_3.mp4'

################################################################################
#### START - FUNCTION TO READ AN RSP CAMERA VIDEO #############################


def gstreamerPipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
#### END - FUNCTION TO READ AN RSP CAMERA VIDEO ################################
################################################################################


################################################################################
#### START - FUNCTION TO READ AN INPUT IMAGE ###################################
def readVideo():

    # Read input video from current working directory
    inpImage = cv2.VideoCapture(os.path.join(CWD_PATH, test_video))

    return inpImage
#### END - FUNCTION TO READ AN INPUT IMAGE #####################################
################################################################################


################################################################################
#### START - FUNCTION TO PROCESS IMAGE #########################################
def processImage(inpImage, custom_green, custon_red, custom_white, custom_white_row, custom_thresh):
    # 그림자 영역 제거
    # green_threshold = custom_green
    # thresholds = (inpImage[:, :, 1] < green_threshold) | (
    #     inpImage[:, :, 2] < custon_red)
    # inpImage[thresholds] = [0, 0, 0]

    # Apply HLS color filtering to filter out white lane lines
    # ( 흰색 영역 HSL 필터 )
    # hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HSV)

    # default = 0, 130, 10
    # lower_white = np.array([0, custom_white, custom_white_row])
    # upper_white = np.array([255, 255, 255])     # default = 255, 255, 255
    # mask = cv2.inRange(inpImage, lower_white, upper_white)

    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(inpImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, custom_thresh, 255, cv2.THRESH_BINARY)      # default = 160, 255 ( 🍒 흰 or 검... 범위 설정)
    # 🎃 adaptiveThreshold 백업..
    # th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             cv2.THRESH_BINARY, 15, 2)
    th2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    blur = cv2.GaussianBlur(th2, (3, 3), 0)
    hls_result = cv2.bitwise_not(blur)
    canny = cv2.Canny(blur, 60, 180)             # default = 40, 60


    # Display the processed images
    # cv2.imshow("Image", inpImage)
    # cv2.imshow("HLS Filtered", cv2.flip(hls_result, 0))
    cv2.imshow("th2", cv2.flip(canny, 0))
    # plt.imshow(cv2.flip(blur, 0))
    print(cv2.flip(blur, 0)[700][600])
    # plt.show()
    # cv2.imshow("Thresholded", thresh)
    # cv2.imshow("Blurred", blur)
    # cv2.imshow("Canny Edges", canny)

    return hls_result, gray, hls_result, blur, canny
#### END - FUNCTION TO PROCESS IMAGE ###########################################
################################################################################


################################################################################
#### START - FUNCTION TO APPLY PERSPECTIVE WARP ################################
def perspectiveWarp(inpImage):

    # 이미지 사이즈 추출
    height, width = inpImage.shape[:2]
    img_size = (width, height)

    # Perspective points to be warped
    # 왜곡될 관점 포인트 <좌하, 우하, 좌상, 우상>
    # 🐸 ( Type 1 )
    # src = np.float32([[0, height-150],
    #                   [width, height-150],
    #                   [200, 300],
    #                   [width-200, 300]])

    # 🐸 ( Type 2 기본값 )
    # src = np.float32([[200, height-200],
    #                   [width, height-200],
    #                   [200, 400],
    #                   [width, 400]])
    # (200, 520)  (1280, 520)  (200, 400)  (1280, 400)

    # 🐸 ( Type 3 넓은 시야 )
    # src = np.float32([[0, height-150],
    #                   [width, height-150],
    #                   [400, 150],
    #                   [width-400, 150]])

    # 🐸 ( real_camera_2 Type 1 넓은 시야 )
    src = np.float32([[0, 580],
                      [width-90, 580],
                      [170, 360],
                      [905, 360]])
    
    # src = np.float32([[0, 580],             #좌하
    #                   [width-90, 580],      #우하
    #                   [0, 300],           #좌상
    #                   [width, 300]])  # 우상

    # Window to be shown
    # 표시할 윈도우
    dst = np.float32([[0, 0],         # 480
                      [1280, 0],         # 800
                      [0, 720],
                      [1280, 720]])

    # Matrix to warp the image for birdseye window
    # Birdseye 창의 이미지를 왜곡하는 매트릭스
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Inverse matrix to unwarp the image for final window
    # 최종 창의 이미지를 왜곡 해제하는 역행렬
    minv = cv2.getPerspectiveTransform(dst, src)
    birdseye = cv2.warpPerspective(inpImage, matrix, img_size)

    # Get the birdseye window dimensions
    # Birdseye 창 크기 가져오기
    height, width = birdseye.shape[:2]

    # Divide the birdseye view into 2 halves to separate left & right lanes
    # Birdseye 뷰를 2등분하여 왼쪽 및 오른쪽 차선 분리
    birdseyeLeft = birdseye[0:height, 0:width // 2]
    birdseyeRight = birdseye[0:height, width // 2:width]

    # Display birdseye view image
    # Birdseye 뷰이미지 표시

    # cv2.imshow("Birdseye", birdseye)
    # cv2.imshow("Birdseye Left", birdseyeLeft)
    # cv2.imshow("Birdseye Right", birdseyeRight)

    return birdseye, birdseyeLeft, birdseyeRight, minv
#### END - FUNCTION TO APPLY PERSPECTIVE WARP ##################################
################################################################################


################################################################################
#### START - 왜곡된 이미지의 히스토그램을 플롯하는 기능 ########################
def plotHistogram(inpImage):

    histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis=0)

    midpoint = np.int32(histogram.shape[0] / 2)
    leftxBase = np.argmax(histogram[:midpoint])
    rightxBase = np.argmax(histogram[midpoint:]) + midpoint

    plt.xlabel("Image X Coordinates")
    plt.ylabel("Number of White Pixels")

    # 픽셀 단위의 계산식에 필요한
    # ( 왼쪽, 오른쪽 차선의 히스토그램 ) 및 ( x 좌표 ) 반환
    return histogram, leftxBase, rightxBase
#### END - FUNCTION TO PLOT THE HISTOGRAM OF WARPED IMAGE ######################
################################################################################
