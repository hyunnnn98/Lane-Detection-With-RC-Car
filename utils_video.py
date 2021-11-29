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

test_video = 'tracks/커브.mp4'

################################################################################
#### START - FUNCTION TO READ AN RSP CAMERA VIDEO #############################
def gstreamerPipeline(
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=25,
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
    # inpImage = cv2.VideoCapture(gstreamerPipeline(), cv2.CAP_GSTREAMER)
    
    

    return inpImage
#### END - FUNCTION TO READ AN INPUT IMAGE #####################################
################################################################################


################################################################################
#### START - FUNCTION TO PROCESS IMAGE #########################################
def processImage(inpImage):
    # 🎃 adaptiveThreshold 백업..
    # th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             cv2.THRESH_BINARY, 15, 2)
    
    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(inpImage, cv2.COLOR_BGR2GRAY)
    
    # threshold 평균치 계산
    avg_thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    
    # 이미지 블러 처리
    blur = cv2.GaussianBlur(avg_thr, (3, 3), 0)
    
    thresh_result = cv2.bitwise_not(blur)
    
    # canny = cv2.Canny(blur, 60, 180)             # default = 40, 60
    # cv2.imshow("Thresholded", thresh)
    # cv2.imshow("Blurred", blur)
    # cv2.imshow("Canny Edges", canny)

    return thresh_result
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
    src = np.float32([[0, 240],
                      [width-70, 240],
                      [85, 150],
                      [430, 150]])
    
    # src = np.float32([[0, 580],             # 좌하
    #                   [width-90, 580],      # 우하
    #                   [0, 300],             # 좌상
    #                   [width, 300]])        # 우상

    # Window to be shown
    # 표시할 윈도우
    dst = np.float32([[0, 0],         # 480
                      [640, 0],         # 800
                      [0, 360],
                      [640, 360]])

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

def onMouse(x):
    pass
