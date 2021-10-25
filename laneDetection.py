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
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

test_video = 'real_camera.mp4'
# test_video = 'real_camera_speed.mp4'

# Defining variables to hold meter-to-pixel conversion
# meter-to-pixel 변환을 유지하기 위한 변수 정의
ym_per_pix = 30 / 720

# Standard lane width is 3.7 meters divided by lane width in pixels which is
# calculated to be approximately 720 pixels not to be confused with frame height
# 표준 차선 너비 3.7m를 차선 너비(픽셀)로 나눈 값으로
# 계산할때는 프레임의 높이와 혼동하지 않기 위해 약 720 픽셀로 계산
xm_per_pix = 12 / 720  # ( 기본 값 : 3.7m )

# Get path to the current working directory
# 현재 작업 디렉토리의 경로 가져오기
CWD_PATH = os.getcwd()

################################################################################
######## START - FUNCTIONS TO PERFORM IMAGE PROCESSING #########################
######## START - 이미지 처리를 수행하는 기능 ###################################
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
def processImage(inpImage):

    # Apply HLS color filtering to filter out white lane lines
    # ( 흰색 영역 HSL 필터 )
    hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)

    lower_white = np.array([0, 100, 10])         # default = 0, 130, 10
    upper_white = np.array([255, 255, 255])      # default = 255, 255, 255
    mask = cv2.inRange(inpImage, lower_white, upper_white)
    hls_result = cv2.bitwise_and(inpImage, inpImage, mask=mask)

    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    canny = cv2.Canny(blur, 40, 120)            # default = 40, 60

    # Display the processed images
    # cv2.imshow("Image", inpImage)
    # cv2.imshow("HLS Filtered", hls_result)
    # cv2.imshow("Grayscale", gray)
    # cv2.imshow("Thresholded", thresh)
    # cv2.imshow("Blurred", blur)
    # cv2.imshow("Canny Edges", canny)

    return image, hls_result, gray, thresh, blur, canny
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

    # 🐸 ( Type 3 조금 넓은 시야 )
    src = np.float32([[0, height-150],
                      [width, height-150],
                      [300, 200],
                      [width-300, 200]])

    # 🐸 ( Type 4 많이 넓은 시야 )
    # src = np.float32([[0, height-150],
    #                   [width, height-150],
    #                   [400, 150],
    #                   [width-400, 150]])

    # Window to be shown
    # 표시할 윈도우
    dst = np.float32([[0, 0],
                      [1280, 0],
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
    cv2.imshow("Birdseye Left", birdseyeLeft)
    cv2.imshow("Birdseye Right", birdseyeRight)

    return birdseye, birdseyeLeft, birdseyeRight, minv
#### END - FUNCTION TO APPLY PERSPECTIVE WARP ##################################
################################################################################


################################################################################
#### START - FUNCTION TO PLOT THE HISTOGRAM OF WARPED IMAGE ####################
#### START - 왜곡된 이미지의 히스토그램을 플롯하는 기능 ########################
def plotHistogram(inpImage):

    histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis=0)

    midpoint = np.int32(histogram.shape[0] / 2)
    leftxBase = np.argmax(histogram[:midpoint])
    rightxBase = np.argmax(histogram[midpoint:]) + midpoint

    plt.xlabel("Image X Coordinates")
    plt.ylabel("Number of White Pixels")

    # Return histogram and x-coordinates of left & right lanes to calculate
    # lane width in pixels
    # 픽셀 단위의 계산식에 필요한
    # ( 왼쪽, 오른쪽 차선의 히스토그램 ) 및 ( x 좌표 ) 반환
    return histogram, leftxBase, rightxBase
#### END - FUNCTION TO PLOT THE HISTOGRAM OF WARPED IMAGE ######################
################################################################################


################################################################################
#### START - APPLY SLIDING WINDOW METHOD TO DETECT CURVES ######################
def slide_window_search(binary_warped, histogram):

    # Find the start of left and right lane lines using histogram info
    # 히스토그램 정보를 사용하여 왼쪽, 오른쪽 차선의 시작점 찾기
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int32(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # A total of 9 windows will be used
    # 총 12개의 창을 사용하여 알고리즘 계산
    # ( default 윈도우 9, 마진 120, 최소값 50 )
    nwindows = 6
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 120
    minpix = 10
    left_lane_inds = []
    right_lane_inds = []

    #### START - Loop to iterate through windows and search for lane lines #####
    #### START - 루프를 통해 창을 반복 -> 차선 검출 알고리즘 #####
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 슬라이딩 윈도우 객채 생성
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
    #### END - Loop to iterate through windows and search for lane lines #######

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Apply 2nd degree polynomial fit to fit curves
    # 곡선에 맞게 2차 다항식 피팅 적용
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)
    # plt.plot(right_fitx)
    # plt.show()

    # 주행 가능 도로 이미지 색상
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]

    # 중심 축 이미지 색상
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # plt.imshow(out_img)
    # cv2.imshow('sliding_window', out_img)

    plt.plot(left_fitx,  ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    return ploty, left_fit, right_fit, ltx, rtx
#### END - APPLY SLIDING WINDOW METHOD TO DETECT CURVES ########################
################################################################################


################################################################################
#### START - APPLY GENERAL SEARCH METHOD TO DETECT CURVES ######################
#### START - 곡선을 감지하는 일반 검색 방법 적용 ###############################
def general_search(binary_warped, left_fit, right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                         left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                           right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ## VISUALIZATION ###########################################################
    ## 시각화 ##################################################################

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    # 결과 도출
    # general_search_result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # cv2.imshow('general_search_result', general_search_result)
    # plt.imshow(general_search_result)

    plt.plot(left_fitx,  ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty

    return ret
#### END - APPLY GENERAL SEARCH METHOD TO DETECT CURVES ########################
################################################################################


################################################################################
#### START - FUNCTION TO MEASURE CURVE RADIUS ##################################
#### START - 곡선 반경 측정 기능 ###############################################
def measure_lane_curvature(ploty, leftx, rightx):

    # ( y축의 위에서 아래로 일치하도록 반전 )
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Choose the maximum y-value, corresponding to the bottom of the image
    # 이미지 하단에 해당하는 최대의 y 값 추출
    y_eval = np.max(ploty)

    # Fit new polynomials to x, y in world space
    # 차선 공간의 x,y 좌표에 새롭게 정의된 다항식 적용
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    # 새로운 곡률 반경 계산
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                            right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
    # 아래 return 반환 함수부터 곡률 반경은 미터 단위..!

    # print(left_curverad, 'm', right_curverad, 'm')

    # Decide if it is a left or a right curve
    # 왼쪽 or 오른쪽 커브인지 결정 ( default value = 60 )
    if leftx[0] - leftx[-1] > 60:
        curve_direction = 'Left Curve'
    elif leftx[-1] - leftx[0] > 60:
        curve_direction = 'Right Curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad + right_curverad) / 2.0, curve_direction
#### END - FUNCTION TO MEASURE CURVE RADIUS ####################################
################################################################################


################################################################################
#### START - FUNCTION TO VISUALLY SHOW DETECTED LANES AREA #####################
#### START - 감지된 차선 영역을 시각적으로 보여주는 기능 #######################
def draw_lane_lines(original_image, warped_image, Minv, draw_info):

    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (150, 130, 0))       # 주행 라인
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))  # 중심 축

    newwarp = cv2.warpPerspective(
        color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.5, 0)

    return pts_mean, result
#### END - FUNCTION TO VISUALLY SHOW DETECTED LANES AREA #######################
################################################################################


#### START - FUNCTION TO CALCULATE DEVIATION FROM LANE CENTER ##################
#### START - 차선 중심으로부터의 편차를 계산하는 기능 ##########################
################################################################################
def offCenter(meanPts, inpFrame):

    # Calculating deviation in meters
    # 미터 단위의 편차 계산
    mpts = meanPts[-1][-1][-2].astype(int)
    pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
    deviation = pixelDeviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"

    return deviation, direction
################################################################################
#### END - FUNCTION TO CALCULATE DEVIATION FROM LANE CENTER ####################


################################################################################
#### START - FUNCTION TO ADD INFO TEXT TO FINAL IMAGE ##########################
#### START - 최종 이미지에 주행 정보 텍스트를 추가하는 기능 ####################
def addText(img, radius, direction, deviation, devDirection):

    # Add the radius and center position to the image
    # 이미지에 반경과 중심 위치 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_rgb = (240, 248, 255)

    if (direction != 'Straight'):
        text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
        text1 = 'Curve Direction: ' + (direction)

    else:
        text = 'Radius of Curvature: ' + 'N/A'
        text1 = 'Curve Direction: ' + (direction)

    cv2.putText(img, text, (400, 180), font, 0.8, font_rgb, 2, cv2.LINE_AA)
    cv2.putText(img, text1, (400, 230), font, 0.8, font_rgb, 2, cv2.LINE_AA)

    # Deviation ( 편차 )
    deviation_text = 'Off Center: ' + \
        str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
    cv2.putText(img, deviation_text, (400, 280),
                font, 0.8, font_rgb, 2, cv2.LINE_AA)

    print('🐸 커브 방향:', direction, ', 🎃 서브 모터:', devDirection)

    return img
#### END - FUNCTION TO ADD INFO TEXT TO FINAL IMAGE ############################
################################################################################

################################################################################
######## END - FUNCTIONS TO PERFORM IMAGE PROCESSING ###########################
################################################################################

################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
######## START - MAIN FUNCTION #################################################
################################################################################


# Read the input image
image = readVideo()

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
        img, hls, grayscale, thresh, blur, canny = processImage(birdView)
        imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(
            birdViewL)
        imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(
            birdViewR)

        # 🐸 좌 / 우 차선 구별
        # 1. 밝기 값이 적용된 thresh 파일 불러오기
        # 2. "get_histogram()" 함수를 호출하여 히스토그램을 플롯하고 표시
        hist, leftBase, rightBase = plotHistogram(thresh)
        # print(rightBase - leftBase)
        # plt.plot(hist)
        # plt.show()

        # 슬라이딩 윈도우 계산
        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(
            thresh, hist)
        plt.plot(left_fit)

        draw_info = general_search(thresh, left_fit, right_fit)

        curveRad, curveDir = measure_lane_curvature(
            ploty, left_fitx, right_fitx)

        # 감지된 차선 영역을 파란색으로 채우기
        meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)

        deviation, directionDev = offCenter(meanPts, frame)

        # 차선 정보 추가
        finalImg = addText(result, curveRad, curveDir, deviation, directionDev)

        # 최종 이미지 출력
        cv2.imshow("Final", finalImg)
    except:
        print("라인 검출 에러")

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


##
