################################################################################
######## START - 이미지 처리를 수행하는 기능 ###################################
################################################################################

# IMPORT NECESSARY LIBRARIES
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt, cm, colors
# 🍒 meter-to-pixel 변환을 유지하기 위한 변수 정의
ym_per_pix = 30 / 720

# 🍒 표준 차선 너비 3.7m를 차선 너비(픽셀)로 나눈 값으로
# 계산할때는 프레임의 높이와 혼동하지 않기 위해 약 720 픽셀로 계산
xm_per_pix = 12 / 720  # ( 기본 값 : 3.7m )

################################################################################
#### START - APPLY SLIDING WINDOW METHOD TO DETECT CURVES ######################


def slide_window_search(binary_warped, histogram):

    # 히스토그램 정보를 사용하여 왼쪽, 오른쪽 차선의 시작점 찾기
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int32(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 총 12개의 창을 사용하여 알고리즘 계산
    # ( default 윈도우 9, 마진 120, 최소값 50 )
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 60
    minpix = 130
    left_lane_inds = []
    right_lane_inds = []

    #### START - 루프를 통해 창을 반복 -> 차선 검출 알고리즘 ###################
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
    #### END - 루프를 통해 창을 반복 -> 차선 검출 알고리즘 #####################

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

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
    
    cv2.imshow('sliding_window', cv2.flip(cv2.resize(out_img, (320, 180)), 0))

    plt.plot(left_fitx,  ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    return ploty, left_fit, right_fit, ltx, rtx
#### END - 루프를 통해 창을 반복 -> 차선 검출 알고리즘 #########################
################################################################################


################################################################################
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
    # cv2.imshow('general_search_result', out_img)
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
#### END - 곡선을 감지하는 일반 검색 방법 적용 #################################
################################################################################


################################################################################
#### START - 곡선 반경 측정 기능 ###############################################
def measure_lane_curvature(ploty, leftx, rightx):

    # ( y축의 위에서 아래로 일치하도록 반전 )
    leftx = leftx[::-1]    # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # 이미지 하단에 해당하는 최대의 y 값 추출
    y_eval = np.max(ploty)

    # 차선 공간의 x,y 좌표에 새롭게 정의된 다항식 적용
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # 새로운 곡률 반경 계산
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                            right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # 아래 return 반환 함수부터 곡률 반경은 미터 단위..!
    # 왼쪽 or 오른쪽 커브인지 결정 ( default value = 60 )
    if leftx[0] - leftx[-1] > 300:
        curve_direction = 'Right Curve'
    elif leftx[-1] - leftx[0] > 300:
        curve_direction = 'Left Curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad + right_curverad) / 2.0, curve_direction
#### END - 곡선 반경 측정 기능 #################################################
################################################################################


################################################################################
#### START - 감지된 차선 영역을 시각적으로 보여주는 기능 #######################
def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    img_x = original_image.shape[1]
    img_y = original_image.shape[0]

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
    # print('중심축 : ', pts_mean)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (img_x, img_y))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.5, 0)

    return pts_mean, result
#### END - 감지된 차선 영역을 시각적으로 보여주는 기능 #########################
################################################################################


################################################################################
#### START - 차선 중심으로부터의 편차를 계산하는 기능 ##########################
def offCenter(meanPts, inpFrame):

    # 미터 단위의 편차 계산
    mpts = meanPts[-1][-1][-2].astype(int)
    pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
    deviation = pixelDeviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"

    return deviation, direction
################################################################################
#### END - 차선 중심으로부터의 편차를 계산하는 기능 ############################


################################################################################
#### START - 최종 이미지에 주행 정보 텍스트를 추가하는 기능 ####################
def addText(img, radius, direction, deviation, devDirection):

    # 이미지에 반경과 중심 위치 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_rgb = (240, 248, 255)

    if (direction != 'Straight'):
        text = 'Curvature: ' + '{:04.0f}'.format(radius) + 'm'
        text1 = 'Curve Direction: ' + (direction)

    else:
        text = 'Curvature: ' + 'N/A'
        text1 = 'Curve Direction: ' + (direction)

    cv2.putText(img, text, (400, 180), font, 0.8, font_rgb, 2, cv2.LINE_AA)
    cv2.putText(img, text1, (400, 230), font, 0.8, font_rgb, 2, cv2.LINE_AA)

    steeringWheelRadius = round(abs(deviation), 3)

    # Deviation ( 편차 )
    deviation_text = 'Off Center: ' + \
        str(steeringWheelRadius) + 'm' + ' to the ' + devDirection
    cv2.putText(img, deviation_text, (400, 280),
                font, 0.8, font_rgb, 2, cv2.LINE_AA)

    if direction == 'Straight':
        direction = '직진 코스'
    elif direction == 'Left Curve':
        direction = '우회전 코스'
    elif direction == 'Right Curve':
        direction = '좌회전 코스'
    
    if devDirection == 'left':
        steeringWheelRadius = steeringWheelRadius * -1

    # print('🐸 방향:', direction, ', 🎃 서브 모터:', devDirection)

    return img, steeringWheelRadius
#### END - 최종 이미지에 주행 정보 텍스트를 추가하는 기능 ######################
################################################################################

################################################################################
######## END - FUNCTIONS TO PERFORM IMAGE PROCESSING ###########################
################################################################################
