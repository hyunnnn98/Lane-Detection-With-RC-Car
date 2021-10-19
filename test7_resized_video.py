import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import time
from utils_resized_video import Line, color_invert, find_LR_lines, draw_lane, print_road_status, display_lines, average_slope_intercept, region_of_interest, canny, gaussian_blur, warp_image

# 카메라 프레임 조정
prev_time = 0
FPS = 8

# 라인 객체 생성
left_line = Line()
right_line = Line()

cap = cv2.VideoCapture("tracks/curve1.mp4")
width, height = 640, 360

# 관심영역 설정 후 이미지
vertices = np.array([[
    (0, 350), (130, 55), (640, 350), (475, 55)
]], dtype=np.int32)


while (cap.isOpened()):
    ret, img = cap.read()

    if (ret is True) :
        # 이미지 리사이징
        lane_img = cv2.resize(img, (width, height))

        # 색상 반전
        color_interted_img = color_invert(lane_img)

        '''        
        # 교실 환경전용 색상 반전
        # color_interted_img = cv2.bitwise_not(color_interted_img)
        '''

        #  이미지의 노이즈를 줄이기 위해 가우시안 효과 적용
        gaussian = gaussian_blur(color_interted_img, 5)

        # 캐니 & 관심영역 적용
        canny_img = canny(gaussian, 40, 120)
        cropped_image = region_of_interest(canny_img, vertices)
        cv2.imshow("region_of_interest", cropped_image)

        #######################################################
        src = np.float32([[0, height], [width, height], [0, 0], [width, 0]])

        # 좌하, 우하, 좌상, 우상
        dst = np.float32(
            [[285, height], [390, height], [0, 0], [width, 0]])
        # 해당 점의 네 쌍에서 원근 변환을 계산
        transforted_M = cv2.getPerspectiveTransform(src, dst)  


        # 역변환 ( 원본 이미지 출력을 위해...! )
        original_M = cv2.getPerspectiveTransform(dst, src)
 
        # ROI 이미지 영역(자르기)에 슬라이싱 적용
        bird_cropped_image = canny_img[225:(225+height), 0:width]
        # 최종 이미지
        brid_eye_view_img = cv2.warpPerspective(
            bird_cropped_image, transforted_M, (width, height))

        cv2.imshow("brid_eye_view_img", brid_eye_view_img)
        ########################################################

        # 라인 보정
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
                                100, np.array([]), minLineLength=40, maxLineGap=5)

        averaged_lines = average_slope_intercept(cropped_image, lines)
        line_img = display_lines(color_interted_img, averaged_lines)

        # 좌우 라인 검출
        searching_img = find_LR_lines(cropped_image, left_line, right_line)
        w_comb_result, w_color_result = draw_lane(
            searching_img, left_line, right_line)
        cv2.imshow("cropped_image", searching_img)

        # combo_img = cv2.addWeighted(color_interted_img, 0.6, line_img, 1, 1)
        combo_img = cv2.addWeighted(lane_img, 0.8, w_comb_result, 1, 1)
        result_img = print_road_status(combo_img, left_line, right_line)

        cv2.imshow("Result", result_img)
        # plt.imshow(result_img)
        # plt.show()
        # plt.close()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("종료...")
cap.release()
cv2.destroyAllWindows()
