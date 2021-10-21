import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import time
from utils_resized_video import Line, color_invert, find_LR_lines, draw_lane, print_road_status, display_lines, average_slope_intercept, region_of_interest, canny, gaussian_blur, warp_image, grayscale

# 카메라 프레임 조정
prev_time = 0
FPS = 8

# 라인 객체 생성
left_line = Line()
right_line = Line()

cap = cv2.VideoCapture("tracks/real_camera.mp4")
width, height = 640, 360

# 관심영역 설정 후 이미지
# vertices = np.array([[
#     (0, 350), (130, 155), (640, 350), (475, 155)
# ]], dtype=np.int32)

# # 좌하, 좌상, 우하, 우상(before, after)
vertices = np.array([[
    (0, 320), (110, 105), (640, 350), (495, 105)
]], dtype=np.int32)


while (cap.isOpened()):
    ret, img = cap.read()

    if (ret is False):
        break

    try:
        # 이미지 리사이징
        lane_img = cv2.resize(img, (width, height))
        # 색상 반전
        # color_interted_img = grayscale(lane_img)
        color_interted_img = color_invert(lane_img)
        '''        
        # 교실 환경전용 색상 반전
        # color_interted_img = cv2.bitwise_not(color_interted_img)
        # cv2.imshow("color_interted_img", color_interted_img)
        '''
        #  이미지의 노이즈를 줄이기 위해 가우시안 효과 적용
        # gaussian = gaussian_blur(color_interted_img, 5)
        # 캐니 & 관심영역 적용
        canny_img = canny(color_interted_img, 30, 90)
        # cv2.imshow("canny_img", canny_img)

        # cropped_image = region_of_interest(canny_img, vertices)
        # cv2.imshow("region_of_interest", canny_img)
        #######################################################
        # TODO src roi 범위 조정 ...
        # 좌하, 우하, 좌상, 우상
        src = np.float32([[0, height-100], [width, height-100], [0, 120], [width, 120]])
        # 좌하, 우하, 좌상, 우상
        dst = np.float32(
            [[200, height+300], [410, height+300], [0, 0], [width, 0]])
        # 해당 점의 네 쌍에서 원근 변환을 계산
        M = cv2.getPerspectiveTransform(src, dst)
        # 역변환 ( 원본 이미지 출력을 위해...! )
        Minv = cv2.getPerspectiveTransform(dst, src)
        # ROI 이미지 영역(자르기)에 슬라이싱 적용
        bird_cropped_image = canny_img[40:(225+height), 0:width]
        # 최종 이미지 ( height 값 + -> 초기 라인 인식 범위 커짐 )
        brid_eye_view_img = cv2.warpPerspective(
            bird_cropped_image, M, (640, height+210))
        # cv2.imshow("brid_eye_view_img", brid_eye_view_img)

        # 원본 이미지 세트   
        bird_temp_image = lane_img[40:(225+height), 0:width]
        brid_eye_original_img = cv2.warpPerspective(
            bird_temp_image, M, (640, height+210))
        ########################################################
        # 라인 보정
        temp_resized_img1 = np.copy(brid_eye_view_img)
        temp_resized_img2 = np.copy(brid_eye_original_img)
        temp_height, temp_weight = temp_resized_img1.shape[:2]
        # print(temp_w,temp_h)
        temp_resized_img1 = cv2.resize(
            temp_resized_img1, (int(temp_height/3), int(temp_weight/3)))
        cv2.imshow("1", temp_resized_img1)
        temp_resized_img2 = cv2.resize(
            temp_resized_img2, (int(temp_height/3), int(temp_weight/3)))
        cv2.imshow("2", temp_resized_img2)


        lines = cv2.HoughLinesP(temp_resized_img2, 2, np.pi/180,
                                100, np.array([]), minLineLength=3, maxLineGap=3)

        # left_fit = []
        # right_fit = []

        # for line in lines:
        # # 검출된 선 그리기 ---③
        #     x1, y1, x2, y2 = line[0]
        #     parameters = np.polyfit((x1, x2), (y1, y2), 1)

        #     slope = parameters[0]
        #     intercept = parameters[1]

        #     if slope < 0:
        #         left_fit.append((slope, intercept))
        #         # print("좌측 라인 검출")
        #     else:
        #         right_fit.append((slope, intercept))
        #         # print("우측 라인 검출")

        #     cv2.line(temp_resized_img2, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # cv2.imshow("brid_eye_original_img", temp_resized_img2)

        averaged_lines = average_slope_intercept(temp_resized_img1, lines)
        line_img = display_lines(temp_resized_img1, averaged_lines)
        cv2.imshow("123", line_img)

        # # 좌우 라인 검출
        # searching_img = find_LR_lines(averaged_lines, left_line, right_line)
        # w_comb_result, w_color_result = draw_lane(
        #     searching_img, left_line, right_line)

        # combo_img = cv2.addWeighted(lane_img, 0.8, w_comb_result, 1, 1)
        # # result_img = print_road_status(combo_img, left_line, right_line)
        # cv2.imshow("Result", combo_img)
        # plt.show()
        # plt.close()
    except:
        print("라인 검출 에러...!")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("종료...")
cap.release()
cv2.destroyAllWindows()
