import cv2
import numpy as np

# import time
from utils_resized_video import Line, color_invert, find_LR_lines, draw_lane, print_road_status, display_lines, average_slope_intercept, region_of_interest, canny, gaussian_blur, onMouse

# 카메라 프레임 조정
prev_time = 0
FPS = 8

# BGR 제한 값 설정
blue_threshold = 116
green_threshold = 57
red_threshold = 43
bgr_threshold = [blue_threshold, green_threshold, red_threshold]

# 라인 객체 생성
left_line = Line()
right_line = Line()

cap = cv2.VideoCapture("tracks/real_camera.mp4")
width, height = 640, 360

# 관심영역 설정 후 이미지
vertices = np.array([[
    (0, 350), (130, 55), (640, 350), (475, 55)
]], dtype=np.int32)


cv2.namedWindow('Window')
cv2.createTrackbar('BLUE', 'Window', blue_threshold, 255, onMouse)
cv2.createTrackbar('GREEN', 'Window', green_threshold, 255, onMouse)
cv2.createTrackbar('RED', 'Window', red_threshold, 255, onMouse)

custom_blue = cv2.getTrackbarPos('BLUE', 'Window')
custom_green = cv2.getTrackbarPos('GREEN', 'Window')
custom_red = cv2.getTrackbarPos('RED', 'Window')

while (cap.isOpened()):
    ret, img = cap.read()
    # print(custom_red, custom_blue, custom_green)

    if (ret is False):
        break

    try:

        # 이미지 리사이징
        lane_img = cv2.resize(img, (width, height))

        # 색상 반전
        color_interted_img = color_invert(
            lane_img, [custom_blue, custom_green, custom_red])
        cv2.imshow("lane_img", lane_img)
        # cv2.imshow("color_interted_img", coloqr_interted_img)

        '''        
        # 교실 환경전용 색상 반전
        # color_interted_img = cv2.bitwise_not(color_interted_img)
        '''

        # #  이미지의 노이즈를 줄이기 위해 가우시안 효과 적용
        # gaussian = gaussian_blur(color_interted_img, 5)

        # 캐니 & 관심영역 적용
        canny_img = canny(color_interted_img, 40, 120)
        # cv2.imshow("canny_img", canny_img)
        cropped_image = region_of_interest(canny_img, vertices)
        # cv2.imshow("cropped_image", cropped_image)

        # #######################################################
        # 좌하, 우하, 좌상, 우상
        src = np.float32(
            [[0, height-100], [width, height-100], [0, 120], [width, 120]])
        # 좌하, 우하, 좌상, 우상
        dst = np.float32(
            [[200, height+300], [410, height+300], [0, 0], [width, 0]])
        # 해당 점의 네 쌍에서 원근 변환을 계산
        M = cv2.getPerspectiveTransform(src, dst)
        # 역변환 ( 원본 이미지 출력을 위해...! )
        Minv = cv2.getPerspectiveTransform(dst, src)
        # ROI 이미지 영역(자르기)에 슬라이싱 적용
        bird_cropped_area = canny_img[40:(225+height), 0:width]

        bird_original_cropped_area = color_interted_img[40:(225+height), 0:width]
        # 최종 이미지 ( height 값 + -> 초기 라인 인식 범위 커짐 )
        brid_eye_view_img = cv2.warpPerspective(
            bird_cropped_area, M, (640, height+210))
        brid_eye_original_img = cv2.warpPerspective(
            bird_original_cropped_area, M, (640, height+210))
        cv2.imshow("brid_eye_view_img", brid_eye_original_img)
        ########################################################

        # # 라인 보정
        # lines = cv2.HoughLinesP(brid_eye_view_img, 2, np.pi/180,
        #                         50, np.array([]), minLineLength=70, maxLineGap=3)

        # averaged_lines = average_slope_intercept(brid_eye_view_img, lines)
        # line_img = display_lines(brid_eye_original_img, averaged_lines)
        # averaged_canny_lines = canny(line_img, 40, 120)
        # cv2.imshow("averaged_canny_lines", averaged_canny_lines)
        # print(averaged_canny_lines.shape[:2])

        # # 좌우 라인 검출
        # searching_img = find_LR_lines(
        #     averaged_canny_lines, left_line, right_line)
        # w_comb_result, w_color_result = draw_lane(
        #     searching_img, left_line, right_line)
        # cv2.imshow("test", w_comb_result)

        # Drawing the lines back down onto the road
        # color_result = cv2.warpPerspective(
        #     w_color_result, transforted_M, (bird_cropped_cols, bird_cropped_rows))
        # lane_color = np.zeros_like(lane_img)
        # lane_color[110:360 - 12, 0:640] = color_result

        # Combine the result with the original image
        # combine_result = cv2.addWeighted(lane_img, 1, lane_color, 0.3, 0)
        # cv2.imshow("combine_result", color_result)

        # combo_img = cv2.addWeighted(color_interted_img, 0.6, line_img, 1, 1)
        # combo_img = cv2.addWeighted(lane_img, 0.8, w_comb_result, 1, 1)
        # result_img = print_road_status(
        #     w_comb_result, left_line, right_line)
        # cv2.imshow("Result", result_img)

        # plt.imshow(result_img)
        # plt.show()
        # plt.close()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        custom_blue = cv2.getTrackbarPos('BLUE', 'Window')
        custom_green = cv2.getTrackbarPos('GREEN', 'Window')
        custom_red = cv2.getTrackbarPos('RED', 'Window')

        # cv2.waitKey(50)
    except:
        print("라인 검출 에러..!")

print("종료...")
cap.release()
cv2.destroyAllWindows()
