import cv2
import numpy as np
from calibration import calib, undistort
from threshold import gradient_combine, hls_combine, comb_result
from utils import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map

cap = cv2.VideoCapture("strate.mp4")

# 라인 객체 생성
left_line = Line()
right_line = Line()

# threshold 값
th_sobelx, th_sobely, th_mag, th_dir = (
    35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
# mtx, dist = calib()


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, range):     # 관심영역 지정
    polygons = np.array([range])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):     # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def display_lines(img, lines):      # 화면에 라인 표시
    line_image = np.zeros_like(img)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # print(line)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    return line_image

# def color_invert(img):  # 색 반전
#     inverted_image = np.copy(img)
#     thresholds = (img[:, :, 1] < bgr_threshold[0]) \
#         | (img[:, :, 1] < bgr_threshold[1]) \
#         | (img[:, :, 2] < bgr_threshold[2])

#     inverted_image[thresholds] = [0, 0, 0]

#     return inverted_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            parameters = np.polyfit((x1, x2), (y1, y2), 1)

            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, 'Left', left_fit_average)
    right_line = make_coordinates(image, 'Right', right_fit_average)

    return np.array([left_line, right_line])


def make_coordinates(img, line_type, line_parameters):  # 라인 범위 지정
    try:
        slope, intercept = line_parameters
    except TypeError:
        print("ERR)", line_type, "라인을 인식하지 못했습니다.")
        slope, intercept = 0.001, 0
    # print(img.shape)
    y1 = img.shape[0]
    y2 = int(y1*(2/5))  # 이거 변경하면 라인이 따지는 길이가 변경됨

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


if __name__ == '__main__':
    while (cap.isOpened()):
        ret, img = cap.read()

        # # resize video
        # lane_img = cv2.resize(img, None, fx=0.5,
        #                       fy=0.5, interpolation=cv2.INTER_AREA)
        lane_img = cv2.resize(img, (1280, 720))

        rows, cols = lane_img.shape[:2]
        # cv2.imshow('warp', lane_img)
        print(rows, cols)
        # 색상 반전
        combined_gradient = gradient_combine(
            lane_img, th_sobelx, th_sobely, th_mag, th_dir)
        combined_hls = hls_combine(lane_img, th_h, th_l, th_s)
        combined_result = comb_result(combined_gradient, combined_hls)

        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
        s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        # 좌상 좌하 우상 우하
        dst = np.float32([(0, 720), (0, 0), (550, 0), (550, 720)])

        warp_img, M, Minv = warp_image(
            combined_result, src, dst, (720, 720))
        cv2.imshow('warp', warp_img)

        searching_img = find_LR_lines(warp_img, left_line, right_line)

        w_comb_result, w_color_result = draw_lane(
            searching_img, left_line, right_line)
        # cv2.imshow('w_comb_result', w_comb_result)

        # Drawing the lines back down onto the road
        color_result = cv2.warpPerspective(
            w_color_result, Minv, (c_cols, c_rows))
        lane_color = np.zeros_like(lane_img)
        lane_color[220:rows - 12, 0:cols] = color_result

        # Combine the result with the original image
        result = cv2.addWeighted(lane_img, 1, lane_color, 0.3, 0)
        # cv2.imshow('result', result.astype(np.uint8))

        info, info2 = np.zeros_like(result),  np.zeros_like(result)
        info[5:110, 5:190] = (255, 255, 255)
        info2[5:110, cols-111:cols-6] = (255, 255, 255)
        info = cv2.addWeighted(result, 1, info, 0.2, 0)
        info2 = cv2.addWeighted(info, 1, info2, 0.2, 0)
        road_map = print_road_map(w_color_result, left_line, right_line)
        info2[10:105, cols-106:cols-11] = road_map
        info2 = print_road_status(info2, left_line, right_line)
        cv2.imshow('road info', info2)
        # cv2.imshow('warp', searching_img)
        # 캐니 적용
        # canny_img = canny(gaussian, 60, 180)

        # # left_cropped_image = region_of_interest(canny_img, left_roi)
        # # right_cropped_image = region_of_interest(canny_img, right_roi)

        # vertices = np.array([[
        #     (150, 500), (300, 80), (1050, 500), (900, 80)
        # ]], dtype=np.int32)

        # cropped_image = region_of_interest(canny_img, vertices)

        # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
        #                         100, np.array([]), minLineLength=40, maxLineGap=5)

        # # Drawing the lines back down onto the road
        # averaged_lines = average_slope_intercept(lane_img, lines)

        # line_img = display_lines(color_interted_img, averaged_lines)

        # # combo_img = cv2.addWeighted(color_interted_img, 0.6, line_img, 1, 1)
        # combo_img = cv2.addWeighted(lane_img, 0.6, line_img, 1, 1)

        # cv2.imshow("ROI", combo_img)
        # cv2.imshow("REAL", color_interted_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
