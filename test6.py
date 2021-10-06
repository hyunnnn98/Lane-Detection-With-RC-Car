import cv2
import numpy as np

cap = cv2.VideoCapture("ex_track_2.mp4")
width, height = 1200, 600

# 관심영역 설정 후 이미지
left_roi = [(0, height), (600, height), (300, 0)]
right_roi = [(600, height), (1200, height), (900, 0)]
line_thick = 10

#  BGR 제한 값 설정
blue_threshold = 10
green_threshold = 190
red_threshold = 230
bgr_threshold = [blue_threshold, green_threshold, red_threshold]


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
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), line_thick)

    return line_image


def mix_roi(left_roi, right_roi):   # 관심영역 레이어 마스크 설정
    roi_masked_image = cv2.add(left_roi, right_roi)

    return roi_masked_image


def color_invert(img):  # 색 반전
    inverted_image = np.copy(img)
    thresholds = (img[:, :, 1] < bgr_threshold[0]) \
        | (img[:, :, 1] < bgr_threshold[1]) \
        | (img[:, :, 2] < bgr_threshold[2])

    inverted_image[thresholds] = [0, 0, 0]

    return inverted_image


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
    y2 = int(y1*(3/5))  # 이거 변경하면 라인이 따지는 길이가 변경됨

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


while (True):
    ret, img = cap.read()
    lane_img = cv2.resize(img, (width, height))

    # 색상 반전
    color_interted_img = color_invert(lane_img)

    # 그레이 스케일링
    # gray = grayscale(interted_image)

    #  이미지의 노이즈를 줄이기 위해 가우시안 효과 적용
    gaussian = gaussian_blur(color_interted_img, 5)

    # 캐니 적용
    canny_img = canny(gaussian, 60, 180)

    # left_cropped_image = region_of_interest(canny_img, left_roi)
    # right_cropped_image = region_of_interest(canny_img, right_roi)

    vertices = np.array([[
        (150, 500), (300, 80), (1050, 500), (900, 80)
    ]], dtype=np.int32)

    cropped_image = region_of_interest(canny_img, vertices)

    #  관심영역 합치기
    # roi_masked_image = mix_roi(left_cropped_image, right_cropped_image)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
                            100, np.array([]), minLineLength=40, maxLineGap=5)

    averaged_lines = average_slope_intercept(lane_img, lines)

    line_img = display_lines(color_interted_img, averaged_lines)

    combo_img = cv2.addWeighted(color_interted_img, 0.6, line_img, 1, 1)

    cv2.imshow("ROI", combo_img)
    # cv2.imshow("REAL", color_interted_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()