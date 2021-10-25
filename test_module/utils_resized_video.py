import numpy as np
import cv2
from PIL import Image
import matplotlib.image as mpimg

#  BGR 제한 값 설정
# blue_threshold = 116
# green_threshold = 57
# red_threshold = 43
# bgr_threshold = [blue_threshold, green_threshold, red_threshold]

# 라인 굵기
line_thick = 5
window_number = 10

# 슬라이딩 창 수 선택
num_windows = 10

class Line:
    def __init__(self):
        # 마지막 반복에서 라인이 감지되었습니까?
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 60
        # 마지막 n회 반복에 대한 적합선의 x 값
        self.prevx = []
        # 가장 최근 피팅에 대한 다항식 계수
        self.current_fit = [np.array([False])]
        # 일부 단위에서 선의 곡률 반경
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_inf = None
        self.curvature = None
        self.deviation = None

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


def draw_lines(img, lines, color=[0, 0, 255], thickness=1):     # 선 그리기
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


def color_invert(img, bgr_threshold):  # 색 반전
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
            # print(x1, y1, x2, y2)

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
    y2 = int(y1*(1/5))  # 이거 변경하면 라인이 따지는 길이가 변경됨

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])



def warp_image(img, src, dst, size):
    """ 원근 변환 """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv


def rad_of_curvature(left_line, right_line):
    """ 곡률 반경 측정 """

    ploty = left_line.ally
    leftx, rightx = left_line.allx, right_line.allx

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Define conversions in x and y from pixels space to meters
    # 픽셀 공간에서 미터로의 x 및 y 변환 정의
    width_lanes = abs(right_line.startx - left_line.startx)
    ym_per_pix = 22 / 570  # y 차원의 픽셀당 미터
    # meters per pixel in x dimension
    xm_per_pix = 3.7*(570/640) / width_lanes

    # 곡률 반경을 원하는 y 값을 정의
    # 이미지 하단에 해당하는 최대 y값
    y_eval = np.max(ploty)

    # 공간에서 x, y에 새로운 다항식 맞추기
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # 곡률 반경 계산
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # 곡률 반경 결과
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad


def smoothing(lines, pre_lines=3):
    # 라인 수집 및 평균 라인 그리기
    lines = np.squeeze(lines)
    avg_line = np.zeros((570))

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line


def blind_search(b_img, left_line, right_line):
    """
    블라인드 서치 - 첫 번째 프레임, 차선 이탈
    히스토그램 및 슬라이딩 윈도우 사용
    """

    # 이미지의 아래쪽 절반의 히스토그램을 가져옵니다.
    histogram = np.sum(b_img[int(b_img.shape[0] / 2):, :], axis=0)

    # 드로잉할 출력 이미지 생성 및 결과 시각화
    output = np.dstack((b_img, b_img, b_img)) * 255

    # 히스토그램의 왼쪽과 오른쪽 절반의 피크를 찾습니다.
    # 이것은 왼쪽과 오른쪽 라인의 시작점이 될 것입니다.
    midpoint = np.int(histogram.shape[0] / 2)
    start_leftX = np.argmax(histogram[:midpoint])
    start_rightX = np.argmax(histogram[midpoint:]) + midpoint

    # 창 높이 설정
    window_height = np.int(b_img.shape[0] / num_windows)

    # 이미지에서 0이 아닌 모든 픽셀의 x 및 y 위치 식별
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 각 창에 대해 업데이트될 현재 위치
    current_leftX = start_leftX
    current_rightX = start_rightX

    # 창을 중앙에 맞추기 위해 찾은 최소 픽셀 수 설정
    min_num_pixel = 100

    # 왼쪽 및 오른쪽 차선 픽셀 인덱스를 수신하는 빈 목록 만들기
    win_left_lane = []
    win_right_lane = []

    window_margin = left_line.window_margin

    # 단계별 처리 시작
    for window in range(num_windows):
        # x 및 y(및 오른쪽 및 왼쪽)에서 창 경계 식별
        win_y_low = b_img.shape[0] - (window + 1) * window_height
        win_y_high = b_img.shape[0] - window * window_height
        win_leftx_min = current_leftX - window_margin
        win_leftx_max = current_leftX + window_margin
        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        # 시각화 이미지에 창 그리기
        cv2.rectangle(output, (win_leftx_min, win_y_low),
                      (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_rightx_min, win_y_low),
                      (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # 창 내에서 x 및 y에서 0이 아닌 픽셀 식별
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
            nonzerox <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
            nonzerox <= win_rightx_max)).nonzero()[0]
        # 인덱스를 목록에 추가
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_inds) > min_num_pixel:
            current_leftX = np.int(np.mean(nonzerox[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_rightX = np.int(np.mean(nonzerox[right_window_inds]))

    # 인덱스 배열 연결
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # 왼쪽 및 오른쪽 라인 픽셀 위치 추출
    leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
    rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

    output[lefty, leftx] = [255, 0, 0]
    output[righty, rightx] = [0, 0, 255]

    # 각각에 2차 다항식 맞추기
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_line.current_fit = left_fit
    right_line.current_fit = right_fit

    # 플로팅을 위한 x 및 y 값 생성
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + \
            left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + \
            right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    left_line.startx, right_line.startx = left_line.allx[len(
        left_line.allx)-1], right_line.allx[len(right_line.allx)-1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    left_line.detected, right_line.detected = True, True

    # 곡률 반영
    rad_of_curvature(left_line, right_line)
    return output


def prev_window_refer(b_img, left_line, right_line):
    """
    refer to previous window info - after detecting lane lines in previous frame
    """
    # Create an output image to draw on and  visualize the result
    output = np.dstack((b_img, b_img, b_img)) * 255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set margin of windows
    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + \
        left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + \
        left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + \
        right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + \
        right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # Identify the nonzero pixels in x and y within the window
    left_inds = ((nonzerox >= leftx_min) & (
        nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (
        nonzerox <= rightx_max)).nonzero()[0]

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    output[lefty, leftx] = [255, 0, 0]
    output[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    leftx_avg = np.average(left_plotx)
    rightx_avg = np.average(right_plotx)

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + \
            left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + \
            right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    # goto blind_search if the standard value of lane lines is high.
    standard = np.std(right_line.allx - left_line.allx)

    if (standard > 80):
        left_line.detected = False

    left_line.startx, right_line.startx = left_line.allx[len(
        left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    # print radius of curvature
    rad_of_curvature(left_line, right_line)
    return output


def find_LR_lines(binary_img, left_line, right_line):
    """
    find left, right lines & isolate left, right lines
    blind search - first frame, lost lane lines
    previous window - after detecting lane lines in previous frame
    """

    # 라인정보가 없을 경우
    if left_line.detected == False:
        return blind_search(binary_img, left_line, right_line)
    # 라인 정보가 있을 경우
    else:
        return prev_window_refer(binary_img, left_line, right_line)


def draw_lane(img, left_line, right_line, lane_color=(255, 0, 0), road_color=(0, 255, 0)):
    """ 선 그리기 & 현재 주행 가능한 공간 """
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_pts_l = np.array(
        [np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
    left_pts_r = np.array(
        [np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))
    right_pts_l = np.array(
        [np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
    right_pts_r = np.array(
        [np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array(
        [np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # 주행 가능 영역 그리기
    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    result = cv2.addWeighted(img, 1, window_img, 0.4, 0)

    return result, window_img


def road_info(left_line, right_line):
    """ print road information onto result image """
    curvature = (left_line.radius_of_curvature +
                 right_line.radius_of_curvature) / 2

    direction = ((left_line.endx - left_line.startx) +
                 (right_line.endx - right_line.startx)) / 2

    if curvature > 2000 and abs(direction) < 100:
        road_inf = '직진'
        curvature = -1
    elif curvature <= 2000 and direction < - 50:
        road_inf = '좌회전'
    elif curvature <= 2000 and direction > 50:
        road_inf = '우회전'
    else:
        if left_line.road_inf != None:
            road_inf = left_line.road_inf
            curvature = left_line.curvature
        else:
            road_inf = 'None'
            curvature = curvature

    center_lane = (right_line.startx + left_line.startx) / 2
    lane_width = right_line.startx - left_line.startx

    center_car = 570 / 2
    if center_lane > center_car:
        deviation = round(abs(center_lane - center_car) /
                          (lane_width / 2)*100, 3)
    elif center_lane < center_car:
        deviation = round(abs(center_lane - center_car) /
                          (lane_width / 2)*100, 3)
    else:
        deviation = 50
    left_line.road_inf = road_inf
    left_line.curvature = curvature
    left_line.deviation = deviation

    return road_inf, curvature, deviation


def print_road_status(img, left_line, right_line):
    """ print road status (curve direction, radius of curvature, deviation) """
    road_inf, curvature, deviation = road_info(left_line, right_line)
    # cv2.putText(img, 'Road Status', (22, 30),
    #             cv2.FONT_HERSHEY_COMPLEX, 0.7, (80, 80, 80), 2)
    curve_rate = None

    if deviation > 50:
        curve_rate = str(int(deviation - 50)) + '% '
    elif deviation < 50:
        curve_rate = str(int(50 - deviation)) + '% '
    elif deviation == 0:
        curve_rate = '직진'
        road_inf = ''

    if road_inf == '직진':
        curve_rate = ''

    lane_inf = 'Servo 모터 : ' + curve_rate + road_inf

    # curve_rate = 50 - deviation

    print('Main 모터 : 직진 ', lane_inf)
    # print(lane_curve)

    return img


def print_road_map(image, left_line, right_line):
    """ 미니 맵 """
    img = cv2.imread('images/top_view_car.png', -1)
    img = cv2.resize(img, (120, 246))

    rows, cols = image.shape[:2]
    window_img = np.zeros_like(image)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally
    lane_width = right_line.startx - left_line.startx
    lane_center = (right_line.startx + left_line.startx) / 2
    lane_offset = cols / 2 - (2*left_line.startx + lane_width) / 2
    car_offset = int(lane_center - 570)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_pts_l = np.array([np.transpose(np.vstack(
        [right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack(
        [right_plotx + lane_offset - lane_width + window_margin / 4, ploty])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))
    right_pts_l = np.array(
        [np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
    right_pts_r = np.array([np.flipud(np.transpose(
        np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_pts]), (140, 0, 170))
    cv2.fillPoly(window_img, np.int_([right_pts]), (140, 0, 170))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack(
        [right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(
        np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

    #window_img[10:133,300:570] = img
    road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
    window_img = Image.fromarray(window_img)
    img = Image.fromarray(img)
    road_map.paste(window_img, (0, 0))
    road_map.paste(img, (300-car_offset, 590), mask=img)
    road_map = np.array(road_map)
    road_map = cv2.resize(road_map, (95, 95))
    road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)
    return road_map

def onMouse(x):
    pass
