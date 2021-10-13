import numpy as np
import cv2
from PIL import Image
import matplotlib.image as mpimg


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 60
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
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


def warp_image(img, src, dst, size):
    """ Perspective Transform """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv


def rad_of_curvature(left_line, right_line):
    """ measure radius of curvature  """

    ploty = left_line.ally
    leftx, rightx = left_line.allx, right_line.allx

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Define conversions in x and y from pixels space to meters
    width_lanes = abs(right_line.startx - left_line.startx)
    ym_per_pix = 22 / 720  # meters per pixel in y dimension
    # meters per pixel in x dimension
    xm_per_pix = 3.7*(720/1280) / width_lanes

    # Define y-value where we want radius of curvature
    # the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # radius of curvature result
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad


def smoothing(lines, pre_lines=3):
    # collect lines & print average line
    lines = np.squeeze(lines)
    avg_line = np.zeros((720))

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line


def blind_search(b_img, left_line, right_line):
    """
    blind search - first frame, lost lane lines
    using histogram & sliding window
    give different weight in color info(0.8) & gradient info(0.2) using weighted average
    """
    # Create an output image to draw on and  visualize the result
    # output = np.dstack((b_img, b_img, b_img)) * 255
    output = cv2.cvtColor(b_img, cv2.COLOR_GRAY2RGB)

    # Choose the number of sliding windows
    num_windows = 9
    # Set height of windows
    window_height = np.int(b_img.shape[0] / num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if left_line.startx == None:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(b_img[int(b_img.shape[0] * 2 / 3):, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        start_leftX = np.argmax(histogram[:midpoint])
        start_rightX = np.argmax(histogram[midpoint:]) + midpoint

        # Current positions to be updated for each window
        current_leftX = start_leftX
        current_rightX = start_rightX
    else:
        current_leftX = left_line.startx
        current_rightX = right_line.startx

    # Set minimum number of pixels found to recenter window
    min_num_pixel = 50

    # Create empty lists to receive left and right lane pixel indices
    win_left_lane = []
    win_right_lane = []

    left_weight_x, left_weight_y = [], []
    right_weight_x, right_weight_y = [], []
    window_margin = left_line.window_margin

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = b_img.shape[0] - (window + 1) * window_height
        win_y_high = b_img.shape[0] - window * window_height
        win_leftx_min = int(current_leftX - window_margin)
        win_leftx_max = int(current_leftX + window_margin)
        win_rightx_min = int(current_rightX - window_margin)
        win_rightx_max = int(current_rightX + window_margin)

        if win_rightx_max > 720:
            win_rightx_min = b_img.shape[1] - 2 * window_margin
            win_rightx_max = b_img.shape[1]

        # Draw the windows on the visualization image
        cv2.rectangle(output, (win_leftx_min, win_y_low),
                      (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_rightx_min, win_y_low),
                      (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
            nonzerox <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
            nonzerox <= win_rightx_max)).nonzero()[0]
        # Append these indices to the lists
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_inds) > min_num_pixel:

            win = b_img[win_y_low:win_y_high, win_leftx_min:win_leftx_max]
            temp, count_g, count_h = 0, 0, 0
            for i in range(win.shape[1]):
                for j in range(win.shape[0]):
                    if win[j, i] >= 70 and win[j, i] <= 130:
                        temp += 0.2 * (i + win_leftx_min)
                        count_g += 1
                        output[j + win_y_low, i + win_leftx_min] = (255, 0, 0)
                    elif win[j, i] > 220:
                        temp += 0.8 * (i + win_leftx_min)
                        count_h += 1
                        output[j + win_y_low, i + win_leftx_min] = (0, 0, 255)
                        # else:
                        #    output[j + win_y_low, i + win_leftx_min] = (255, 255, 255)
            if not (count_h == 0 and count_g == 0):
                left_w_x = temp / (0.2 * count_g + 0.8 *
                                   count_h)  # + win_leftx_min
                #cv2.circle(output, (int(left_w_x), int((win_y_low + win_y_high) / 2)), 10, (255, 0, 0), -1)
                #cv2.circle(output, (int(current_leftX), int((win_y_low + win_y_high) / 2)), 10, (255, 0, 0), -1)
                left_weight_x.append(int(left_w_x))
                left_weight_y.append(int((win_y_low + win_y_high) / 2))

                current_leftX = int(left_w_x)

        if len(right_window_inds) > min_num_pixel:

            win = b_img[win_y_low:win_y_high, win_rightx_min:win_rightx_max]
            temp, count_g, count_h = 0, 0, 0
            for i in range(win.shape[1]):
                for j in range(win.shape[0]):
                    if win[j, i] >= 70 and win[j, i] <= 130:
                        temp += 0.2 * (i + win_rightx_min)
                        count_g += 1
                        output[j + win_y_low, i + win_rightx_min] = (255, 0, 0)
                    elif win[j, i] > 200:
                        temp += 0.8 * (i + win_rightx_min)
                        count_h += 1
                        output[j + win_y_low, i + win_rightx_min] = (0, 0, 255)
                        # else:
                        #    output[j + win_y_low, i + win_rightx_min] = (255, 255, 255)
            if not (count_h == 0 and count_g == 0):
                right_w_x = temp / (0.2 * count_g + 0.8 *
                                    count_h)  # + win_leftx_min
                #cv2.circle(output, (int(right_w_x), int((win_y_low + win_y_high) / 2)), 10, (255, 0, 0), -1)
                #cv2.circle(output, (int(current_rightX), int((win_y_low + win_y_high) / 2)), 10, (255, 0, 0), -1)
                right_weight_x.append(int(right_w_x))
                right_weight_y.append(int((win_y_low + win_y_high) / 2))
                current_rightX = int(right_w_x)

    # Concatenate the arrays of indices
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
    rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

    #output[lefty, leftx] = [255, 0, 0]
    #output[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_weight_y, left_weight_x, 2)
    right_fit = np.polyfit(right_weight_y, right_weight_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    # frame to frame smoothing
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
    # print radius of curvature
    rad_of_curvature(left_line, right_line)
    return output


def prev_window_refer(b_img, left_line, right_line):
    """
    refer to previous window info - after detecting lane lines in previous frame
    give different weight in color info(0.8) & gradient info(0.2) using weighted average
    """
    # Create an output image to draw on and  visualize the result
    output = cv2.cvtColor(b_img, cv2.COLOR_GRAY2RGB)

    # Set margin of windows
    window_margin = left_line.window_margin

    left_weight_x, left_weight_y = [], []
    right_weight_x, right_weight_y = [], []

    temp, count_g, count_h = 0, 0, 0
    for i, j in enumerate(left_line.allx):
        for m in range(window_margin):
            j1, j2 = int(j) + m, int(j) - m

            if b_img[i, j1] >= 70 and b_img[i, j1] <= 130:
                temp += 0.2 * j1
                count_g += 1
                output[i, j1] = (255, 0, 0)
            if b_img[i, j2] >= 70 and b_img[i, j2] <= 130:
                temp += 0.2 * j2
                count_g += 1
                output[i, j2] = (255, 0, 0)
            if b_img[i, j1] > 220:
                temp += 0.8 * j1
                count_h += 1
                output[i, j1] = (0, 0, 255)
            if b_img[i, j2] > 220:
                temp += 0.8 * j2
                count_h += 1
                output[i, j2] = (0, 0, 255)
        if (i+1) % 80 == 0:
            if not (count_h == 0 and count_g == 0):
                left_w_x = temp / (0.2 * count_g + 0.8 *
                                   count_h)  # + win_leftx_min
                #cv2.circle(output, (int(left_w_x), (i+1-40)), 10, (255, 0, 0), -1)
                left_weight_x.append(int(left_w_x))
                left_weight_y.append((i+1-40))
            temp, count_g, count_h = 0, 0, 0

    temp, count_g, count_h = 0, 0, 0
    for i, j in enumerate(right_line.allx):
        if j >= 720 - (window_margin):
            for m in range(2*(window_margin)):
                k = 720 - 2*(window_margin) + m
                if b_img[i, k] >= 70 and b_img[i, k] <= 130:
                    temp += 0.2 * k
                    count_g += 1
                    output[i, k] = (255, 0, 0)
                if b_img[i, k] > 220:
                    temp += 0.8 * k
                    count_h += 1
                    output[i, k] = (0, 0, 255)
        else:
            for m in range(window_margin):
                j1, j2 = int(j) + m, int(j) - m
                if b_img[i, j1] >= 70 and b_img[i, j1] <= 130:
                    temp += 0.2 * j1
                    count_g += 1
                    output[i, j1] = (255, 0, 0)
                if b_img[i, j2] >= 70 and b_img[i, j2] <= 130:
                    temp += 0.2 * j2
                    count_g += 1
                    output[i, j2] = (255, 0, 0)
                if b_img[i, j1] > 220:
                    temp += 0.8 * j1
                    count_h += 1
                    output[i, j1] = (0, 0, 255)
                if b_img[i, j2] > 220:
                    temp += 0.8 * j2
                    count_h += 1
                    output[i, j2] = (0, 0, 255)
        if (i + 1) % 80 == 0:
            if not (count_h == 0 and count_g == 0):
                right_w_x = temp / (0.2 * count_g + 0.8 * count_h)
                #cv2.circle(output, (int(right_w_x), (i+1-40)), 10, (255, 0, 0), -1)
                right_weight_x.append(int(right_w_x))
                right_weight_y.append((i+1-40))
            temp, count_g, count_h = 0, 0, 0

    #output[lefty, leftx] = [255, 0, 0]
    #output[righty, rightx] = [0, 0, 255]

    if len(left_weight_x) <= 5:
        left_weight_x = left_line.allx
        left_weight_y = left_line.ally
    if len(right_weight_x) <= 5:
        right_weight_x = right_line.allx
        right_weight_y = right_line.ally

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_weight_y, left_weight_x, 2)
    right_fit = np.polyfit(right_weight_y, right_weight_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    # frame to frame smoothing
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

    # if don't have lane lines info
    if left_line.detected == False:
        return blind_search(binary_img, left_line, right_line)
    # if have lane lines info
    else:
        return prev_window_refer(binary_img, left_line, right_line)


def draw_lane(img, left_line, right_line, lane_color=(255, 0, 255), road_color=(0, 255, 0)):
    """ draw lane lines & current driving space """
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

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

    return result, window_img


def road_info(left_line, right_line):
    """ print road information onto result image """
    curvature = (left_line.radius_of_curvature +
                 right_line.radius_of_curvature) / 2

    direction = ((left_line.endx - left_line.startx) +
                 (right_line.endx - right_line.startx)) / 2
    #print('direction : ', direction, 'curvature : ',curvature)
    if curvature > 2100:  # and abs(direction) < 80:
        road_inf = 'No Curve'
        curvature = -1
    elif curvature <= 2100 and direction < - 50:
        road_inf = 'Left Curve'
    elif curvature <= 2100 and direction > 50:
        road_inf = 'Right Curve'
    else:
        if left_line.road_inf != None:
            road_inf = left_line.road_inf
            curvature = left_line.curvature
        else:
            road_inf = 'None'
            curvature = curvature

    center_lane = (right_line.startx + left_line.startx) / 2
    lane_width = right_line.startx - left_line.startx

    center_car = 720 / 2
    if center_lane > center_car:
        deviation = 'Left ' + \
            str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
    elif center_lane < center_car:
        deviation = 'Right ' + \
            str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
    else:
        deviation = 'Center'
    left_line.road_inf = road_inf
    left_line.curvature = curvature
    left_line.deviation = deviation

    return road_inf, curvature, deviation


def print_road_status(img, left_line, right_line):
    """ print road status (curve direction, radius of curvature, deviation) """
    road_inf, curvature, deviation = road_info(left_line, right_line)
    cv2.putText(img, 'Road Status', (22, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (80, 80, 80), 2)

    lane_inf = 'Lane Info : ' + road_inf
    if curvature == -1:
        lane_curve = 'Curvature : Straight line'
    else:
        lane_curve = 'Curvature : {0:0.3f}m'.format(curvature)
    deviate = 'Deviation : ' + deviation

    cv2.putText(img, lane_inf, (10, 63), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100, 100, 100), 1)
    cv2.putText(img, lane_curve, (10, 83),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(img, deviate, (10, 103), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100, 100, 100), 1)

    return img


def print_road_map(image, left_line, right_line):
    """ print simple road map """
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
    car_offset = int(lane_center - 360)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_pts_l = np.array([np.transpose(np.vstack(
        [right_plotx - lane_width + lane_offset - window_margin / 4, ploty]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack(
        [right_plotx - lane_width + lane_offset + window_margin / 4, ploty])))])
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
        [right_plotx - lane_width + lane_offset + window_margin / 4, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(
        np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

    #window_img[10:133,300:360] = img
    road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
    window_img = Image.fromarray(window_img)
    img = Image.fromarray(img)
    road_map.paste(window_img, (0, 0))
    road_map.paste(img, (300-car_offset, 590), mask=img)
    road_map = np.array(road_map)
    road_map = cv2.resize(road_map, (95, 95))
    road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)
    return road_map

# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.image as mpimg

# class Line:
#     def __init__(self):
#         # 마지막 반복에서 라인이 감지되었습니까?
#         self.detected = False
#         # Set the width of the windows +/- margin
#         self.window_margin = 45
#         # 마지막 n회 반복에 대한 적합선의 x 값
#         self.prevx = []
#         # 가장 최근 피팅에 대한 다항식 계수
#         self.current_fit = [np.array([False])]
#         # 일부 단위에서 선의 곡률 반경
#         self.radius_of_curvature = None
#         # starting x_value
#         self.startx = None
#         # ending x_value
#         self.endx = None
#         # x values for detected line pixels
#         self.allx = None
#         # y values for detected line pixels
#         self.ally = None
#         # road information
#         self.road_inf = None
#         self.curvature = None
#         self.deviation = None


# def warp_image(img, src, dst, size):
#     """ 원근 변환 """
#     M = cv2.getPerspectiveTransform(src, dst)
#     Minv = cv2.getPerspectiveTransform(dst, src)
#     warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

#     return warp_img, M, Minv


# def rad_of_curvature(left_line, right_line):
#     """ 곡률 반경 측정 """

#     ploty = left_line.ally
#     leftx, rightx = left_line.allx, right_line.allx

#     leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
#     rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

#     # Define conversions in x and y from pixels space to meters
#     # 픽셀 공간에서 미터로의 x 및 y 변환 정의
#     width_lanes = abs(right_line.startx - left_line.startx)
#     ym_per_pix = 30 / 720  # y 차원의 픽셀당 미터
#     # meters per pixel in x dimension
#     xm_per_pix = 3.7*(720/1280) / width_lanes


#     # 곡률 반경을 원하는 y 값을 정의
#     # 이미지 하단에 해당하는 최대 y값
#     y_eval = np.max(ploty)

#     # 공간에서 x, y에 새로운 다항식 맞추기
#     left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
#     right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

#     # 곡률 반경 계산
#     left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
#         2 * left_fit_cr[0])
#     right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
#         2 * right_fit_cr[0])

#     # 곡률 반경 결과
#     left_line.radius_of_curvature = left_curverad
#     right_line.radius_of_curvature = right_curverad


# def smoothing(lines, pre_lines=3):
#     # 라인 수집 및 평균 라인 그리기
#     lines = np.squeeze(lines)
#     avg_line = np.zeros((720))

#     for ii, line in enumerate(reversed(lines)):
#         if ii == pre_lines:
#             break
#         avg_line += line
#     avg_line = avg_line / pre_lines

#     return avg_line


# def blind_search(b_img, left_line, right_line):
#     """
#     블라인드 서치 - 첫 번째 프레임, 차선 이탈
#     히스토그램 및 슬라이딩 윈도우 사용
#     """

#     # 이미지의 아래쪽 절반의 히스토그램을 가져옵니다.
#     histogram = np.sum(b_img[int(b_img.shape[0] / 2):, :], axis=0)

#     # 드로잉할 출력 이미지 생성 및 결과 시각화
#     output = np.dstack((b_img, b_img, b_img)) * 255

#     # 히스토그램의 왼쪽과 오른쪽 절반의 피크를 찾습니다.
#     # 이것은 왼쪽과 오른쪽 라인의 시작점이 될 것입니다.
#     midpoint = np.int(histogram.shape[0] / 2)
#     start_leftX = np.argmax(histogram[:midpoint])
#     start_rightX = np.argmax(histogram[midpoint:]) + midpoint


#     # 슬라이딩 창 수 선택
#     num_windows = 9
#     # 창 높이 설정
#     window_height = np.int(b_img.shape[0] / num_windows)


#     # 이미지에서 0이 아닌 모든 픽셀의 x 및 y 위치 식별
#     nonzero = b_img.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])

#     # 각 창에 대해 업데이트될 현재 위치
#     current_leftX = start_leftX
#     current_rightX = start_rightX


#     # 창을 중앙에 맞추기 위해 찾은 최소 픽셀 수 설정
#     min_num_pixel = 50

#     # 왼쪽 및 오른쪽 차선 픽셀 인덱스를 수신하는 빈 목록 만들기
#     win_left_lane = []
#     win_right_lane = []

#     window_margin = left_line.window_margin

#     # 단계별 처리 시작
#     for window in range(num_windows):
#         # x 및 y(및 오른쪽 및 왼쪽)에서 창 경계 식별
#         win_y_low = b_img.shape[0] - (window + 1) * window_height
#         win_y_high = b_img.shape[0] - window * window_height
#         win_leftx_min = current_leftX - window_margin
#         win_leftx_max = current_leftX + window_margin
#         win_rightx_min = current_rightX - window_margin
#         win_rightx_max = current_rightX + window_margin

#         # 시각화 이미지에 창 그리기
#         cv2.rectangle(output, (win_leftx_min, win_y_low),
#                       (win_leftx_max, win_y_high), (0, 255, 0), 2)
#         cv2.rectangle(output, (win_rightx_min, win_y_low),
#                       (win_rightx_max, win_y_high), (0, 255, 0), 2)

#         # 창 내에서 x 및 y에서 0이 아닌 픽셀 식별
#         left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
#             nonzerox <= win_leftx_max)).nonzero()[0]
#         right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
#             nonzerox <= win_rightx_max)).nonzero()[0]
#         # 인덱스를 목록에 추가
#         win_left_lane.append(left_window_inds)
#         win_right_lane.append(right_window_inds)

#         # If you found > minpix pixels, recenter next window on their mean position
#         if len(left_window_inds) > min_num_pixel:
#             current_leftX = np.int(np.mean(nonzerox[left_window_inds]))
#         if len(right_window_inds) > min_num_pixel:
#             current_rightX = np.int(np.mean(nonzerox[right_window_inds]))

#     # 인덱스 배열 연결
#     win_left_lane = np.concatenate(win_left_lane)
#     win_right_lane = np.concatenate(win_right_lane)


#     # 왼쪽 및 오른쪽 라인 픽셀 위치 추출
#     leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
#     rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

#     output[lefty, leftx] = [255, 0, 0]
#     output[righty, rightx] = [0, 0, 255]


#     # 각각에 2차 다항식 맞추기
#     left_fit = np.polyfit(lefty, leftx, 2)
#     right_fit = np.polyfit(righty, rightx, 2)

#     left_line.current_fit = left_fit
#     right_line.current_fit = right_fit

#     # 플로팅을 위한 x 및 y 값 생성
#     ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

#     # ax^2 + bx + c
#     left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
#     right_plotx = right_fit[0] * ploty ** 2 + \
#         right_fit[1] * ploty + right_fit[2]

#     left_line.prevx.append(left_plotx)
#     right_line.prevx.append(right_plotx)

#     if len(left_line.prevx) > 10:
#         left_avg_line = smoothing(left_line.prevx, 10)
#         left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
#         left_fit_plotx = left_avg_fit[0] * ploty ** 2 + \
#             left_avg_fit[1] * ploty + left_avg_fit[2]
#         left_line.current_fit = left_avg_fit
#         left_line.allx, left_line.ally = left_fit_plotx, ploty
#     else:
#         left_line.current_fit = left_fit
#         left_line.allx, left_line.ally = left_plotx, ploty

#     if len(right_line.prevx) > 10:
#         right_avg_line = smoothing(right_line.prevx, 10)
#         right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
#         right_fit_plotx = right_avg_fit[0] * ploty ** 2 + \
#             right_avg_fit[1] * ploty + right_avg_fit[2]
#         right_line.current_fit = right_avg_fit
#         right_line.allx, right_line.ally = right_fit_plotx, ploty
#     else:
#         right_line.current_fit = right_fit
#         right_line.allx, right_line.ally = right_plotx, ploty

#     left_line.startx, right_line.startx = left_line.allx[len(
#         left_line.allx)-1], right_line.allx[len(right_line.allx)-1]
#     left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

#     left_line.detected, right_line.detected = True, True

#     # 곡률 반영
#     rad_of_curvature(left_line, right_line)
#     return output


# def prev_window_refer(b_img, left_line, right_line):
#     """
#     refer to previous window info - after detecting lane lines in previous frame
#     """
#     # Create an output image to draw on and  visualize the result
#     output = np.dstack((b_img, b_img, b_img)) * 255

#     # Identify the x and y positions of all nonzero pixels in the image
#     nonzero = b_img.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])

#     # Set margin of windows
#     window_margin = left_line.window_margin

#     left_line_fit = left_line.current_fit
#     right_line_fit = right_line.current_fit
#     leftx_min = left_line_fit[0] * nonzeroy ** 2 + \
#         left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
#     leftx_max = left_line_fit[0] * nonzeroy ** 2 + \
#         left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
#     rightx_min = right_line_fit[0] * nonzeroy ** 2 + \
#         right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
#     rightx_max = right_line_fit[0] * nonzeroy ** 2 + \
#         right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

#     # Identify the nonzero pixels in x and y within the window
#     left_inds = ((nonzerox >= leftx_min) & (
#         nonzerox <= leftx_max)).nonzero()[0]
#     right_inds = ((nonzerox >= rightx_min) & (
#         nonzerox <= rightx_max)).nonzero()[0]

#     # Extract left and right line pixel positions
#     leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
#     rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

#     output[lefty, leftx] = [255, 0, 0]
#     output[righty, rightx] = [0, 0, 255]

#     # Fit a second order polynomial to each
#     left_fit = np.polyfit(lefty, leftx, 2)
#     right_fit = np.polyfit(righty, rightx, 2)

#     # Generate x and y values for plotting
#     ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

#     # ax^2 + bx + c
#     left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
#     right_plotx = right_fit[0] * ploty ** 2 + \
#         right_fit[1] * ploty + right_fit[2]

#     leftx_avg = np.average(left_plotx)
#     rightx_avg = np.average(right_plotx)

#     left_line.prevx.append(left_plotx)
#     right_line.prevx.append(right_plotx)

#     if len(left_line.prevx) > 10:
#         left_avg_line = smoothing(left_line.prevx, 10)
#         left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
#         left_fit_plotx = left_avg_fit[0] * ploty ** 2 + \
#             left_avg_fit[1] * ploty + left_avg_fit[2]
#         left_line.current_fit = left_avg_fit
#         left_line.allx, left_line.ally = left_fit_plotx, ploty
#     else:
#         left_line.current_fit = left_fit
#         left_line.allx, left_line.ally = left_plotx, ploty

#     if len(right_line.prevx) > 10:
#         right_avg_line = smoothing(right_line.prevx, 10)
#         right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
#         right_fit_plotx = right_avg_fit[0] * ploty ** 2 + \
#             right_avg_fit[1] * ploty + right_avg_fit[2]
#         right_line.current_fit = right_avg_fit
#         right_line.allx, right_line.ally = right_fit_plotx, ploty
#     else:
#         right_line.current_fit = right_fit
#         right_line.allx, right_line.ally = right_plotx, ploty

#     # goto blind_search if the standard value of lane lines is high.
#     standard = np.std(right_line.allx - left_line.allx)

#     if (standard > 80):
#         left_line.detected = False

#     left_line.startx, right_line.startx = left_line.allx[len(
#         left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1]
#     left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

#     # print radius of curvature
#     rad_of_curvature(left_line, right_line)
#     return output


# def find_LR_lines(binary_img, left_line, right_line):
#     """
#     find left, right lines & isolate left, right lines
#     blind search - first frame, lost lane lines
#     previous window - after detecting lane lines in previous frame
#     """

#     # if don't have lane lines info
#     if left_line.detected == False:
#         return blind_search(binary_img, left_line, right_line)
#     # if have lane lines info
#     else:
#         return prev_window_refer(binary_img, left_line, right_line)


# def draw_lane(img, left_line, right_line, lane_color=(255, 0, 255), road_color=(0, 255, 0)):
#     """ draw lane lines & current driving space """
#     window_img = np.zeros_like(img)

#     window_margin = left_line.window_margin
#     left_plotx, right_plotx = left_line.allx, right_line.allx
#     ploty = left_line.ally

#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_pts_l = np.array(
#         [np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
#     left_pts_r = np.array(
#         [np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
#     left_pts = np.hstack((left_pts_l, left_pts_r))
#     right_pts_l = np.array(
#         [np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
#     right_pts_r = np.array(
#         [np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
#     right_pts = np.hstack((right_pts_l, right_pts_r))

#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
#     cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

#     # Recast the x and y points into usable format for cv2.fillPoly()
#     pts_left = np.array(
#         [np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
#     pts_right = np.array(
#         [np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
#     pts = np.hstack((pts_left, pts_right))

#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([pts]), road_color)
#     result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

#     return result, window_img


# def road_info(left_line, right_line):
#     """ print road information onto result image """
#     curvature = (left_line.radius_of_curvature +
#                  right_line.radius_of_curvature) / 2

#     direction = ((left_line.endx - left_line.startx) +
#                  (right_line.endx - right_line.startx)) / 2

#     if curvature > 2000 and abs(direction) < 100:
#         road_inf = 'No Curve'
#         curvature = -1
#     elif curvature <= 2000 and direction < - 50:
#         road_inf = 'Left Curve'
#     elif curvature <= 2000 and direction > 50:
#         road_inf = 'Right Curve'
#     else:
#         if left_line.road_inf != None:
#             road_inf = left_line.road_inf
#             curvature = left_line.curvature
#         else:
#             road_inf = 'None'
#             curvature = curvature

#     center_lane = (right_line.startx + left_line.startx) / 2
#     lane_width = right_line.startx - left_line.startx

#     center_car = 720 / 2
#     if center_lane > center_car:
#         deviation = 'Left ' + \
#             str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
#     elif center_lane < center_car:
#         deviation = 'Right ' + \
#             str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
#     else:
#         deviation = 'Center'
#     left_line.road_inf = road_inf
#     left_line.curvature = curvature
#     left_line.deviation = deviation

#     return road_inf, curvature, deviation


# def print_road_status(img, left_line, right_line):
#     """ print road status (curve direction, radius of curvature, deviation) """
#     road_inf, curvature, deviation = road_info(left_line, right_line)
#     # cv2.putText(img, 'Road Status', (22, 30),
#     #             cv2.FONT_HERSHEY_COMPLEX, 0.7, (80, 80, 80), 2)

#     lane_inf = 'Lane Info : ' + road_inf
#     if curvature == -1:
#         lane_curve = 'Curvature : Straight line'
#     else:
#         lane_curve = 'Curvature : {0:0.3f}m'.format(curvature)
#     deviate = 'Deviation : ' + deviation

#     cv2.putText(img, lane_inf, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.45, (33, 33, 33), 1)
#     cv2.putText(img, lane_curve, (10, 43),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (33, 33, 33), 1)
#     cv2.putText(img, deviate, (10, 63), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.45, (33, 33, 33), 1)

#     return img


# def print_road_map(image, left_line, right_line):
#     """ print simple road map """
#     img = cv2.imread('images/top_view_car.png', -1)
#     img = cv2.resize(img, (120, 246))

#     rows, cols = image.shape[:2]
#     window_img = np.zeros_like(image)

#     window_margin = left_line.window_margin
#     left_plotx, right_plotx = left_line.allx, right_line.allx
#     ploty = left_line.ally
#     lane_width = right_line.startx - left_line.startx
#     lane_center = (right_line.startx + left_line.startx) / 2
#     lane_offset = cols / 2 - (2*left_line.startx + lane_width) / 2
#     car_offset = int(lane_center - 360)
#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_pts_l = np.array([np.transpose(np.vstack(
#         [right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
#     left_pts_r = np.array([np.flipud(np.transpose(np.vstack(
#         [right_plotx + lane_offset - lane_width + window_margin / 4, ploty])))])
#     left_pts = np.hstack((left_pts_l, left_pts_r))
#     right_pts_l = np.array(
#         [np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
#     right_pts_r = np.array([np.flipud(np.transpose(
#         np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
#     right_pts = np.hstack((right_pts_l, right_pts_r))

#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([left_pts]), (140, 0, 170))
#     cv2.fillPoly(window_img, np.int_([right_pts]), (140, 0, 170))

#     # Recast the x and y points into usable format for cv2.fillPoly()
#     pts_left = np.array([np.transpose(np.vstack(
#         [right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
#     pts_right = np.array([np.flipud(np.transpose(
#         np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
#     pts = np.hstack((pts_left, pts_right))

#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

#     #window_img[10:133,300:360] = img
#     road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
#     window_img = Image.fromarray(window_img)
#     img = Image.fromarray(img)
#     road_map.paste(window_img, (0, 0))
#     road_map.paste(img, (300-car_offset, 590), mask=img)
#     road_map = np.array(road_map)
#     road_map = cv2.resize(road_map, (95, 95))
#     road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)
#     return road_map
