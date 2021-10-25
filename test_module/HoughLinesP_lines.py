import cv2
import sys
import math
import cv2 as cv
import numpy as np

WIDTH, HEIGHT = 800, 600
WHITE_COLOR = (0, 255, 0, 10)

cap = cv2.VideoCapture("y_playground_video.mp4")


def region_of_interest(img, vertices, color3=WHITE_COLOR, color1=255):
    mask = np.zeros_like(img)

    # if len(img.shape) > 2:
    #     color = color3
    # else:
    #     color = color1

    cv2.fillPoly(mask, vertices, (255, 0, 0))

    ROI_image = cv2.bitwise_and(img, mask)

    return ROI_image


while (True):
    ret, src = cap.read()

    src = cv2.resize(src, (WIDTH, HEIGHT))

    # set canny
    dst = cv.Canny(src, 50, 150, None, 3)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)

    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    # set roi
    vertices = np.array([[(60, HEIGHT), (WIDTH/2-100, 0),
                          (WIDTH, HEIGHT), (WIDTH-50, HEIGHT)]], dtype=np.int32)

    dst = region_of_interest(dst, vertices)

    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #         cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    # 허프라인 적용
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 220, None, 100, 50)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]),
                    (0, 255, 0), 3, cv.LINE_AA)

    ############ 원본 영상 ###############

    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(src, (l[0], l[1]), (l[2], l[3]),
    #                 (0, 255, 0), 3, cv.LINE_AA)

    # cv.imshow("Source", src)

    ############ 원본 영상 ###############

    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
