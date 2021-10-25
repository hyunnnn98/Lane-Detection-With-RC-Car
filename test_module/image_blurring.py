import numpy as np
import cv2

def onMouse(x):
    pass

def bluring():
    img = cv2.imread('assets/y_playground_1.png')

    cv2.namedWindow('Window')
    cv2.createTrackbar('RED', 'Window', 0, 255, onMouse)
    cv2.createTrackbar('BLUE', 'Window', 0, 255, onMouse)
    cv2.createTrackbar('GREEN', 'Window', 0, 255, onMouse)

    custom_red = cv2.getTrackbarPos('RED', 'Window')
    custom_blue = cv2.getTrackbarPos('BLUE', 'Window')
    custom_green = cv2.getTrackbarPos('GREEN', 'Window')

    while True:
        cv2.imshow('Blur', img)
        print(custom_red, custom_blue, custom_green)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        custom_red = cv2.getTrackbarPos('RED', 'Window')
        custom_blue = cv2.getTrackbarPos('BLUE', 'Window')
        custom_green = cv2.getTrackbarPos('GREEN', 'Window')

    cv2.destroyAllWindows()

bluring()
