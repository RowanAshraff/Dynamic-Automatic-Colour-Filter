# User guide:
#   a left click on the screen chooses the filter colour
#   'f' activates and deactivates the filter
#   'a' activates filter 2D
#   'b' activates Gaussian blur
#   'c' activates median blur
#   'd' activates bilateral filter
#   'w' activates erode
#   'x' activates dilate
#   'y' activates opening
#   'z' activates closing
#   'r' resets everything
#   Esc exits the program


import cv2
import numpy as np
refPt = []
hsv = []
b = 0
g = 0
r = 0
c = 38
f = 1


def trackbar_callback(x):
    pass


def trackbar_on_image():
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 400, 240)

    # create new trackbars
    cv2.createTrackbar('l_blue', 'image', 0, 255, trackbar_callback)
    cv2.createTrackbar('u_blue', 'image', 0, 255, trackbar_callback)
    cv2.createTrackbar('l_green', 'image', 0, 255, trackbar_callback)
    cv2.createTrackbar('u_green', 'image', 0, 255, trackbar_callback)
    cv2.createTrackbar('l_red', 'image', 0, 255, trackbar_callback)
    cv2.createTrackbar('u_red', 'image', 0, 255, trackbar_callback)


def mix(hsv, l, u):
    # makes sure lower and upper values stay within range
    for i in range(3):
        if l[i] > 255:
            l[i] = 255
        elif l[i] < 0:
            l[i] = 0
        if u[i] > 255:
            u[i] = 255
        elif u[i] < 0:
            u[i] = 0
    mask1 = cv2.inRange(hsv, l, u)
    res1 = cv2.bitwise_and(frame, frame, mask=mask1)
    return res1


def click_event(event, x, y, flags, params):
    global b, g, r, c, hsv, c
    # checks for a left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Put the clicks coordinates in a variable
        refPt.append([y, x])
        if len(refPt) > 1:
                refPt.clear()
                refPt.append([y, x])
        b, g, r = hsv[y, x]
        # set the trackbars positions to new values
        cv2.setTrackbarPos('l_blue', 'image', b - c)
        cv2.setTrackbarPos('u_blue', 'image', b + c)
        cv2.setTrackbarPos('l_green', 'image', g - c)
        cv2.setTrackbarPos('u_green', 'image', g + c)
        cv2.setTrackbarPos('l_red', 'image', r - c)
        cv2.setTrackbarPos('u_red', 'image', r + c)


def blurring(n):
    # case 'a'
    if n == 1:
        blur = cv2.filter2D(frame, -1, kernel)
    # case 'b'
    elif n == 2:
        blur = cv2.GaussianBlur(frame, (9, 9), 0)
    # case 'c'
    elif n == 3:
        blur = cv2.medianBlur(frame, 9)
    # case 'd'
    elif n == 4:
        blur = cv2.bilateralFilter(frame, 9, 75, 75)
    else:
        blur = frame
    return blur


def morphologicalTransformations(m):
    # case 'w'
    if m == 1:
        img = cv2.erode(frame, kernel, iterations=1)
    # case 'x'
    elif m == 2:
        img = cv2.dilate(frame, kernel, iterations=1)
    # case 'y'
    elif m == 3:
        img = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    # case 'z'
    elif m == 4:
        img = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    else:
        img = frame
    return img


if __name__ == "__main__":
    trackbar_on_image()
    n = 0
    m = 0
    kernel = np.ones((5, 5), np.float32) / 25
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if len(refPt) >= 1 and f:
                l = np.array([b - c, g - c, r - c])
                u = np.array([b + c, g + c, r + c])
                frame = mix(hsv, l, u)
            frame = blurring(n)
            frame = morphologicalTransformations(m)
            cv2.setMouseCallback("frame", click_event)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == 27:
                break
            elif k == ord('a'):
                n = 1
            elif k == ord('b'):
                n = 2
            elif k == ord('c'):
                n = 3
            elif k == ord('d'):
                n = 4
            if k == ord('w'):
                m = 1
            elif k == ord('x'):
                m = 2
            elif k == ord('y'):
                m = 3
            elif k == ord('z'):
                m = 4
            if k == ord('f'):
                f = not f
            if k == ord('r'):
                n = 0
                m = 0
                refPt.clear()
    cv2.destroyAllWindows()
    cap.release()
