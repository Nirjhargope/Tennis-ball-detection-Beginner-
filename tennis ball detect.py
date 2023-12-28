# import the necessary packages
from collections import deque
import numpy as np
import cv2
import time

greenLower = (30, 100, 100)
greenUpper = (50, 255, 255)
pts = deque(maxlen=20)

cap = cv2.VideoCapture(0)

time.sleep(2.0)

while True:

    ret, frame = cap.read()
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contrs = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contrs = contrs[0]
    center = None

    if  len(contrs) >  0:

        c = max(contrs, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m20"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 15:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)

    for i in range(1, len(pts)):

        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

