import cv2
import numpy as np
from playsound import playsound
import time
import datetime as dt
import os
import random

# the minimum distance between face and finger
# that will trigger an event
finger_distance = 90

# load classifier
face_cascade = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface.xml')

# connect to webcam
cap = cv2.VideoCapture(0)

while (True):
    # capture frame by frame
    ret, frame = cap.read()

    # -- FACE DETECTION --
    # convert the video frame into grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the frame  using cascade
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # draw a rectangles around faces
    center_face = ()
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cX = np.int(x + (w / 2))
        cY = np.int(y + (h / 2))
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1, 1)
        cv2.putText(frame, "face", (cX - 25, cY - 25),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
        center_face = (cX, cY)

    # -- HAND DETECTION --
    # convert the frame into HSV (hue, saturation, value) color format
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # detect blue in the frame
    # set the hsv ranges to detect blue
    min_blue = np.array([89, 98, 165])
    max_blue = np.array([117, 225, 255])

    # apply a blur for better edge detection
    hsv_frame = cv2.GaussianBlur(hsv_frame, (7, 7), 0)

    # create a mask
    mask = cv2.inRange(hsv_frame, min_blue, max_blue)

    # remove all pixels from the mask that are smaller than a 5x5 kernel
    mask = cv2.erode(mask, kernel=np.ones((5, 5), np.uint8))

    # bitwise to cut out all but mask
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw outlines on all the contours
    # cv2.drawContours(result, contours, -1, (0, 255, 255), 2)

    # create a bounding rectangle for the contours
    center_fingers = []  # array for center points of the fingers
    for contour in contours:
        # create a bounding rectangle for the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # draw a rectangle around the contours
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # put a dot in the middle
        cX = np.int(x + (w/2))
        cY = np.int(y + (h/2))
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1, 1)

        # add the center point of each contour to the array
        center_fingers.append([cX, cY])

        # add some text for flavor
        cv2.putText(frame, "finger", (cX - 25, cY - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # find the distance (D) between center of fingers and center of face
    if len(center_face) > 0 and len(center_fingers) > 0:

        for idx, finger in enumerate(center_fingers):

            dx = center_face[0] - finger[0]
            dy = center_face[1] - finger[1]

            D = round(np.sqrt(dx*dx+dy*dy), 2)  # pythagoras

            # draw a line between the finger and the face
            cv2.line(frame, center_face,
                     (finger[0], finger[1]), (255, 255, 255), 1)

            # write the distance from the face
            cv2.putText(frame, str(D), (finger[0] + 25, finger[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if D <= finger_distance:
                playsound(
                    f"./audio/{random.choice(os.listdir('./audio/'))}", block=False)
                cv2.imwrite(
                    f'./face_touches/face_touch_{dt.datetime.now().strftime("%Y%m%d%h%M%S")}.jpg', frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Press Q to quit
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
