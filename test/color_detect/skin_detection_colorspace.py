# import the necessary packages
import imutils
import numpy as np
import argparse
import cv2

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'

videoCapture = cv2.VideoCapture(1)


def hsv(frame):
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    # frame = imutils.resize(frame, width=)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    # show the skin in the image along with the mask
    img = np.hstack([frame, skin])
    img = imutils.resize(img, width=2000)
    return img


def cs(title="0x123", img=None):
    img = imutils.resize(img, width=2000)
    cv2.imshow(title, img)
    # cv2.waitKey(0)
    # cv2.destroyWindow(title)

# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

def YCrCb(frame):
    lower = np.array([0, 133, 77], dtype="uint8")
    upper = np.array([235, 179, 127], dtype="uint8")
    # frame = imutils.resize(frame, width=)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    # show the skin in the image along with the mask
    img = np.hstack([frame, skin])
    img = imutils.resize(img, width=2000)
    return img


# keep looping over the frames in the video
while videoCapture.isOpened():
    d, frame = videoCapture.read()
    if not d:
        break

    # img = hsv(frame)
    img = YCrCb(frame)
    cv2.imshow("hsv", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()
