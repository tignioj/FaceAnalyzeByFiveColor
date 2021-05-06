import matplotlib.pyplot as plt
import numpy as  np
import cv2

# https://blog.csdn.net/wsp_1138886114/article/details/80660014

def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256);

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
    return histImg


original_img = cv2.imread("../../faces/7.jpeg")
img = cv2.resize(original_img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
b, g, r = cv2.split(img)
histImgB = calcAndDrawHist(b, [255, 0, 0])
histImgG = calcAndDrawHist(g, [0, 255, 0])
histImgR = calcAndDrawHist(r, [0, 0, 255])
