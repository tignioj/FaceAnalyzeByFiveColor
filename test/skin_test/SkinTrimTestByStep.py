"""
https://ieeexplore.ieee.org/document/7783247
Skin Color Segmentation Using
Multi-Color Space Threshold
1Romi Fadillah Rahmat, 2Tengku Chairunnisa, 3Dani Gunawan, 4Opim Salim Sitompul
1,2,3Department of Information Technology
Faculty of Computer Science and Information Technology
University of Sumatera Utara
Medan, Indonesia
1romi.fadillah@usu.ac.id
2tengku.chairunnisa@students.usu.ac.id
3danigunawan@usu.ac.id
4opim@usu.ac.id
978-1-5090-2549-7/16/$31.00 ©2016 IEEE
Skin Color Segmentation Using
Multi-Color Space Threshold
"""

# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

# Required modules
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from utils.hsv_rgb_ycrcb import Skin_Detect
from utils.ImageUtils import ImgUtils


class SkinTrimUtilsTestByStep:
    _sd = Skin_Detect()

    COLOR_SPACE_RGB = 0
    COLOR_SPACE_HSV = 1
    COLOR_SPACE_YCrCb = 2
    COLOR_SPACE_Lab = 3
    COLOR_SPACE_RGB_HSV_YCrCb = 4

    @staticmethod
    def getColorSpaceMask(image, colorType=COLOR_SPACE_RGB_HSV_YCrCb):
        minRange = np.array([0, 0, 0], np.uint8)
        maxRange = np.array([255, 255, 255], np.uint8)
        if colorType == SkinTrimUtilsTestByStep.COLOR_SPACE_RGB_HSV_YCrCb:
            """
               数据来源：
               Nusirwan Anwar bin Abdul Rahman, Kit Chong Wei and John See†
               RGB-H-CbCr Skin Colour Model for Human Face Detection
               Faculty of Information Technology, Multimedia University
               johnsee@mmu.edu.my†
               :param image:
               :return:
               """
            mask = SkinTrimUtilsTestByStep._sd.RGB_H_CbCr(image, False)
            mask = mask.astype(np.uint8)
            mask *= 255
            # mask = SkinTrimUtilsTestByStep.erode(mask)
            # mask = SkinTrimUtilsTestByStep.dilate(mask)
            return mask
        elif colorType == SkinTrimUtilsTestByStep.COLOR_SPACE_RGB:
            """
            数据来源：
            Available online at www.sciencedirect.com
            Manuel C. Sanchez-Cuevas, Ruth M. Aguilar-Ponce, J. Luis Tecpanecatl-Xihuitl
            A Comparison of Color Models for Color Face Segmentation
            100 < R < 160, 80 < G <, 50 < B < 120
            :param img:
            :return:
            """
            minRange = np.array([50, 80, 100], np.uint8)
            maxRange = np.array([120, 255, 160], np.uint8)
        elif colorType == SkinTrimUtilsTestByStep.COLOR_SPACE_Lab:
            """
            范围:
            This outputs 0≤L≤100, −127≤a≤127, −127≤b≤127 . The values are then converted to the destination data type:
            8-bit images: L←L∗255/100,a←a+128,b←b+128
            L:0-180
            a:0-255
            b:0-255
            :param image:
            :return:
            """
            minRange = np.array([0, 128, 127], np.uint8)
            maxRange = np.array([235, 143, 158], np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        elif colorType == SkinTrimUtilsTestByStep.COLOR_SPACE_HSV:
            """
            色相是颜色模型的颜色部分，并表示为0到360度之间的数字。在OpenCV中为0-180。定义主色[R，Y，G，C，B，M]
            饱和度是颜色中的灰色量，从0到100％。
            值与饱和度结合使用，并描述颜色的亮度或强度（0％至100％）。

            范围来源: Tsekeridou, S. and Pitas, I., “Facial feature extraction in frontal
            views using biometric analogie,” in 9th European Signal Processing
            Conference (EUSIPCO 1998), 1998, pp. 1-4

            In previous research, Tsekeridou and Pitas [22] had
            selected pixels having skin color by setting thresholds as
            换算关系：
            8-bit images: V←255V,S←255S,H←H/2(to fit to 0 to 255)
            标准:
            V >= 40,
            0.2 < S < 0.6
            0 < H < 25 OR 335< H < 360
            转换成OpenCV
            V >= 40
            51 < S < 153
            0 < H < 13或者 177 < H < 180
            :param image:
            :return:
            """
            # minRange = np.array([0, 51, 40], dtype="uint8")
            # maxRange = np.array([13, 255, 255], dtype="uint8")
            minRange = np.array([0, 0, 0], dtype="uint8")
            maxRange = np.array([13, 255, 255], dtype="uint8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif colorType == SkinTrimUtilsTestByStep.COLOR_SPACE_YCrCb:
            minRange = np.array([0, 133, 77], np.uint8)
            maxRange = np.array([255, 173, 127], np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        mask = cv2.inRange(image, minRange, maxRange)
        # mask = SkinTrimUtilsTestByStep.erode(mask)
        # mask = SkinTrimUtilsTestByStep.dilate(mask)
        return mask

    @staticmethod
    def erode(skinMask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=1)
        return skinMask

    @staticmethod
    def dilate(skinMask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.dilate(skinMask, kernel, iterations=1)
        return skinMask

    @staticmethod
    def gaussianBlur(skinMask):
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        return skinMask


def testVideo():
    videoCapture = cv2.VideoCapture(1)
    while videoCapture.isOpened():
        # image = cv2.imread("../..")
        flag, frame = videoCapture.read()
        if not flag:
            break

        frame = imutils.resize(frame, width=800)
        iteration = 0
        blur = (0, 0)
        cv2.imshow('rgb', SkinTrimUtilsTestByStep.rgb(frame))
        cv2.imshow('hsv', SkinTrimUtilsTestByStep.hsv(frame))
        cv2.imshow('yCrCb', SkinTrimUtilsTestByStep.YCrCb(frame))
        cv2.imshow('lab', SkinTrimUtilsTestByStep.Lab(frame))
        cv2.imshow('rgb_hsv_ycbcr', SkinTrimUtilsTestByStep.rgb_hsv_ycbcr(frame))
        # cv2.imshow('melt', melt(frame))

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()


def putTextTo(img, text):
    cv2.putText(img, text, (1, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), cv2.LINE_4)


def showImage(frame,title=None):
    # frame = cv2.imread("../../faces/7.jpeg")
    frame = imutils.resize(frame, width=250)
    cv2.imshow("original_" + title, frame)
    melt = SkinTrimUtilsTestByStep.getColorSpaceMask(frame, SkinTrimUtilsTestByStep.COLOR_SPACE_RGB_HSV_YCrCb)
    putTextTo(melt, "melt")
    hsv = SkinTrimUtilsTestByStep.getColorSpaceMask(frame, SkinTrimUtilsTestByStep.COLOR_SPACE_HSV)
    putTextTo(hsv, "hsv")
    rgb = SkinTrimUtilsTestByStep.getColorSpaceMask(frame, SkinTrimUtilsTestByStep.COLOR_SPACE_RGB)
    putTextTo(rgb, "rgb")
    lab = SkinTrimUtilsTestByStep.getColorSpaceMask(frame, SkinTrimUtilsTestByStep.COLOR_SPACE_Lab)
    putTextTo(lab, "lab")
    ycrcb = SkinTrimUtilsTestByStep.getColorSpaceMask(frame, SkinTrimUtilsTestByStep.COLOR_SPACE_YCrCb)
    putTextTo(ycrcb, "ycrcb")
    mask = np.hstack([melt, hsv, rgb, lab, ycrcb])
    cv2.imshow(title, mask)

def testImage():
    showImage(cv2.imread("../../faces/7.jpeg"), "1")
    showImage(cv2.imread("../../faces/white.jpg"), "2")
    showImage(cv2.imread("../../faces/deepdark.jpg"), "3")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # testVideo()
    testImage()
