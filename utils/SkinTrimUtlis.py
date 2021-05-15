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

# OpenCV 文档 https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

# Required modules
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from utils.hsv_rgb_ycrcb import Skin_Detect


class SkinTrimUtils:
    _sd = Skin_Detect()

    @staticmethod
    def _maskAndErodeAndDilateAndSmooth(srcImg, colorSpaceImg, min, max):
        """
        掩码后，执行收缩膨胀，最后平滑处理
        :param colorSpaceImg:
        :param min:
        :param max:
        :return:
        """
        skinMask = cv2.inRange(colorSpaceImg, min, max)
        skinMask = SkinTrimUtils._getMask(skinMask)
        skin = cv2.bitwise_and(srcImg, srcImg, mask=skinMask)
        return skin

    @staticmethod
    def rgb_hsv_ycbcr(image):
        """
        数据来源：
        Nusirwan Anwar bin Abdul Rahman, Kit Chong Wei and John See†
        RGB-H-CbCr Skin Colour Model for Human Face Detection
        Faculty of Information Technology, Multimedia University
        johnsee@mmu.edu.my†
        :param image:
        :return:
        """
        skinMask = SkinTrimUtils._sd.RGB_H_CbCr(image, False)
        skinMask = SkinTrimUtils._getMask(skinMask)
        skin = cv2.bitwise_and(image, image, mask=skinMask)
        return skin

    @staticmethod
    def _getMask(skinMask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        return skinMask

    @staticmethod
    def rgb_normalization(image):
        """
        数据来源
        Gomez and Morales [20] produced a rule for skin
        detection using normalized RGB, formulated as equation 4
        to equation 6.
        R' = R /(R+G+B)
        G' = G /(R+G+B)
        B' = B /(R+G+B)

        (R' / G' )>  1.185
        (R' * G') / (R' + G' + B')^2 > 0.107
        (R' * G') / (R' + G' + B')^2 > 0.112
        :param image:
        :return:
        """

    min_rgb = np.array([50, 80, 100], np.uint8)
    max_rgb = np.array([120, 255, 160], np.uint8)

    min_Lab = np.array([0, 128, 127], np.uint8)
    max_Lab = np.array([235, 143, 158], np.uint8)

    min_HSV = np.array([0, 51, 40], dtype=np.uint8)
    max_HSV = np.array([13, 255, 255], dtype=np.uint8)

    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)



    @staticmethod
    def rgb(image, minRange=None, maxRange=None):
        """
        数据来源：
        Available online at www.sciencedirect.com
        Manuel C. Sanchez-Cuevas, Ruth M. Aguilar-Ponce, J. Luis Tecpanecatl-Xihuitl
        A Comparison of Color Models for Color Face Segmentation
        100 < R < 160, 80 < G <, 50 < B < 120
        :param img:
        :return:
        """
        if minRange is None: minRange = SkinTrimUtils.min_rgb
        if maxRange is None: maxRange = SkinTrimUtils.max_rgb
        skinRGB = SkinTrimUtils._maskAndErodeAndDilateAndSmooth(image, image, minRange, maxRange)
        # return np.hstack([image, skinRGB])
        return skinRGB

    @staticmethod
    def Lab(image, minRange=None, maxRange=None):
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

        # min_Lab = np.array([127, 128, 0], np.uint8)
        # max_Lab = np.array([158, 143, 255], np.uint8)

        imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        if minRange is None: minRange = SkinTrimUtils.min_Lab
        if maxRange is None: maxRange = SkinTrimUtils.max_Lab
        skinLab = SkinTrimUtils._maskAndErodeAndDilateAndSmooth(image, imageLab, minRange, maxRange)

        # return np.hstack([image, skinLab])
        return skinLab

    @staticmethod
    def hsv(image, minRange=None, maxRange=None):
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
        if minRange is None: minRange = SkinTrimUtils.min_HSV
        if maxRange is None: maxRange = SkinTrimUtils.max_HSV

        # Get pointer to video frames from primary device
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skinHSV = SkinTrimUtils._maskAndErodeAndDilateAndSmooth(image, imageHSV, minRange,
                                                                maxRange)
        return skinHSV

    @staticmethod
    def YCrCb(image, minRange, maxRange):
        """
        相关论文
        [1]Vladimir Vezhnevets ∗ Vassili Sazonov. Alla Andreeva. Comparison between YCbCr Color Space and CIELab Color Space for Skin Color Segmentation[D]. International Conference Graphicon 2003. Moscow, Russia. http://www.graphicon.ru/
        [2]
        范围来源:
        Chai, D. and Ngan, K. N. (1999). Face segmentation using skincolor
        map in videophone applications. IEEE Transactions on
        Circuits and Systems for Video Technology. 9(4), pp. 551-564.

        范围：133 < Cr < 173; 77 < Cb < 127
        :param image:
        :return:
        """
        if minRange is None: minRange = SkinTrimUtils.min_YCrCb
        if maxRange is None: maxRange = SkinTrimUtils.max_YCrCb
        imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        skinYCrCb = SkinTrimUtils._maskAndErodeAndDilateAndSmooth(image, imageYCrCb, minRange, maxRange)
        return skinYCrCb


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
        cv2.imshow('rgb', SkinTrimUtils.rgb(frame))
        cv2.imshow('hsv', SkinTrimUtils.hsv(frame))
        cv2.imshow('yCrCb', SkinTrimUtils.YCrCb(frame))
        cv2.imshow('lab', SkinTrimUtils.Lab(frame))
        cv2.imshow('rgb_hsv_ycbcr', SkinTrimUtils.rgb_hsv_ycbcr(frame))
        # cv2.imshow('melt', melt(frame))

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()


def testImage():
    # frame = cv2.imread("../faces/7.jpeg")
    frame = cv2.imread("../faces/dark.jpg")
    frame = imutils.resize(frame, width=600)
    cv2.imshow('rgb', SkinTrimUtils.rgb(frame.copy()))
    cv2.imshow('hsv', SkinTrimUtils.hsv(frame.copy()))
    cv2.imshow('yCrCb', SkinTrimUtils.YCrCb(frame.copy()))
    cv2.imshow('lab result', SkinTrimUtils.Lab(frame.copy()))
    cv2.imshow('rgb_hsv_ycbcr', SkinTrimUtils.rgb_hsv_ycbcr(frame.copy()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # testVideo()
    testImage()
