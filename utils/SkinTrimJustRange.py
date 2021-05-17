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
import random

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

from core.const_var import VIDEO_YCrCb, VIDEO_Lab, VIDEO_Melt, VIDEO_RGB, VIDEO_HSV
from utils.hsv_rgb_ycrcb import Skin_Detect


class SkinTrimUtils:
    _sd = Skin_Detect()

    @staticmethod
    def _maskAndErodeAndDilateAndSmooth(srcImg, colorSpaceImg, min, max, kernelSize=11, iteration=2):
        """
        掩码后，执行收缩膨胀，最后平滑处理
        :param colorSpaceImg:
        :param min:
        :param max:
        :return:
        """
        skinMaskRange = cv2.inRange(colorSpaceImg, min, max)
        skinMaskAfter = SkinTrimUtils._getMask(skinMaskRange, kernelSize, iteration)
        skin = cv2.bitwise_and(srcImg, srcImg, mask=skinMaskAfter)
        return skin, skinMaskRange, skinMaskAfter

    @staticmethod
    def rgb_hsv_ycbcr(image, kernelSize=11, iteration=2):
        """
        数据来源：
        Nusirwan Anwar bin Abdul Rahman, Kit Chong Wei and John See†
        RGB-H-CbCr Skin Colour Model for Human Face Detection
        Faculty of Information Technology, Multimedia University
        johnsee@mmu.edu.my†
        :param image:
        :return:
        """
        skinMaskRange = SkinTrimUtils._sd.RGB_H_CbCr(image, False)
        skinMaskRange = skinMaskRange.astype(np.uint8)
        skinMaskRange *= 255
        skinMaskAfter = SkinTrimUtils._getMask(skinMaskRange, kernelSize, iteration)
        skin = cv2.bitwise_and(image, image, mask=skinMaskAfter)
        return skin, skinMaskRange, skinMaskAfter



    @staticmethod
    def _getMask(skinMask, kernelSize=11, iterations=2):
        if kernelSize is not None and iterations is not None and kernelSize > 0 and iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize, kernelSize))
            skinMask = cv2.erode(skinMask, kernel, iterations=iterations)
            skinMask = cv2.dilate(skinMask, kernel, iterations=iterations)

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

    min_rgb = [50, 80, 100]
    max_rgb = [120, 255, 160]
    # min_rgb = [100, 80, 50]
    # max_rgb = [160, 255, 120]

    min_Lab = [0, 128, 127]
    max_Lab = [235, 143, 158]

    min_HSV = [0, 51, 40]
    max_HSV = [25, 255, 255]

    min_YCrCb = [0, 133, 77]
    max_YCrCb = [255, 173, 127]

    @staticmethod
    def rgb(image, minRange=None, maxRange=None, kernelSize=None, iteration=None):
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

        minRange = np.asarray(minRange, dtype=np.uint8)
        maxRange = np.asarray(maxRange, dtype=np.uint8)
        skinRGB, skinMaskRange, skinMaskAfter = SkinTrimUtils._maskAndErodeAndDilateAndSmooth(image, image, minRange,
                                                                                              maxRange, kernelSize,
                                                                                              iteration)
        # return np.hstack([image, skinRGB])
        return skinRGB, skinMaskRange, skinMaskAfter

    @staticmethod
    def Lab(image, minRange=None, maxRange=None, kernelSize=None, iteration=None):
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

        minRange = np.asarray(minRange, dtype=np.uint8)
        maxRange = np.asarray(maxRange, dtype=np.uint8)

        skinLab, skinMaskRange, skinMaskAfter = SkinTrimUtils._maskAndErodeAndDilateAndSmooth(image, imageLab, minRange,
                                                                                              maxRange, kernelSize,
                                                                                              iteration)

        # return np.hstack([image, skinLab])
        return skinLab, skinMaskRange, skinMaskAfter

    @staticmethod
    def hsv(image, minRange=None, maxRange=None, kernelSize=None, iteration=None):
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

        minRange = np.asarray(minRange, dtype=np.uint8)
        maxRange = np.asarray(maxRange, dtype=np.uint8)
        # Get pointer to video frames from primary device
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skinHSV, skinMaskRange, skinMaskAfter = SkinTrimUtils._maskAndErodeAndDilateAndSmooth(image, imageHSV, minRange,
                                                                                              maxRange, kernelSize,
                                                                                              iteration)
        return skinHSV, skinMaskRange, skinMaskAfter

    @staticmethod
    def trimByParam(img, skinParamDict):
        mode = skinParamDict['currentMode']
        paramDict = skinParamDict['paramDict']
        param = paramDict[mode]
        if mode == VIDEO_RGB:
            skin, mask1, mask2 = SkinTrimUtils.rgb(img,
                                                   minRange=param['tmin'],
                                                   maxRange=param['tmax'],
                                                   kernelSize=param['KernelSize'],
                                                   iteration=param['Iterations']
                                                   )
        elif mode == VIDEO_HSV:
            skin, mask1, mask2 = SkinTrimUtils.hsv(img,
                                                   minRange=param['tmin'],
                                                   maxRange=param['tmax'],
                                                   kernelSize=param['KernelSize'],
                                                   iteration=param['Iterations']
                                                   )

        elif mode == VIDEO_Lab:
            skin, mask1, mask2 = SkinTrimUtils.Lab(img,
                                                   minRange=param['tmin'],
                                                   maxRange=param['tmax'],
                                                   kernelSize=param['KernelSize'],
                                                   iteration=param['Iterations']
                                                   )
        elif mode == VIDEO_YCrCb:
            skin, mask1, mask2 = SkinTrimUtils.YCrCb(img,
                                                     minRange=param['tmin'],
                                                     maxRange=param['tmax'],
                                                     kernelSize=param['KernelSize'],
                                                     iteration=param['Iterations']
                                                     )
        elif mode == VIDEO_Melt:
            skin, mask1, mask2 = SkinTrimUtils.rgb_hsv_ycbcr(img,
                                                             kernelSize=param['KernelSize'],
                                                             iteration=param['Iterations']
                                                             )
        else:
            raise Exception("没有指定模式！或者模式不存在")

        return skin, mask1, mask2

    @staticmethod
    def YCrCb(image, minRange=None, maxRange=None, kernelSize=None, iteration=None):
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

        minRange = np.asarray(minRange, dtype=np.uint8)
        maxRange = np.asarray(maxRange, dtype=np.uint8)

        imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        skinYCrCb, skinMaskRange, skinMaskAfter = SkinTrimUtils._maskAndErodeAndDilateAndSmooth(image, imageYCrCb,
                                                                                                minRange, maxRange,
                                                                                                kernelSize,
                                                                                                iteration)
        return skinYCrCb, skinMaskRange, skinMaskAfter


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
    def showImg(title, skin):
        cv2.imshow(title + "skin", skin[0])
        cv2.imshow(title + " range mask", skin[1])

    # frame = cv2.imread("../faces/7.jpeg")
    frame = cv2.imread("../faces/white.jpg")
    frame = imutils.resize(frame, width=300)
    # np.ones((5, 5), np.uint8)
    cv2.imshow("src", frame)
    showImg("rgb", SkinTrimUtils.rgb(frame.copy()))
    showImg("lab", SkinTrimUtils.Lab(frame.copy()))
    showImg("ycrcb", SkinTrimUtils.YCrCb(frame.copy()))
    showImg("hsv", SkinTrimUtils.hsv(frame.copy()))
    showImg("melt", SkinTrimUtils.rgb_hsv_ycbcr(frame.copy()))
    # SkinTrimUtils.hsv(frame.copy()))
    # cv2.imshow('yCrCb', SkinTrimUtils.YCrCb(frame.copy()))
    # cv2.imshow('lab result', SkinTrimUtils.Lab(frame.copy()))
    # cv2.imshow('rgb_hsv_ycbcr', SkinTrimUtils.rgb_hsv_ycbcr(frame.copy()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # testVideo()
    testImage()
