import json
import threading
import time

import imutils
from matplotlib import pyplot as plt

from utils.ImageUtils import ImgUtils
from core.const_var import *
from core.const_var import FACIAL_LANDMARKS_NAME_DICT

import cv2

# 官方地址 https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# 颜色空间的有关问题：http://poynton.ca/notes/colour_and_gamma/ColorFAQ.html
from utils.LogUtils import LogUtils
from utils.SkinTrimUtlis import SkinTrimUtils
from core.const_var import COLORDICT

COLOR_MODE_RGB = 0
COLOR_MODE_HSV = 1
COLOR_MODE_Lab = 2
COLOR_MODE_YCrCb = 3


class ColorModeNotAllowException(Exception):
    def __init__(self, expression=None, message=None):
        self.expression = expression
        self.message = message


class ScatterException(Exception):
    def __init__(self, expression=None, message=None):
        self.expression = expression
        self.message = message


class HistogramException(Exception):
    def __init__(self, expression=None, message=None):
        self.expression = expression
        self.message = message


with open(SKIN_PARAM_PATH, 'r') as infile:
    j = json.load(infile)
paramDict = j

class SkinUtils:
    @staticmethod
    def trimSkin(img):
        """
        使用RGB_HSV_YCrCb的融合方法去掉非皮肤
        :param img:
        :return:
        """
        skin, mask1, mask2 = SkinTrimUtils.trimByParam(img, paramDict)
        return skin
        # return SkinTrimUtils.YCrCb(img)

    @staticmethod
    def getFourColorSampleByROIName(roiName, sampleDict):
        """
        根据ROI名称返回四个颜色的样本
        :param roiName:
        :param sampleDict:
        :return:
        """
        d = sampleDict

        red = d[ImgUtils.KEY_SAMPLE_RED]
        yellow = d[ImgUtils.KEY_SAMPLE_YELLOW]
        black = d[ImgUtils.KEY_SAMPLE_BLACK]
        white = d[ImgUtils.KEY_SAMPLE_WHITE]

        redSample = red[roiName]
        yellowSample = yellow[roiName]
        blackSample = black[roiName]
        whiteSample = white[roiName]

        return redSample, yellowSample, blackSample, whiteSample

    @staticmethod
    def getFourColorSampleByROINameAndColorSpace(roiName, sampleDict, colorSpace):
        """
        获取对应空间的ROI
        :param roiName:
        :param sampleDict:
        :param colorSpace:
        :return:
        """
        redSample, yellowSample, blackSample, whiteSample = SkinUtils.getFourColorSampleByROIName(roiName, sampleDict)

        if colorSpace == COLOR_MODE_YCrCb:
            mode = cv2.COLOR_BGR2YCrCb
        elif colorSpace == COLOR_MODE_HSV:
            mode = cv2.COLOR_BGR2HSV
        elif colorSpace == COLOR_MODE_Lab:
            mode = cv2.COLOR_BGR2Lab
        elif colorSpace == COLOR_MODE_RGB:
            mode = cv2.COLOR_BGR2RGB
        else:
            mode = None

        if mode is not None:
            return cv2.cvtColor(redSample, mode), \
                   cv2.cvtColor(yellowSample, mode), \
                   cv2.cvtColor(blackSample, mode), \
                   cv2.cvtColor(whiteSample, mode)
        else:
            return redSample, yellowSample, blackSample, whiteSample

    @staticmethod
    def showScatter(img, channelA, channelB, labelX, labelY, roiName, sampleDict, colorSpace, xLim=None, yLim=None):
        # draw predict
        plt.scatter(img[:, :, channelA].flatten(), img[:, :, channelB].flatten(), alpha=0.5, c='green', label=roiName)
        # draw sample
        redSample, yellowSample, blackSample, whiteSample = SkinUtils.getFourColorSampleByROINameAndColorSpace(roiName,
                                                                                                               sampleDict,
                                                                                                               colorSpace)

        def drawSample(sample, color=None, label=None):
            plt.scatter(sample[:, :, channelA].flatten(), sample[:, :, channelB].flatten(), alpha=0.5, c=color,
                        label=label)

        drawSample(redSample, "red", "red")
        drawSample(yellowSample, "yellow", "yellow")
        drawSample(whiteSample, "lightBlue", "white")
        drawSample(blackSample, "purple", "black")

        if xLim is not None:
            plt.xlim(xLim)

        if yLim is not None:
            plt.ylim(yLim)

        plt.xlabel(labelX)
        plt.ylabel(labelY)

    @staticmethod
    def show_histogram(img, title, channelA, channelB, luminanceChannel, color=['b', 'g', 'r'], label=['a', 'b', 'c'],
                       roiName=None,
                       sampleDict=None, colorspace=COLOR_MODE_RGB):
        """
        画出直方图和散点图
        :param img:  要画图的原数组
        :param title:  标题
        :param channelA: 要展示的维度1
        :param channelB: 要展示的维度2
        :param luminanceChannel: 亮度通道，用来排除黑点
        :param color: 三个通道颜色怎么显示
        :param label: 三个通道在图中显示的标签。
        :param roiName: ROI的名字
        :param sampleDict: 样本
        :param colorspace: 画什么颜色空间的直方图和散点图
        :return:
        """

        def sp(img):
            """
            分割出亮度通道，并提取出亮度>0的数值
            :param img:
            :return:
            """
            c = img[:, :, luminanceChannel]
            a = img[:, :, channelA][c > 0]
            b = img[:, :, channelB][c > 0]
            return a, b, c

        a, b, c = sp(img)
        ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
        abins = abs(int(a.max() - a.min()))
        bbins = abs(int(b.max() - b.min()))
        cbins = abs(int(c.max() - c.min()))
        if abins < 1: abins = 1
        if bbins < 1: bbins = 1
        if cbins < 1: cbins = 1
        LogUtils.log("SkinUtils-show_histogram",
                     ",abins=" + str(abins) + ",bbins=" + str(bbins) + ",cbin=" + str(cbins))

        arange = (a.min(), a.max())
        brange = (b.min(), b.max())
        crange = (c.min(), c.max())

        ax.hist(a.ravel(), bins=abins, range=arange, alpha=.7, label=label[0], color=color[0])
        ax.hist(b.ravel(), bins=bbins, range=brange, alpha=.7, label=label[1], color=color[1])
        ax.hist(c.ravel(), bins=cbins, range=crange, alpha=.7, label=label[2], color=color[2])
        ax.set_title(title)
        ax.legend(loc="best")

        ax1 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
        ax1.hist(a.ravel(), bins=abins, range=arange, alpha=.7, label=label[0], color=color[0])
        ax1.set_xlabel(label[0])

        ax2 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
        ax2.hist(b.ravel(), bins=bbins, range=brange, alpha=.7, label=label[1], color=color[1])
        ax2.set_xlabel(label[1])

        ax3 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
        ax3.hist(c.ravel(), bins=cbins, range=crange, alpha=.7, label=label[2], color=color[2])
        ax3.set_xlabel(label[2])

        def scatterSample(ax):
            a, b, c = sp(redSample)
            ax.scatter(a, b, alpha=0.5, c='red', label="red")
            a, b, c = sp(blackSample)
            ax.scatter(a, b, alpha=0.5, c='purple', label="black")
            a, b, c = sp(yellowSample)
            ax.scatter(a, b, alpha=0.5, c='yellow', label="yellow")
            a, b, c = sp(whiteSample)
            ax.scatter(a, b, alpha=0.5, c='lightblue', label="white")
            a, b, c = sp(img)
            ax.scatter(a, b, alpha=0.3, c='green', label=roiName, marker='^')
            ax.legend(loc="best")

        redSample, yellowSample, blackSample, whiteSample = \
            SkinUtils.getFourColorSampleByROINameAndColorSpace(roiName, sampleDict, colorspace)

        ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        scatterSample(ax4)
        ax4.set_xlabel(label[channelA])
        ax4.set_ylabel(label[channelB])
        plt.show()

    @staticmethod
    def skinHistogram(fig, img=None, colormode=COLOR_MODE_RGB, roiName=None, sampleDict=None):
        """
        :param img: 接受一个BGR模式的图片
        :param colormode:  选择绘制什么直方图
        :return:
        """
        # fig = plt.figure(figsize=(12, 8))  # 画布大小
        # img = SkinUtils.trimSkin(img)
        img =imutils.resize(img, width=40)
        if colormode == COLOR_MODE_RGB:
            SkinUtils.show_histogram(img, "RGB", 0, 1, 2, ('b', 'g', 'r'), ['b', 'g', 'r'], roiName, sampleDict,
                                     colormode)
        elif colormode == COLOR_MODE_HSV:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            SkinUtils.show_histogram(img, "HSV", 0, 1, 2, ('b', 'g', 'r'), ['H', 'S', 'V'], roiName, sampleDict,
                                     colormode)
        elif colormode == COLOR_MODE_Lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            SkinUtils.show_histogram(img, "Lab", 1, 2, 0, ('b', 'g', 'r'), ['L', 'a', 'b'], roiName, sampleDict,
                                     colormode)
        elif colormode == COLOR_MODE_YCrCb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            SkinUtils.show_histogram(img, "YCrCb", 1, 2, 0, ('b', 'g', 'r'), ['Y', 'Cr', 'Cb'], roiName, sampleDict,
                                     colormode)
        else:
            raise HistogramException("SkinUtils, 没有指定颜色空间！")
        fig.tight_layout()
        # plt.legend(loc="best")
        return ImgUtils.getcvImgFromFigure(fig)

    @staticmethod
    def skinScatter(fig, img=None, colormode=COLOR_MODE_HSV, roiName=None, sampleDict=None):
        """
        绘制散点图
        :param img: 接受一个BGR模式的图片
        :param colormode:  选择绘制什么散点图
        :return:
        """
        if roiName is None:
            raise ScatterException("roi名字不能为空！")
        # fig = plt.figure(figsize=(12, 8))  # 画布大小
        img = SkinUtils.trimSkin(img)
        if colormode == COLOR_MODE_RGB:
            raise ColorModeNotAllowException("不支持RGB二维散点图")
        elif colormode == COLOR_MODE_HSV:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            SkinUtils.showScatter(img, 0, 1, "H", "S", roiName, sampleDict, colormode)
        elif colormode == COLOR_MODE_Lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            SkinUtils.showScatter(img, 1, 2, "a", "b", roiName, sampleDict, colormode)
        elif colormode == COLOR_MODE_YCrCb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            SkinUtils.showScatter(img, 1, 2, "Cr", "Cb", roiName, sampleDict, colormode)
        else:
            pass
        return ImgUtils.getcvImgFromFigure(fig)

    @staticmethod
    def skinColorDetect(img):
        return None

    @staticmethod
    def roiTotalColorDetect(rois):
        for roi in rois:
            pass

    @staticmethod
    def getDistanceByRGB(predict, sample):
        def trimBlack(img):
            k = 0
            B, G, R = 0, 0, 0
            for row in img:
                for v in row:
                    # 排除黑色
                    # if v[0] != 0:
                    if not (v[0] == 0 and v[1] == 0 and v[2] == 0):
                        k = k + 1
                        R += v[0]
                        G += v[1]
                        R += v[2]
            if k == 0:
                return 0, 0, 0
            # 计算出了LAB的均值
            R0 = int(round(R / k))
            G0 = int(round(G / k))
            B0 = int(round(R / k))
            return R0, G0, B0

        pr, pg, pb = trimBlack(predict)
        sr, sg, sb = trimBlack(sample)

        # distance = ((pa - sa) ** 2 + (pb - sb) ** 2) ** 0.5
        distance = ((pr - sr) ** 2 + (pg - sg) ** 2 + (pb - sb) ** 2) ** 0.5
        return distance

    @staticmethod
    def getDistanceByLab(predict, sample):
        """
        :param predict:
        :param sample:
        :return:
        """

        def trimBlack(img):
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            k = 0
            L, A, B = 0, 0, 0
            for row in img_lab:
                for v in row:
                    # 排除黑色
                    if v[0] != 0:
                        k = k + 1
                        L += v[0]
                        A += v[1]
                        B += v[2]
            if k == 0:
                return 0, 0, 0
            # 计算出了LAB的均值
            L0 = int(round(L / k))
            A0 = int(round(A / k))
            B0 = int(round(B / k))
            return L0, A0, B0

        pl, pa, pb = trimBlack(predict)
        sl, sa, sb = trimBlack(sample)

        distance = ((pa - sa) ** 2 + (pb - sb) ** 2) ** 0.5
        # distance = ((pl - sl) ** 2 + (pa - sa) ** 2 + (pb - sb) ** 2)
        # distance = ((pl - sl) ** 2 + (pa - sa) ** 2 + (pb - sb) ** 2)
        return distance

    @staticmethod
    def getDistanceByHSV(predict, sample):
        """
        :param predict:
        :param sample:
        :return:
        """

        def trimBlack(img):
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            k = 0
            H, S, V = 0, 0, 0
            for row in img_hsv:
                for v in row:
                    # 排除黑色
                    if v[2] != 0:
                        k = k + 1
                        H += v[0]
                        S += v[1]
                        V += v[2]

            if k == 0:
                return 0, 0, 0
            # 计算出了LAB的均值
            H0 = int(round(H / k))
            S0 = int(round(S / k))
            V0 = int(round(V / k))
            return H0, S0, V0

        ph, ps, pv = trimBlack(predict)
        sh, ss, sv = trimBlack(sample)
        # distance = ((ph - sh) ** 2 + (ps - ss) ** 2 + (pv - sv) ** 2)
        distance = ((ph - sh) ** 2 + (ps - ss) ** 2) ** 0.5
        return distance

    @staticmethod
    def getDistanceYCrCb(predict, sample):
        """
        :param predict:
        :param sample:
        :return:
        """

        def trimBlack(img):
            img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            k = 0
            Y, Cr, Cb = 0, 0, 0
            for row in img_YCrCb:
                for v in row:
                    # 排除黑色
                    if v[0] != 0:
                        k = k + 1
                        Y += v[0]
                        Cr += v[1]
                        Cb += v[2]
            if k == 0:
                return 0, 0, 0
            # 计算出了Y, Cr, Cb的均值
            Y0 = round(Y / k)
            Cr0 = round(Cr / k)
            Cb0 = round(Cb / k)
            a, b, c = cv2.split(img_YCrCb)
            # print("calc:Y", Y0, ", Cr", Cr0, ",Cb", Cb0)
            # print("mean:Y", round((a[a > 0]).mean()), ", Cr", round((b[a > 0]).mean()), ",Cb", round((c[a > 0]).mean()))
            return Y0, Cr0, Cb0

        pY, pCr, pCb = trimBlack(predict)
        sY, sCr, sCb = trimBlack(sample)
        distance = ((pCr - sCr) ** 2 + (pCb - sCb) ** 2) ** 0.5
        return distance

    @staticmethod
    def getResultByOneColor(roiKey, color):
        # // TODO 更多的咨询
        return FACIAL_LANDMARKS_NAME_DICT[roiKey] + "颜色偏向于:" + ImgUtils.COLOR_SAMPLE_CN_NAME_BY_KEY[color]

    @staticmethod
    def trimSkinRealTime(img, skinParamDict=None):
        if img is None:
            return
        if skinParamDict is not None:
            skin, mask1, mask2 = SkinTrimUtils.trimByParam(img, skinParamDict)
        else:
            skin, mask1, mask2 = SkinTrimUtils.YCrCb(img)
        return skin


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


#
# def _color_space_show():
#     # fig1 = Figure()
#     # canvas = FigureCanvas(fig1)
#     # fig1 = Figure()
#     # fig1 = Figure()
#     # fig1 = Figure()
#     fig1 = plt.figure()
#     plt.yticks([i for i in range(256)])
#     plt.title("hello")
#     while videoCapture.isOpened():
#         flag, img = videoCapture.read()
#         if not flag:
#             break
#         fig1.clear()
#         hist_rgb = SkinUtils.skinHistogram(fig1, img)
#         cv2.imshow('hist_rgb', hist_rgb)
#
#         fig1.clear()
#         hist_lab = SkinUtils.skinHistogram(fig1, img, COLOR_MODE_Lab)
#         cv2.imshow('hist_lab', hist_lab)
#
#         fig1.clear()
#         hist_hsv = SkinUtils.skinHistogram(fig1, img, COLOR_MODE_HSV)
#         cv2.imshow('hist_hsv', hist_hsv)
#
#         fig1.clear()
#         hist_ycrcb = SkinUtils.skinHistogram(fig1, img, COLOR_MODE_YCrCb)
#         cv2.imshow('hist_ycrcb', hist_ycrcb)
#
#         k = cv2.waitKey(33) & 0xFF
#         if k == 27:
#             break
#
#     videoCapture.release()
#     cv2.destroyAllWindows()
#
#
# def _distance_test():
#     pass
#

# videoCapture = cv2.VideoCapture(1)

def _testHist():
    # img = cv2.imread("../result/predict_white/ke.jpg")
    img = cv2.imread("../result/chi/ke.jpg")
    sampleDict = ImgUtils.getSampleDict()
    roiName = KEY_ke
    fig = plt.figure()
    result0 = SkinUtils.skinHistogram(fig, img, COLOR_MODE_RGB, roiName, sampleDict)

    fig = plt.figure()
    result1 = SkinUtils.skinHistogram(fig, img, COLOR_MODE_HSV, roiName, sampleDict)

    fig = plt.figure()
    result2 = SkinUtils.skinHistogram(fig, img, COLOR_MODE_Lab, roiName, sampleDict)

    fig = plt.figure()
    result3 = SkinUtils.skinHistogram(fig, img, COLOR_MODE_YCrCb, roiName, sampleDict)

    cv2.imshow("RGB", result0)
    cv2.imshow("HSV", result1)
    cv2.imshow("Lab", result2)
    cv2.imshow("YCrCb", result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def _testScatter():
    # img = cv2.imread("../result/predict_white/ke.jpg")
    img = cv2.imread("../result/chi/ke.jpg")
    sampleDict = ImgUtils.getSampleDict()
    fig = plt.figure()
    # result1 = SkinUtils.skinScatter(fig, img, COLOR_MODE_HSV, KEY_ting, sampleDict)

    result1 = SkinUtils.skinScatter(fig, img, COLOR_MODE_HSV, KEY_ting, sampleDict)

    fig = plt.figure()
    result2 = SkinUtils.skinScatter(fig, img, COLOR_MODE_Lab, KEY_ting, sampleDict)

    fig = plt.figure()
    result3 = SkinUtils.skinScatter(fig, img, COLOR_MODE_YCrCb, KEY_ting, sampleDict)
    cv2.imshow("HSV", result1)
    cv2.imshow("Lab", result2)
    cv2.imshow("YCrCb", result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # _color_space_show()
    _testHist()
