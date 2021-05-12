import threading
import time

from utils.ImageUtils import ImgUtils
from core.const_var import *
from core.const_var import FACIAL_LANDMARKS_NAME_DICT

import cv2

# 官方地址 https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# 颜色空间的有关问题：http://poynton.ca/notes/colour_and_gamma/ColorFAQ.html
from utils.SkinTrimUtlis import *
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


class SkinUtils:

    @staticmethod
    def trimSkin(img):
        """
        使用RGB_HSV_YCrCb的融合方法去掉非皮肤
        :param img:
        :return:
        """
        return SkinTrimUtils.rgb_hsv_ycbcr(img)
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
    def show_histogram(img, title, channelA, channelB, color=['b', 'g', 'r'], label=['a', 'b', 'c'], roiName=None,
                       sampleDict=None, colorspace=COLOR_MODE_RGB):
        a, b, c = cv2.split(img)

        ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
        abins = a.max() - a.min()
        bbins = b.max() - b.min()
        cbins = c.max() - c.min()

        arange = (a.min(), a.max())
        brange = (b.min(), b.max())
        crange = (c.min(), c.max())

        ax.hist(a.ravel(), bins=abins, range=arange, alpha=.7, label=label[0], color=color[0])
        ax.hist(b.ravel(), bins=bbins, range=brange, alpha=.7, label=label[1], color=color[1])
        ax.hist(c.ravel(), bins=cbins, range=crange, alpha=.7, label=label[2], color=color[2])
        ax.set_title(title)
        ax.legend()

        ax1 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
        ax1.hist(a.ravel(), bins=abins, range=arange, alpha=.7, label=label[0], color=color[0])
        ax1.set_xlabel(label[0])

        ax2 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
        ax2.hist(b.ravel(), bins=bbins, range=brange, alpha=.7, label=label[1], color=color[1])
        ax2.set_xlabel(label[1])

        ax3 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
        ax3.hist(c.ravel(), bins=cbins, range=crange, alpha=.7, label=label[2], color=color[2])
        ax3.set_xlabel(label[2])

        def sp(img):
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            return img[:, :, channelA], img[:, :, channelB]

        def scatterSample(ax):
            a, b = sp(redSample)
            ax.scatter(a, b, alpha=0.5, c='red', label="red")
            a, b = sp(blackSample)
            ax.scatter(a, b, alpha=0.5, c='purple', label="black")
            a, b = sp(yellowSample)
            ax.scatter(a, b, alpha=0.5, c='yellow', label="yellow")
            a, b = sp(whiteSample)
            ax.scatter(a, b, alpha=0.5, c='lightblue', label="white")
            a, b = sp(img)
            ax.scatter(a, b, alpha=0.5, c='green', label=roiName, marker='^')
            ax.legend()

        def scatterSample3D(ax):
            a, b, c = cv2.split(redSample)
            ax.scatter(a, b, c, alpha=0.5, c='red', label="red")
            a, b, c = cv2.split(blackSample)
            ax.scatter(a, b, c, alpha=0.5, c='purple', label="black")
            a, b, c = cv2.split(yellowSample)
            ax.scatter(a, b, c, alpha=0.5, c='yellow', label="yellow")
            a, b, c = cv2.split(whiteSample)
            ax.scatter(a, b, c, alpha=0.5, c='lightblue', label="white")
            a, b, c = cv2.split(img)
            ax.scatter(a, b, c, alpha=0.5, c='green', label=roiName, marker='^')
            ax.legend()

        redSample, yellowSample, blackSample, whiteSample = \
            SkinUtils.getFourColorSampleByROINameAndColorSpace(roiName, sampleDict, colorspace)

        if colorspace == COLOR_MODE_RGB:
            ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2, projection='3d')
            scatterSample3D(ax4)
            ax4.set_xlabel("B")
            ax4.set_ylabel("G")
            ax4.set_zlabel("R")
        else:
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
        img = SkinUtils.trimSkin(img)
        if colormode == COLOR_MODE_RGB:
            SkinUtils.show_histogram(img, "RGB", 0, 1, ('b', 'g', 'r'), ['b', 'g', 'r'], roiName, sampleDict, colormode)
        elif colormode == COLOR_MODE_HSV:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            SkinUtils.show_histogram(img, "HSV", 0, 1, ('b', 'g', 'r'), ['H', 'S', 'V'], roiName, sampleDict, colormode)
        elif colormode == COLOR_MODE_Lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            SkinUtils.show_histogram(img, "Lab", 1, 2, ('b', 'g', 'r'), ['L', 'a', 'b'], roiName, sampleDict, colormode)
        elif colormode == COLOR_MODE_YCrCb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            SkinUtils.show_histogram(img, "YCrCb", 1, 2, ('b', 'g', 'r'), ['Y', 'Cr', 'Cb'], roiName, sampleDict,
                                     colormode)
        else:
            raise HistogramException("SkinUtils, 没有指定颜色空间！")
        # plt.legend()
        # fig.tight_layout()
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
        # plt.legend()
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
                    if v[0] != 0 and v[1] != 0 and v[2] != 0:
                        k = k + 1
                        R += v[0]
                        G += v[1]
                        R += v[2]
            # 计算出了LAB的均值
            R0 = int(round(R / k))
            G0 = int(round(G / k))
            B0 = int(round(R / k))
            return R0, G0, B0

        # predict_lab = cv2.cvtColor(predict, cv2.COLOR_BGR)
        # sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2Lab)
        pr, pg, pb = trimBlack(predict)
        sr, sg, sb = trimBlack(sample)

        # distance = ((pa - sa) ** 2 + (pb - sb) ** 2) ** 0.5
        distance = ((pr - sr) ** 2 + (pg - sg) ** 2 + (pb - sb) ** 2)
        return distance

    @staticmethod
    def getDistanceByLab(predict, sample):
        """
        :param predict:
        :param sample:
        :return:
        """

        def trimBlack(img_lab):
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
            # 计算出了LAB的均值
            L0 = int(round(L / k))
            A0 = int(round(A / k))
            B0 = int(round(B / k))
            return L0, A0, B0

        predict_lab = cv2.cvtColor(predict, cv2.COLOR_BGR2Lab)
        sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2Lab)
        pl, pa, pb = trimBlack(predict_lab)
        sl, sa, sb = trimBlack(sample_lab)

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

        def trimBlack(img_hsv):
            k = 0
            H, S, V = 0, 0, 0
            for row in img_hsv:
                for v in row:
                    # 排除黑色
                    if v[0] != 0:
                        k = k + 1
                    H += v[0]
                    S += v[1]
                    V += v[2]
            # 计算出了LAB的均值
            H0 = int(round(H / k))
            S0 = int(round(S / k))
            V0 = int(round(V / k))
            return H0, S0, V0

        predict_hsv = cv2.cvtColor(predict, cv2.COLOR_BGR2HSV)
        sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        ph, ps, pv = trimBlack(predict_hsv)
        sh, ss, sv = trimBlack(sample_hsv)
        # distance = ((ph - sh) ** 2 + (ps - ss) ** 2 + (pv - sv) ** 2)
        distance = (ph - sh) ** 2
        return distance

    @staticmethod
    def getDistanceYCrCb(predict, sample):
        """
        :param predict:
        :param sample:
        :return:
        """

        def trimBlack(img_hsv):
            k = 0
            Y, Cr, Cb = 0, 0, 0
            for row in img_hsv:
                for v in row:
                    # 排除黑色
                    if v[0] != 0:
                        k = k + 1
                    Y += v[0]
                    Cr += v[1]
                    Cb += v[2]
            # 计算出了LAB的均值
            H0 = int(round(Y / k))
            S0 = int(round(Cr / k))
            V0 = int(round(Cb / k))
            return H0, S0, V0

        predict_YCrCb = cv2.cvtColor(predict, cv2.COLOR_BGR2YCrCb)
        sample_YCrCb = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        pY, pCr, pCb = trimBlack(predict_YCrCb)
        sY, sCr, sCb = trimBlack(sample_YCrCb)
        distance = ((pY - sY) ** 2 + (pCr - sCr) ** 2 + (pCb - sCb) ** 2)
        return distance

    @staticmethod
    def distance(predict, sample=None):
        # predict = cv2.imread("../../result/predict1/ting_trim.jpg")
        # predict = cv2.imread("../../result/predict2/ting_trim.jpg")
        # predict = cv2.imread("../../result/predict3_white/ting_trim.jpg")
        # predict = cv2.imread("../../result/chi/ming_tang_trim.jpg")
        # predict = cv2.imread("../../result/black/ming_tang_trim.jpg")
        # predict = cv2.imread("../../result/black/ting_trim.jpg")
        # predict = cv2.imread("../../result/predict4_dark/ting_trim.jpg")
        predict = cv2.resize(predict, (sample.shape[1], sample.shape[0]))
        # res = np.hstack([predict, sample])
        # cv2.imshow(name, res)
        return SkinUtils.getDistanceByRGB(predict, sample), SkinUtils.getDistanceByLab(predict,
                                                                                       sample), SkinUtils.getDistanceByHSV(
            predict, sample)

    @staticmethod
    def getResulstByColor(rois, colors):
        pass

    @staticmethod
    def getResultByOneColor(roiKey, color):
        # // TODO 更多的咨询
        return FACIAL_LANDMARKS_NAME_DICT[roiKey] + "颜色偏向于:" + ImgUtils.COLOR_SAMPLE_CN_NAME_BY_KEY[color]

    @staticmethod
    def trimSkinRealTime(img, scale):
        return


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
    img = cv2.imread("../result/predict_white/ke.jpg")
    sampleDict = ImgUtils.getSampleDict()
    roiName = KEY_ting
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
    img = cv2.imread("../result/predict_white/ke.jpg")
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
