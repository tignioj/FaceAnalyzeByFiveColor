import threading
import time

from utils import ImageUtils
from utils.ImageUtils import getcvImgFromFigure, keepSameShape, COLOR_SAMPLE_CN_NAME_BY_KEY, KEY_SAMPLE_YELLOW, \
    KEY_SAMPLE_BLACK, KEY_SAMPLE_WHITE, KEY_SAMPLE_RED
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
    def getFourColorSampleByROIName(roiName):
        d = ImageUtils.getSampleDict()

        red = d[ImageUtils.KEY_SAMPLE_RED]
        yellow = d[ImageUtils.KEY_SAMPLE_YELLOW]
        black = d[ImageUtils.KEY_SAMPLE_BLACK]
        white = d[ImageUtils.KEY_SAMPLE_WHITE]

        redSample = red[roiName]
        yellowSample = yellow[roiName]
        blackSample = black[roiName]
        whiteSample = white[roiName]

        return redSample, yellowSample, blackSample, whiteSample

    @staticmethod
    def showScatter(a, b, labelX, labelY, roiName):
        # draw predict
        plt.scatter(a, b, alpha=0.5, c='green', label=roiName)
        # draw sample
        redSample, yellowSample, blackSample, whiteSample = SkinUtils.getFourColorSampleByROIName(roiName)

        def drawSample(sample, color=None, label=None):
            # sample = cv2.resize(sample, (50,50))
            plt.scatter(sample[:, :, 0].flatten(), sample[:, :, 1].flatten(), alpha=0.5, c=color, label=label)

        drawSample(redSample, "red", "red")
        drawSample(yellowSample, "yellow", "yellow")
        drawSample(whiteSample, "lightBlue", "white")
        drawSample(blackSample, "purple", "black")

        plt.xlabel(labelX)
        plt.ylabel(labelY)

    @staticmethod
    def show_histogram(hist, title, color, label=['a', 'b', 'c']):
        # plt.title(title)
        # 定位图片
        # plt.subplot(3, 2, pos)
        for h, c, l in zip(hist, color, label):  # color: ('b', 'g', 'r'), zip:连接
            plt.plot(h, color=c, label=l)
            # plt.bar([i for i in range(0, 256)], h, color=c, label=l)

    @staticmethod
    def _cal_color_his(image, range=[0, 256]):
        hist = []
        hist.append(cv2.calcHist([image], [0], None, [256], range))
        hist.append(cv2.calcHist([image], [1], None, [256], range))
        hist.append(cv2.calcHist([image], [2], None, [256], range))
        return hist

    @staticmethod
    def skinHistogram(fig, img=None, colormode=COLOR_MODE_RGB):
        """
        :param img: 接受一个BGR模式的图片
        :param colormode:  选择绘制什么直方图
        :return:
        """
        # fig = plt.figure(figsize=(12, 8))  # 画布大小
        img = SkinUtils.trimSkin(img)
        if colormode == COLOR_MODE_RGB:
            SkinUtils.show_histogram(SkinUtils._cal_color_his(img), "RGB", ('b', 'g', 'r'), ['r', 'g', 'b'])
        elif colormode == COLOR_MODE_HSV:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            SkinUtils.show_histogram(SkinUtils._cal_color_his(img), "HSV", ('b', 'g', 'r'), ['H', 'S', 'V'])
        elif colormode == COLOR_MODE_Lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            SkinUtils.show_histogram(SkinUtils._cal_color_his(img), "Lab", ('b', 'g', 'r'), ['L', 'a', 'b'])

        elif colormode == COLOR_MODE_YCrCb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            SkinUtils.show_histogram(SkinUtils._cal_color_his(img), "YCrCb", ('b', 'g', 'r'), ['Y', 'Cr', 'Cb'])
        else:
            pass
        plt.legend()
        return getcvImgFromFigure(fig)





    @staticmethod
    def skinScatter(fig, img=None, colormode=COLOR_MODE_HSV, roiName=None):
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
            h, s, v = cv2.split(img)
            SkinUtils.showScatter(h.flatten(), s.flatten(), "H", "S", roiName)
        elif colormode == COLOR_MODE_Lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            L, a, b = cv2.split(img)
            SkinUtils.showScatter(a.flatten(), b.flatten(), "a", "b", roiName)
        elif colormode == COLOR_MODE_YCrCb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            Y, Cr, Cb = cv2.split(img)
            SkinUtils.showScatter(Cr.flatten(), Cb.flatten(), "Cr", "Cb", roiName)
        else:
            pass
        plt.legend()
        return getcvImgFromFigure(fig)

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
        return FACIAL_LANDMARKS_NAME_DICT[roiKey] + "颜色偏向于:" + COLOR_SAMPLE_CN_NAME_BY_KEY[color]

    @staticmethod
    def trimSkinRealTime(img, scale):
        return


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def _color_space_show():
    # fig1 = Figure()
    # canvas = FigureCanvas(fig1)
    # fig1 = Figure()
    # fig1 = Figure()
    # fig1 = Figure()
    fig1 = plt.figure()
    plt.yticks([i for i in range(256)])
    plt.title("hello")
    while videoCapture.isOpened():
        flag, img = videoCapture.read()
        if not flag:
            break
        fig1.clear()
        hist_rgb = SkinUtils.skinHistogram(fig1, img)
        cv2.imshow('hist_rgb', hist_rgb)

        fig1.clear()
        hist_lab = SkinUtils.skinHistogram(fig1, img, COLOR_MODE_Lab)
        cv2.imshow('hist_lab', hist_lab)

        fig1.clear()
        hist_hsv = SkinUtils.skinHistogram(fig1, img, COLOR_MODE_HSV)
        cv2.imshow('hist_hsv', hist_hsv)

        fig1.clear()
        hist_ycrcb = SkinUtils.skinHistogram(fig1, img, COLOR_MODE_YCrCb)
        cv2.imshow('hist_ycrcb', hist_ycrcb)

        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break

    videoCapture.release()
    cv2.destroyAllWindows()


def _distance_test():
    pass


videoCapture = cv2.VideoCapture(1)
if __name__ == '__main__':
    _color_space_show()
