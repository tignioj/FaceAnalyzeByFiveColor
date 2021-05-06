import threading
import time
from utils.ImageUtils import getcvImgFromFigure

import cv2

# 官方地址 https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
from utils.ColorSpaceTransform import *
from core.const_var import COLORDICT

COLOR_MODE_RGB = 0
COLOR_MODE_HSV = 1
COLOR_MODE_Lab = 2
COLOR_MODE_YCrCb = 3


class SkinUtils:

    @staticmethod
    def trimSkin(img):
        """
        使用RGB_HSV_YCrCb的融合方法去掉非皮肤
        :param img:
        :return:
        """
        return rgb_hsv_ycbcr(img)

    @staticmethod
    def show_histogram(hist, title, color, label=['a', 'b', 'c']):
        global x1, y1, color_glob
        plt.title(title)
        # 定位图片
        # plt.subplot(3, 2, pos)
        for h, c, l in zip(hist, color, label):  # color: ('b', 'g', 'r'), zip:连接
            plt.plot(h, color=c, label=l)

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
    def skinColorDetect(img):
        return None

    @staticmethod
    def roiTotalColorDetect(rois):
        for roi in rois:
            pass

    # img_predict_roi_ting = cv2.resize(img_predict_roi_ting, img_sample_roi_ting.shape[::-1][1:3])
    @staticmethod
    def getDistanceByRGB(predict, sample):
        """
        去掉黑色像素，这是被肤色检测处理过的像素
        :param predict:
        :param sample:
        :return:
        """
        x = predict.shape[0]
        y = predict.shape[1]
        dist_byloop = []
        for i in range(x):
            for j in range(y):
                # 纯黑色不检测，因为这是被肤色检测处理过的像素
                if (predict[i][j] == (0, 0, 0)).all() or (sample[i][j] == (0, 0, 0)).all():
                    continue

                A = predict[i][j]
                B = sample[i][j]
                # np.linalg.norm(A - B) 等同于
                # np.sqrt(np.sum((A[0] - B[0])**2 + (A[1] - B[1])**2 +(A[2] - B[2])**2))
                dist_byloop.append(np.linalg.norm(A - B))
                # sum += np.sqrt(np.sum(np.square(predict[i][j] - sample[i][j])))
        return np.mean(dist_byloop)

    @staticmethod
    def getDistance2ByLab(predict, sample):
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

        # distance = ((pa - sa) ** 2 + (pb - sb) ** 2) ** 0.5
        distance = ((pl - sl) ** 2 + (pa - sa) ** 2 + (pb - sb) ** 2)
        return distance

    @staticmethod
    def getDistance2BHSV(predict, sample):
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
        distance = ((ph - sh) ** 2 + (ps - ss) ** 2 + (pv - sv) ** 2)
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
        return SkinUtils.getDistanceByRGB(predict, sample), SkinUtils.getDistance2ByLab(predict,
                                                                                        sample), SkinUtils.getDistance2BHSV(
            predict, sample)

    @staticmethod
    def getResultByColor(rois):
        pass


def _thread_function():
    fig1 = plt.figure()
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


videoCapture = cv2.VideoCapture(1)
if __name__ == '__main__':
    _thread_function()
