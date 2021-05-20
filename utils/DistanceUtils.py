import numpy as np
import cv2

from utils.SkinUtils import SkinUtils
from utils.ImageUtils import ImgUtils


class DistanceUtils:
    @staticmethod
    def getDistanceByRGB(predict, sample):
        def trimBlack(img):
            a, b, c = cv2.split(img)
            return (a[a > 0]).mean(), (b[b > 0]).mean(), (c[c > 0]).mean()

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
            l, a, b = cv2.split(img_lab)
            return (l[l > 0]).mean(), (a[l > 0]).mean(), (b[l > 0]).mean()

        pl, pa, pb = trimBlack(predict)
        sl, sa, sb = trimBlack(sample)

        # distance = ((pa - sa) ** 2 + (pb - sb) ** 2) ** 0.5
        distance = ((pl - sl) ** 2 + (pa - sa) ** 2 + (pb - sb) ** 2) ** 0.5
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
            h, s, v = cv2.split(img_hsv)
            return (h[v > 1]).mean(), (s[v > 1]).mean(), (v[v > 1]).mean()

        ph, ps, pv = trimBlack(predict)
        sh, ss, sv = trimBlack(sample)
        distance = ((ph - sh) ** 2 + (ps - ss) ** 2 + (pv - sv) ** 2) ** 0.5
        # distance = ((ph - sh) ** 2 + (ps - ss) ** 2) ** 0.5
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
            Y, Cr, Cb = cv2.split(img_YCrCb)
            return (Y[Y > 0]).mean(), (Cr[Y > 0]).mean(), (Cb[Y > 0]).mean()

        pY, pCr, pCb = trimBlack(predict)
        sY, sCr, sCb = trimBlack(sample)
        # distance = ((pCr - sCr) ** 2 + (pCb - sCb) ** 2) ** 0.5
        distance = ((pY - sY) ** 2 + (pCr - sCr) ** 2 + (pCb - sCb) ** 2) ** 0.5
        return distance

    @staticmethod
    def getDistArray(predict=None, sample_red=None, sample_yellow=None, sample_black=None, sample_white=None):
        """
        获取单个ROI的四种算法距离
        获取五种算法后的距离数组，以及预测的颜色，返回值数据结构
        https://en.wikipedia.org/wiki/Color_difference

        {
            'lab': [ distance array, color result]
            'ycrcb': [ distance array, color result]
            'hsv': [ distance array, color result]
            'rgb': [ distance array, color result]
            "order": ['红', '黄', ‘白', '黑']
        }
        :param predict:
        :return:
        """

        # melt_dist_red = DistanceUtils.getDistanceByRGB(predict, sample_red)
        # melt_dist_yellow = DistanceUtils.getDistanceByRGB(predict, sample_yellow)
        # melt_dist_black = DistanceUtils.getDistanceByRGB(predict, sample_black)
        # melt_dist_white = DistanceUtils.getDistanceByRGB(predict, sample_white)

        lab_dist_red = DistanceUtils.getDistanceByLab(predict, sample_red)
        lab_dist_yellow = DistanceUtils.getDistanceByLab(predict, sample_yellow)
        lab_dist_black = DistanceUtils.getDistanceByLab(predict, sample_black)
        lab_dist_white = DistanceUtils.getDistanceByLab(predict, sample_white)

        ycrcb_dist_red = DistanceUtils.getDistanceYCrCb(predict, sample_red)
        ycrcb_dist_yellow = DistanceUtils.getDistanceYCrCb(predict, sample_yellow)
        ycrcb_dist_black = DistanceUtils.getDistanceYCrCb(predict, sample_black)
        ycrcb_dist_white = DistanceUtils.getDistanceYCrCb(predict, sample_white)

        HSV_dist_red = DistanceUtils.getDistanceByHSV(predict, sample_red)
        HSV_dist_yellow = DistanceUtils.getDistanceByHSV(predict, sample_yellow)
        HSV_dist_black = DistanceUtils.getDistanceByHSV(predict, sample_black)
        HSV_dist_white = DistanceUtils.getDistanceByHSV(predict, sample_white)

        RGB_dist_red = DistanceUtils.getDistanceByRGB(predict, sample_red)
        RGB_dist_yellow = DistanceUtils.getDistanceByRGB(predict, sample_yellow)
        RGB_dist_black = DistanceUtils.getDistanceByRGB(predict, sample_black)
        RGB_dist_white = DistanceUtils.getDistanceByRGB(predict, sample_white)

        # melt = [melt_dist_red, melt_dist_yellow, melt_dist_black, melt_dist_white]
        labs = [lab_dist_red, lab_dist_yellow, lab_dist_black, lab_dist_white]
        ycrcbs = [ycrcb_dist_red, ycrcb_dist_yellow, ycrcb_dist_black, ycrcb_dist_white]
        hsvs = [HSV_dist_red, HSV_dist_yellow, HSV_dist_black, HSV_dist_white]
        rgbs = [RGB_dist_red, RGB_dist_yellow, RGB_dist_black, RGB_dist_white]

        colors = [ImgUtils.KEY_SAMPLE_RED, ImgUtils.KEY_SAMPLE_YELLOW, ImgUtils.KEY_SAMPLE_BLACK,
                  ImgUtils.KEY_SAMPLE_WHITE]

        def getColorByMinimunDistance(arr):
            index = arr.index(min(arr))
            return colors[index]

        return {
            # "melt": [melt, getColorByMinimunDistance(melt)],
            "lab": [labs, getColorByMinimunDistance(labs)],
            "ycrcb": [ycrcbs, getColorByMinimunDistance(ycrcbs)],
            "hsv": [hsvs, getColorByMinimunDistance(hsvs)],
            "rgb": [rgbs, getColorByMinimunDistance(rgbs)],
            "order": colors
        }
