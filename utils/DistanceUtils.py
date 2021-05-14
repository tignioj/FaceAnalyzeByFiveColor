import numpy as np
import cv2

from utils.SkinUtils import SkinUtils
from utils.ImageUtils import ImgUtils


class DistanceUtils:
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


        # melt_dist_red = SkinUtils.getDistanceByRGB(predict, sample_red)
        # melt_dist_yellow = SkinUtils.getDistanceByRGB(predict, sample_yellow)
        # melt_dist_black = SkinUtils.getDistanceByRGB(predict, sample_black)
        # melt_dist_white = SkinUtils.getDistanceByRGB(predict, sample_white)

        lab_dist_red = SkinUtils.getDistanceByLab(predict, sample_red)
        lab_dist_yellow = SkinUtils.getDistanceByLab(predict, sample_yellow)
        lab_dist_black = SkinUtils.getDistanceByLab(predict, sample_black)
        lab_dist_white = SkinUtils.getDistanceByLab(predict, sample_white)

        ycrcb_dist_red = SkinUtils.getDistanceYCrCb(predict, sample_red)
        ycrcb_dist_yellow = SkinUtils.getDistanceYCrCb(predict, sample_yellow)
        ycrcb_dist_black = SkinUtils.getDistanceYCrCb(predict, sample_black)
        ycrcb_dist_white = SkinUtils.getDistanceYCrCb(predict, sample_white)

        HSV_dist_red = SkinUtils.getDistanceByHSV(predict, sample_red)
        HSV_dist_yellow = SkinUtils.getDistanceByHSV(predict, sample_yellow)
        HSV_dist_black = SkinUtils.getDistanceByHSV(predict, sample_black)
        HSV_dist_white = SkinUtils.getDistanceByHSV(predict, sample_white)

        RGB_dist_red = SkinUtils.getDistanceByRGB(predict, sample_red)
        RGB_dist_yellow = SkinUtils.getDistanceByRGB(predict, sample_yellow)
        RGB_dist_black = SkinUtils.getDistanceByRGB(predict, sample_black)
        RGB_dist_white = SkinUtils.getDistanceByRGB(predict, sample_white)

        # melt = [melt_dist_red, melt_dist_yellow, melt_dist_black, melt_dist_white]
        labs = [lab_dist_red, lab_dist_yellow, lab_dist_black, lab_dist_white]
        ycrcbs = [ycrcb_dist_red, ycrcb_dist_yellow, ycrcb_dist_black, ycrcb_dist_white]
        hsvs = [HSV_dist_red, HSV_dist_yellow, HSV_dist_black, HSV_dist_white]
        rgbs = [RGB_dist_red, RGB_dist_yellow, RGB_dist_black, RGB_dist_white]

        colors = [ImgUtils.KEY_SAMPLE_RED, ImgUtils.KEY_SAMPLE_YELLOW, ImgUtils.KEY_SAMPLE_BLACK, ImgUtils.KEY_SAMPLE_WHITE]

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
