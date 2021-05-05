import cv2

from utils.ColorSpaceTransform import *


def trimSkin(img):
    """
    使用RGB_HSV_YCrCb的融合方法去掉非皮肤
    :param img:
    :return:
    """
    return rgb_hsv_ycbcr(img)
