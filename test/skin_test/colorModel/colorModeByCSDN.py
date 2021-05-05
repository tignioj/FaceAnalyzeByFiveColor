import cv2
import numpy as np


# https://blog.csdn.net/weixin_40893939/article/details/84527037
def ellipse_detect(img):
    # img = cv2.imread(image, cv2.IMREAD_COLOR)
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 15), 43, 0, 360, (255, 255, 255), -1)
    YCRCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            CR = YCRCB[i, j, 1]
            CB = YCRCB[i, j, 2]
            if skinCrCbHist[CR, CB] > 0:
                skin[i, j] = 255
    dst = cv2.bitwise_and(img, img, mask=skin)
    return dst


def cr_otsu(img):
    """
    2 YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
    原理 针对YCRCB中CR分量的处理，将RGB转换为YCRCB，对CR通道单独进行otsu处理，otsu方法opencv里用threshold
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.namedWindow("image raw", cv2.WINDOW_NORMAL)
    cv2.imshow("image raw", img)
    cv2.namedWindow("image CR", cv2.WINDOW_NORMAL)
    cv2.imshow("image CR", cr1)
    cv2.namedWindow("Skin Cr+OTSU", cv2.WINDOW_NORMAL)
    cv2.imshow("Skin Cr+OTSU", skin)

    dst = cv2.bitwise_and(img, img, mask=skin)
    return dst


def crcb_range_sceening(img):
    """
    3 基于YCrCb颜色空间Cr, Cb范围筛选法
    原理:类似于第二种方法，只不过是对CR和CB两个通道综合考虑
    :param image: 图片路径
    :return: None
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb)

    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            if (cr[i][j] > 140) and (cr[i][j]) < 175 and (cr[i][j] > 100) and (cb[i][j]) < 120:
                skin[i][j] = 255
            else:
                skin[i][j] = 0
    cv2.namedWindow("skin2 cr+cb", cv2.WINDOW_NORMAL)
    cv2.imshow("skin2 cr+cb", skin)
    dst = cv2.bitwise_and(img, img, mask=skin)
    # cv2.imshow("cutout", dst)
    return dst


def hsv_detect(img):
    """
    :param image: 图片路径
    :return: None
    """
    # img = cv2.imread(image, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (_h, _s, _v) = cv2.split(hsv)
    skin = np.zeros(_h.shape, dtype=np.uint8)
    (x, y) = _h.shape

    for i in range(0, x):
        for j in range(0, y):
            if (_h[i][j] > 7) and (_h[i][j] < 20) and (_s[i][j] > 28) and (_s[i][j] < 255) and (_v[i][j] > 50) and (
                    _v[i][j] < 255):
                skin[i][j] = 255
            else:
                skin[i][j] = 0
    cv2.imshow("hsv", skin)
    dst = cv2.bitwise_and(img, img, mask=skin)
    # cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
    # cv2.imshow("cutout", dst)
    return dst


if __name__ == '__main__':
    filepath = "../../../faces/7.jpeg"
    srcimg = cv2.imread(filepath)
    # img = ellipse_detect(srcimg)
    # img = cr_otsu(srcimg)
    # img = crcb_range_sceening(srcimg)
    img = hsv_detect(srcimg)
    cv2.imshow('d', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
