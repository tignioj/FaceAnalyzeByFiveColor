import matplotlib.pyplot as plt
from PIL import ImageFont

from core.const_var import *
import numpy as np
import cv2
from utils.SkinUtils import *
from core.FaceLandMark import faceDetection


def cs(titile="0x123", img=None):
    cv2.imshow(titile, img)
    cv2.waitKey(0)
    cv2.destroyWindow(titile)


# img_predict_roi_ting = cv2.resize(img_predict_roi_ting, img_sample_roi_ting.shape[::-1][1:3])

def getDistance1(predict, sample):
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
            # if (predict[i][j] == (0, 0, 0)).all() or (sample[i][j] == (0, 0, 0)).all():
            if (predict[i][j] == (0, 0, 0)).all():
                continue

            A = predict[i][j]
            B = sample[i][j]
            # np.linalg.norm(A - B) 等同于
            # np.sqrt(np.sum((A[0] - B[0])**2 + (A[1] - B[1])**2 +(A[2] - B[2])**2))
            dist_byloop.append(np.linalg.norm(A - B))
            # sum += np.sqrt(np.sum(np.square(predict[i][j] - sample[i][j])))
    return np.mean(dist_byloop)


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
    # distance = ((pl - sl) ** 2 + (pa - sa) ** 2 + (pb - sb) ** 2)
    distance = ((pa - sa) ** 2 + (pb - sb) ** 2)
    return distance


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
    # distance = ((ph - sh) ** 2 + (ps - ss) ** 2 + (pv - sv) ** 2)
    distance = (ph - sh) ** 2
    return distance


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
    sample_YCrCb = cv2.cvtColor(sample, cv2.COLOR_BGR2YCrCb)
    pY, pCr, pCb = trimBlack(predict_YCrCb)
    sY, sCr, sCb = trimBlack(sample_YCrCb)
    # distance = ((pY - sY) ** 2 + (pCr - sCr) ** 2 + (pCb - sCb) ** 2)
    distance = ((pCr - sCr) ** 2 + (pCb - sCb) ** 2)
    return distance


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


def getImg(path):
    img = cv2.imread(path)
    return imutils.resize(img, 200, 200)


chi_sample = getImg("../../result/chi/ting_trim.jpg")
black_sample = getImg("../../result/black/ting_trim.jpg")
white_sample = getImg("../../result/white/ting_trim.jpg")
yellow_sample = getImg("../../result/yellow/ting_trim.jpg")




def distance(predict, name=""):
    dist_red = getDistance1(predict, chi_sample)
    dist_yellow = getDistance1(predict, yellow_sample)
    dist_black = getDistance1(predict, black_sample)
    dist_white = getDistance1(predict, white_sample)
    d = [dist_red, dist_yellow, dist_black, dist_white]
    ind = d.index(min(d))
    if ind == 0: color = "红色"
    if ind == 1: color = "黄色"
    if ind == 2: color = "黑色"
    if ind == 3: color = "白色"

    font = ImageFont.truetype("../../fonts/simsun.ttc", 10)
    x = ['赤', '黄', '黑', '白']
    plt.title("计算得到:" + color)
    plt.xlabel('four color')
    plt.ylabel('probability')

    # 计算概率
    n = np.asarray([dist_red, dist_yellow, dist_black, dist_white])
    n = sum(n) - n
    norm = np.linalg.norm(n)
    normal_array = n / norm

    barwidth = 0.15
    x1 = list(range(len(x)))  # [0,1,2,3]
    y1 = normal_array
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置x轴刻度
    plt.xticks(x1, x)
    plt.ylim(0, 1)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1])

    for x, y in zip(x1, y1):
        # ha: horizontal alignment
        plt.text(x, y + 0.025, '%.2f' % y, ha='center', va='top')

    plt.bar(x1, y1, label='融合计算')
    plt.legend()


# predict_red = getImg("../../result/chi/ting_trim.jpg")
predict_white = getImg("../../result/predict3_white/ting_trim.jpg")
distance(predict_white, "白色")
plt.show()
