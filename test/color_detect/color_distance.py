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


chi_sample = cv2.imread("../../result/chi/ting_trim.jpg")
black_sample = cv2.imread("../../result/black/ting_trim.jpg")
white_sample = cv2.imread("../../result/white/ting_trim.jpg")
yellow_sample = cv2.imread("../../result/yellow/ting_trim.jpg")


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
            if (predict[i][j] == (0, 0, 0)).all() or (sample[i][j] == (0, 0, 0)).all():
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
    distance = ((pl - sl) ** 2 + (pa - sa) ** 2 + (pb - sb) ** 2)
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
    distance = ((ph - sh) ** 2 + (ps - ss) ** 2 + (pv - sv) ** 2)
    return distance


def distance(name="0x123", sample=None):
    # predict = cv2.imread("../../result/predict1/ting_trim.jpg")
    # predict = cv2.imread("../../result/predict2/ting_trim.jpg")
    # predict = cv2.imread("../../result/predict3_white/ting_trim.jpg")
    # predict = cv2.imread("../../result/chi/ming_tang_trim.jpg")
    # predict = cv2.imread("../../result/black/ming_tang_trim.jpg")
    predict = cv2.imread("../../result/black/ting_trim.jpg")
    # predict = cv2.imread("../../result/predict4_dark/ting_trim.jpg")
    predict = cv2.resize(predict, (sample.shape[1], sample.shape[0]))
    res = np.hstack([predict, sample])
    cv2.imshow(name, res)
    return getDistance1(predict, sample), getDistance2ByLab(predict, sample), getDistance2BHSV(predict, sample)


dt_chi = distance('chi', chi_sample)
dt_black = distance('black', black_sample)
dt_white = distance('white', white_sample)
dt_yellow = distance('yellow', yellow_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()
font = ImageFont.truetype("../../fonts/simsun.ttc", 15)

a = np.vstack([dt_chi, dt_black, dt_white, dt_yellow])
# b = 1 - (a / a.sum(axis=0, keepdims=1))
b = a / a.sum(axis=0, keepdims=1)
# x = ['chi', 'black', 'white', 'yellow']
x = ['赤', '黑', '白', '黄']
# x = np.asarray([1,2,3,4])

# plt.figure(figsize=(100, 50), dpi=8)
plt.ylim(0, 1)
plt.title("预测值")
plt.xlabel('four color')
plt.ylabel('probability')
# $正则$
# plt.yticks([0.2, 0.4, 0.6, 0.8], ['r$very\ bad$', r'$bad$', 'normal', 'good'])
plt.yticks([0.2, 0.4, 0.6, 0.8])

barwidth = 0.24
x1 = list(range(len(x)))
y1 = b.transpose()[:1][0]
x2 = [i + barwidth for i in x1]
y2 = b.transpose()[1:2][0]
x3 = [i + barwidth * 2 for i in x1]
y3 = b.transpose()[2:3][0]

plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置x轴刻度
plt.xticks(x2, x)

i = 0
for xx1, xx2, xx3, yy1, yy2, yy3, in zip(x1, x2, x3, y1, y2, y3):
    # ha: horizontal alignment
    plt.text(xx1, np.round(y1[i] + 0.04, 4), '%.2f' % yy1, ha='center', va='top')
    plt.text(xx2, np.round(y2[i] + 0.04, 4), '%.2f' % yy2, ha='center', va='top')
    plt.text(xx3, np.round(y3[i] + 0.04, 4), '%.2f' % yy3, ha='center', va='top')
    i += 1

plt.bar(x1, y1, label='Euclidean Distance', width=barwidth)
plt.bar(x2, y2, label='Lab distance', width=barwidth)
plt.bar(x3, y3, label='HSV', width=barwidth)
plt.legend()
plt.show()
