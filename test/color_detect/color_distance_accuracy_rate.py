import matplotlib.pyplot as plt
from PIL import ImageFont

from utils.ImageUtils import getSampleDict
from core.const_var import *
import numpy as np
import cv2

from utils.LogUtils import LogUtils
from utils.SkinUtils import *
from core.FaceLandMark import faceDetection
from utils.DistanceUtils import DistanceUtils


def cs(titile="0x123", img=None):
    cv2.imshow(titile, img)
    cv2.waitKey(0)
    cv2.destroyWindow(titile)


# img_predict_roi_ting = cv2.resize(img_predict_roi_ting, img_sample_roi_ting.shape[::-1][1:3])

def getImg(path):
    img = cv2.imread(path)
    return imutils.resize(img, 200, 200)


chi_sample = getImg("../../result/chi/ting_trim.jpg")
black_sample = getImg("../../result/black/ting_trim.jpg")
white_sample = getImg("../../result/white/ting_trim.jpg")
yellow_sample = getImg("../../result/yellow/ting_trim.jpg")


def getProbability(arr):
    # return softmax(arr)
    return norm(arr)
    # return original(arr)
    # return normAndSf(arr)


def normAndSf(arr):
    return softmax(norm(arr))


def original(arr):
    a = np.asarray(arr)
    # return softmax(1 - a / sum(a))
    return 1 - a / sum(a)


def norm(arr):
    nparray = np.asarray(arr)
    # arr = 1 - nparray / nparray.max()
    nparray = sum(nparray) - nparray
    norm = np.linalg.norm(nparray)
    normal_array = nparray / norm
    return normal_array


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def drawHistByDistance(array, color, color_order, number, position, label):
    """
    :param array: 距离数组
    :param color: 数组计算所得到的颜色
    :param color_order: x坐标标签
    :param number: 一共共几个柱状图
    :param position 画在哪个位置
    :return:
    """
    # font = ImageFont.truetype("../../fonts/simsun.ttc", 5)
    x = color_order
    # plt.title("计算得到:" + color)
    # plt.figure(figsize=(9,9))
    plt.xlabel('肤色指数')
    plt.ylabel('probability')

    # 计算概率
    plt.rcParams['axes.unicode_minus'] = False

    barwidth = 0.12
    bias = position * barwidth
    x1 = np.asarray(list(range(len(x)))) + bias  # [0,1,2,3]
    y1 = getProbability(array)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置x轴刻度
    plt.xticks(x1, x)
    # plt.ylim(0, 1)
    # plt.yticks([0.2, 0.4, 0.6, 0.8, 1])

    for x, y in zip(x1, y1):
        # ha: horizontal alignment
        plt.text(x, y + 0.028, '%.2f' % y, ha='center', va='top')

    plt.bar(x1, y1, label=label, width=barwidth)
    plt.legend()


def getDistanceByDifferentColorSpace(roi=None):
    black_sample_dict, yellow_sample_dict, red_sample_dict, white_sample_dict = getSampleDict()
    # 要预测的图片
    # predict = getImg("../../result/predict3_white/ting.jpg")
    # predict = getImg("../../result/chi/ting.jpg")
    # predict = getImg("../../result/yellow/ting.jpg")
    predict = getImg("../../result/white/ting.jpg")

    # 1. 获取阙的四色样本

    # 2. 计算距离
    # 2.1 获取全部颜色空间的距离
    distanceDict = DistanceUtils.getDistArray(predict,
                                              black_sample_dict[KEY_ting],
                                              yellow_sample_dict[KEY_ting],
                                              red_sample_dict[KEY_ting],
                                              white_sample_dict[KEY_ting])

    # 2.2 获取melts算法的距离
    melts = distanceDict["melt"][0]
    melt_color = distanceDict["melt"][1]

    labs = distanceDict["lab"][0]
    lab_color = distanceDict["lab"][1]

    hsvs = distanceDict["hsv"][0]
    hsv_color = distanceDict["hsv"][1]

    rgbs = distanceDict["rgb"][0]
    rgb_color = distanceDict["rgb"][1]

    ycrcbs = distanceDict["ycrcb"][0]
    ycrcbs_color = distanceDict["ycrcb"][1]

    arr = [melts, labs, hsvs, rgbs, ycrcbs]

    order = distanceDict['order']

    # 2.3 根据距离画图
    drawHistByDistance(melts, melt_color, order, 5, 1, 'melt')
    drawHistByDistance(labs, lab_color, order, 5, 2, 'Lab')
    drawHistByDistance(rgbs, rgb_color, order, 5, 3, 'RGB')
    drawHistByDistance(hsvs, rgb_color, order, 5, 4, 'HSV')
    drawHistByDistance(ycrcbs, ycrcbs_color, order, 5, 5, 'YCrCb')


def normArr(arr):
    nparray = np.asarray(arr)
    nparray = sum(nparray) - nparray
    norm = np.linalg.norm(nparray)
    normal_array = nparray / norm
    return normal_array


def normDistance(arr):
    na = [normArr(a) for a in arr]
    nna = normArr(na)
    return nna


getDistanceByDifferentColorSpace()
plt.show()

"""
目的：一次只能提供一个ROI，用五种算法求出与样本的距离
"""


def distance(predict, name=""):
    dist_red = DistanceUtils.getDistance1(predict, chi_sample)
    dist_yellow = DistanceUtils.getDistance1(predict, yellow_sample)
    dist_black = DistanceUtils.getDistance1(predict, black_sample)
    dist_white = DistanceUtils.getDistance1(predict, white_sample)
    d = [dist_red, dist_yellow, dist_black, dist_white]
    index = d.index(min(d))
    if index == 0: color = "红色"
    if index == 1: color = "黄色"
    if index == 2: color = "黑色"
    if index == 3: color = "白色"

    font = ImageFont.truetype("../../fonts/simsun.ttc", 10)
    x = ['赤', '黄', '黑', '白']
    plt.title("计算得到:" + color)
    plt.xlabel('肤色指数')
    plt.ylabel('probability')

    # 计算概率
    nparray = np.asarray(d)
    nparray = sum(nparray) - nparray
    norm = np.linalg.norm(nparray)
    normal_array = nparray / norm

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

# # predict_red = getImg("../../result/chi/ting_trim.jpg")
# # predict_white = getImg("../../result/predict3_white/ting_trim.jpg")
# predict_white = getImg("../../result/predict4_dark/ting_trim.jpg")
# distance(predict_white, "白色")
# plt.show()
