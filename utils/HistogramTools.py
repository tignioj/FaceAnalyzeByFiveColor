import matplotlib.pyplot as plt
import numpy  as np
import itertools
import operator
from core.const_var import FACIAL_LANDMARKS_NAME_DICT
from core.const_var import KEY_ting
from utils.DistanceUtils import DistanceUtils
from utils.ImageUtils import ImgUtils
from utils.LogUtils import LogUtils
from utils.SkinUtils import SkinUtils


class ColorNotFoundException(Exception):
    pass


class HistogramTools:
    @staticmethod
    def _getProbability(arr):
        # return softmax(arr)
        return HistogramTools._norm(arr)
        # return original(arr)
        # return normAndSf(arr)

    @staticmethod
    def _original(arr):
        a = np.asarray(arr)
        # return softmax(1 - a / sum(a))
        return 1 - a / sum(a)

    @staticmethod
    def _norm(arr):
        nparray = np.asarray(arr)
        # arr = 1 - nparray / nparray.max()
        nparray = sum(nparray) - nparray
        norm = np.linalg.norm(nparray)
        normal_array = nparray / norm
        return normal_array

    @staticmethod
    def _softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def _drawHistByDistance(array, color, color_order, number, position, label):
        """
        :param array: 距离数组
        :param color: 数组计算所得到的颜色
        :param color_order: x坐标标签
        :param number: 一共共几个柱状图
        :param position 画在哪个位置
        :return:
        """
        x = color_order

        plt.xlabel('肤色指数')
        plt.ylabel('probability')
        # 正确显示负号
        plt.rcParams['axes.unicode_minus'] = False

        barwidth = 0.12
        bias = position * barwidth
        x1 = np.asarray(list(range(len(x)))) + bias  # [0,1,2,3]
        # y1 = HistogramTools._getProbability(array)
        y1 = array
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

    @staticmethod
    def getDistanceByDifferentColorSpace(fig, predict=None, keyName=None, sampleDict=None):
        """
        :param fig:  传入一直画笔，该画笔必须先调用 fig = plt.figure()
        :param predict: 要预测的ROI图片
        :param keyName: ROI名字, 用于查找相应的样本
        :return: 返回格式：直方图, {诊断结果，对比图}
            colorResultWithImg = {
            'text': colorResultText,
            'img': colorDiffImg
        }
        """
        keyNameCN = FACIAL_LANDMARKS_NAME_DICT[keyName]
        LogUtils.log("HistogramTools", "calculating distance for'" + keyNameCN + "'....")
        if sampleDict is None:
            sampleDict = ImgUtils.getSampleDict()

        black_sample_dict, yellow_sample_dict, red_sample_dict, white_sample_dict = \
            sampleDict[ImgUtils.KEY_SAMPLE_BLACK], sampleDict[ImgUtils.KEY_SAMPLE_YELLOW], sampleDict[
                ImgUtils.KEY_SAMPLE_RED], sampleDict[
                ImgUtils.KEY_SAMPLE_WHITE]
        # 要预测的图片
        # predict = getImg("../../result/predict3_white/ting.jpg")
        # predict = getImg("../../result/chi/ting.jpg")
        # predict = getImg("../../result/yellow/ting.jpg")

        # 1. 获取阙的四色样本

        # 2. 计算距离
        # 2.1 获取全部颜色空间的距离

        LogUtils.log("HistogramTools", keyNameCN + "肤色提纯中...")
        predictTrim = SkinUtils.trimSkin(predict)

        sampleBlack = black_sample_dict[keyName]
        sampleBlackTrim = SkinUtils.trimSkin(sampleBlack)

        sampleYellow = yellow_sample_dict[keyName]
        sampleYellowTrim = SkinUtils.trimSkin(sampleYellow)

        sampleRed = red_sample_dict[keyName]
        sampleRedTrim = SkinUtils.trimSkin(sampleRed)

        sampleWhite = white_sample_dict[keyName]
        sampleWhiteTrim = SkinUtils.trimSkin(sampleWhite)
        LogUtils.log("HistogramTools", keyNameCN + "提纯完毕...")

        distanceDict = DistanceUtils.getDistArray(predictTrim,
                                                  sample_black=sampleBlackTrim,
                                                  sample_yellow=sampleYellowTrim,
                                                  sample_red=sampleRedTrim,
                                                  sample_white=sampleWhiteTrim)

        LogUtils.log("HistogramTools", keyNameCN + "距离数组：", distanceDict)
        # 2.2 获取melts算法的距离
        # melts = distanceDict["melt"][0]
        # melt_color = distanceDict["melt"][1]

        labs = distanceDict["lab"][0]
        lab_color = distanceDict["lab"][1]

        hsvs = distanceDict["hsv"][0]
        hsv_color = distanceDict["hsv"][1]

        rgbs = distanceDict["rgb"][0]
        rgb_color = distanceDict["rgb"][1]

        ycrcbs = distanceDict["ycrcb"][0]
        ycrcbs_color = distanceDict["ycrcb"][1]

        # arr = [melts, labs, hsvs, rgbs, ycrcbs]
        # arr = [melts, labs, hsvs, rgbs, ycrcbs]

        order = distanceDict['order']

        # 2.3 根据距离画图
        # _drawHistByDistance(melts, melt_color, order, 5, 1, 'melt')
        LogUtils.log("HistogramTools", "正在为'" + keyNameCN + "'画直方图")
        HistogramTools._drawHistByDistance(labs, lab_color, order, 5, 2, 'Lab')
        HistogramTools._drawHistByDistance(rgbs, rgb_color, order, 5, 3, 'RGB')
        HistogramTools._drawHistByDistance(hsvs, hsv_color, order, 5, 4, 'HSV')
        HistogramTools._drawHistByDistance(ycrcbs, ycrcbs_color, order, 5, 5, 'YCrCb')

        LogUtils.log("HistogramTools", "完成'" + keyNameCN + "'的绘制")
        colorResult = HistogramTools.getMostFrequentWord([
            lab_color, hsv_color, ycrcbs_color
        ])
        colorCNName = ImgUtils.COLOR_SAMPLE_CN_NAME_BY_KEY[colorResult]
        LogUtils.log("HistogramTools", "统计'" + keyNameCN + "'颜色结果为'" + colorCNName + "'色！")
        # 预测图片和相似样本图片
        LogUtils.log("HistogramTools", "正在搜索部位为'" + keyNameCN + "'的" + colorCNName + "色样本数据'")

        if colorResult == ImgUtils.KEY_SAMPLE_RED:
            sample = sampleRed
            sampleTrim = sampleRedTrim
        elif colorResult == ImgUtils.KEY_SAMPLE_BLACK:
            sample = sampleBlack
            sampleTrim = sampleBlackTrim
        elif colorResult == ImgUtils.KEY_SAMPLE_YELLOW:
            sample = sampleYellow
            sampleTrim = sampleYellowTrim
        elif colorResult == ImgUtils.KEY_SAMPLE_WHITE:
            sample = sampleWhite
            sampleTrim = sampleWhiteTrim
        else:
            raise ColorNotFoundException()

        colorDiffImg = {
            'predict': predict,
            'predictTrim': predictTrim,
            'sample': sample,
            'sampleTrim': sampleTrim
        }

        colorResultText = SkinUtils.getResultByOneColor(keyName, colorResult)
        colorResultWithImg = {
            'text': colorResultText,
            'color': colorResult,
            'img': colorDiffImg
        }
        LogUtils.log("HistogramTools", keyNameCN + "的直方图数据包装完成！")
        return ImgUtils.getcvImgFromFigure(fig), colorResultWithImg

    @staticmethod
    def getMostFrequentWord(arr):
        """
        获取出现频率做多的字符串
        :param arr:
        :return:
        """
        return HistogramTools.getMostCommenWord(arr)

    @staticmethod
    def getMostCommenWord(L):
        # get an iterable of (item, iterable) pairs
        SL = sorted((x, i) for i, x in enumerate(L))
        # print 'SL:', SL
        groups = itertools.groupby(SL, key=operator.itemgetter(0))

        # auxiliary function to get "quality" for an item
        def _auxfun(g):
            item, iterable = g
            count = 0
            min_index = len(L)
            for _, where in iterable:
                count += 1
                min_index = min(min_index, where)
            # print 'item %r, count %r, minind %r' % (item, count, min_index)
            return count, -min_index

        # pick the highest-count/earliest item
        return max(groups, key=_auxfun)[0]


if __name__ == '__main__':
    roi = ImgUtils.getImg("../result/predict_white/ke.jpg")
    fig = plt.figure(figsize=(10, 10))
    sampleDict = ImgUtils.getSampleDict()
    img, color = HistogramTools.getDistanceByDifferentColorSpace(fig, roi, KEY_ting, sampleDict)
    ImgUtils.cvshow(img)
    # d = getMostFrequentWord(['张三', '张三', '李斯', '王五', '赵六', '王五'])
