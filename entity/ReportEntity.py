from core.const_var import OUTPUT_PATH


class ReportEntity:
    def __init__(self, username=None,roiHistograms=None, gender=None,facePart=None,
                 roiDict=None, roiRGBDict=None, roiHSVDict=None, roiYCrCbDict=None, roiLabDict=None, roiColorResults=None,drawImg=None, imgPath=None):
        """
        封装检测后的报告
        :param username: 用户名
        :param roiHistograms: roi的直方图
        :param gender: 性别
        :param facePart: 脸部区域
        :param roiDict: roi原数组
        :param roiRGBDict: roi在RGB空间上的绘制
        :param roiHSVDict: roi在HSV空间上的绘制
        :param roiYCrCbDict: roi在YCrCb空间上的绘制
        :param roiLabDict: roi在Lab空间上的绘制
        :param roiColorResults: roi预测结果
        :param drawImg: 绘制出landmark的图片
        :param imgPath: 图片路径
        """
        self.username = username
        self.gender = gender
        self.facePart = facePart
        self.roiDict = roiDict
        self.roiRGBDict = roiRGBDict
        self.roiHSVDict = roiHSVDict
        self.roiYCrCbDict = roiYCrCbDict
        self.roiLabDict = roiLabDict
        self.roiHistograms = roiHistograms
        self.roiColorResults = roiColorResults
        self.drawImg = drawImg
        self.imgPath = imgPath
