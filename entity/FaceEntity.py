from entity.ROIEntity import ROIEntity


class FaceEntity:
    def __init__(self, num=0, name=None, gender=None,
                 imgPath=None,
                 srcImg=None, drawImg=None, facePart=None, landMark68=None,
                 landMarkROI={},
                 facePartOnlySkin=None, rectanglePoint=None, xywh=None, ):
        """
        封装检测到的人脸
        :param num:  检测到的序号
        :param name:  姓名
        :param gender: 性别
        :param imgPath: 图像保存位置
        :param srcImg: BGR数组形式的原图
        :param drawImg: ROI绘制在脸部的图片
        :param facePart: 切割后的脸部图片
        :param landMark68: 68个人脸关键点
        :param landMarkROI: landMark数组，存放ROIEntity
        :param facePartOnlySkin: 提纯肤色后的图片
        :param rectanglePoint: 脸部检测后的四个矩形坐标，表示脸部的位置
        :param xywh: 脸部检测后的左上角坐标和脸部的长、宽
        """
        self.num = num # 脸部序号
        self.name = name
        self.gender = gender
        self.imgPath = imgPath
        self.landMark68 = landMark68
        self.landMarkROIDict = landMarkROI
        self.srcImg = srcImg
        self.drawImg = drawImg
        self.facePart = facePart
        self.facePartOnlySkin = facePartOnlySkin
        self.xywh = xywh  # point of face part: left, top, width, height
        self.rectanglePoint = rectanglePoint
