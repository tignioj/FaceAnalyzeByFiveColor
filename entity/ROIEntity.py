class ROIEntity:
    def __init__(self, roiName=None, img=None, imgOnlySkin=None, roiRectanglePoints=None,
                 centerPoint=None):
        """
        ROI实体类，封装单个ROI
        :param roiName: ROI名字
        :param img: ROI图片
        :param imgOnlySkin: 肤色提纯后的图片
        :param roiRectanglePoints: ROI矩形做白哦
        :param centerPoint: 中心点
        """
        self.roiName = roiName  # 庭、阙等 key name
        self.img = img
        self.imgOnlySkin = imgOnlySkin
        self.roiRectanglePoints = roiRectanglePoints
        self.centerPoint = centerPoint
