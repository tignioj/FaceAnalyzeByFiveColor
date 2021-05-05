class ROIEntity:
    def __init__(self, roiName=None, img=None, imgOnlySkin=None, belongToLabelName=None, roiRectanglePoints=None,
                 centerPoint=None):
        self.roiName = roiName  # 庭、阙等 key name
        self.img = img
        self.imgOnlySkin = imgOnlySkin
        self.belongToLabelName = belongToLabelName
        self.belongToLabelName = None  # 属于五色中的哪一种,比如青色
        self.roiRectanglePoints = roiRectanglePoints
        self.centerPoint = centerPoint
