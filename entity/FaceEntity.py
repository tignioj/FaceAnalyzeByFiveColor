from entity.ROIEntity import ROIEntity


class FaceEntity:
    def __init__(self, num=0, name=None, gender=None,
                 imgPath=None,
                 srcImg=None, drawImg=None, facePart=None, landMark64=None,
                 landMarkROI={},
                 facePartOnlySkin=None, rectanglePoint=None, xywh=None, ):
        self.num = num
        self.name = name
        self.gender = gender
        self.imgPath = imgPath
        self.landMark64 = landMark64
        self.landMarkROIDict = landMarkROI
        self.srcImg = srcImg
        self.drawImg = drawImg
        self.facePart = facePart
        self.facePartOnlySkin = facePartOnlySkin
        self.xywh = xywh  # point of face part: left, top, width, height
        self.rectanglePoint = rectanglePoint