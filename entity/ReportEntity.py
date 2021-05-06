from core.const_var import OUTPUT_PATH


class ReportEntity:
    def __init__(self, username=None, gender=None, faceColor=None, skinResult=None, facePart=None):
        self.username = username
        self.gender = gender
        self.faceColor = faceColor
        self.facePart = facePart
        self.rois = None
        self.roiRGBs = None
        self.roiHSVs = None
        self.roiYCrCb = None
        self.drawImg = None
        self.skinResult = skinResult
        self.imgPath = None

