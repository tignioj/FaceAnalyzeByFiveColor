from core.const_var import OUTPUT_PATH


class ReportEntity:
    def __init__(self, username=None,roiHistograms=None, gender=None, faceColor=None, skinResult=None, facePart=None):
        self.username = username
        self.gender = gender
        self.faceColor = faceColor
        self.facePart = facePart
        self.roiDict = None
        self.roiRGBDict = None
        self.roiHSVDict = None
        self.roiYCrCbDict = None
        self.roiLabDict = None
        self.roiHistograms = None
        self.roiColorResults = None
        self.drawImg = None
        self.skinResult = skinResult
        self.imgPath = None
