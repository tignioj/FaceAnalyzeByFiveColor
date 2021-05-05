from core.const_var import OUTPUT_PATH


class ReportEntity:
    def __init__(self, username, gender, faceColor, skinResult, glossResult, image):
        self.username = username
        self.gender = gender
        self.faceColor = faceColor
        self.skinResult = skinResult
        self.glossResult = glossResult
        self.image = image
        self.imgPath = OUTPUT_PATH + self.username + '\\_DiagnoseResult.jpg'
