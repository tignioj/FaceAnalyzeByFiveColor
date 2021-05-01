import cv2

"""
给定一张RGB图片路径，将其分割成各种色彩空间的元组:包括
RGB
HSB
HSV
YCbCr
Lab
"""


class ColorSpaceTransform:
    def __init__(self, img):
        self.img = img

    img = cv2.imread("img")
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)