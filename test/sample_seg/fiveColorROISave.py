import os.path

import imutils

from core.const_var import *
from core.FaceLandMark import faceDetection
import cv2

blackImg = cv2.imread("../../four_color_face_sample/black.png")
chiImg = cv2.imread("../../four_color_face_sample/chi.png")
whiteImg = cv2.imread("../../four_color_face_sample/white.png")
yellowImg = cv2.imread("../../four_color_face_sample/yellow.png")
predict1Img = cv2.imread("../../faces/7.jpeg")
predict2Img = cv2.imread("../../faces/5.JPG")
predict3ImgWhite = cv2.imread("../../faces/white.jpg")
predict4ImgDark = cv2.imread("../../faces/dark.jpeg")


def cs(img, title=None):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segROIAndSave(name, img):
    faceEntity = faceDetection.faceDetectByImg(img)[0]
    roiPATH = OUTPUT_PATH + "\\" + name
    if not os.path.isdir(roiPATH):
        os.makedirs(roiPATH)

    cv2.imwrite(roiPATH + "\\" + name + "_draw.jpg", faceEntity.drawImg)

    cv2.imwrite(roiPATH + "\\" + name + "_face.jpg", faceEntity.facePart)

    cv2.imwrite(roiPATH + "\\" + name + "_face_trim.jpg", faceEntity.facePartOnlySkin)

    rois = faceEntity.landMarkROIDict

    for (name, roi) in rois.items():
        roi.img = imutils.resize(roi.img, width=50)
        roi.imgOnlySkin= imutils.resize(roi.imgOnlySkin, width=50)
        print(roi)
        cv2.imwrite(roiPATH + "\\" + roi.roiName + ".jpg", roi.img)
        cv2.imwrite(roiPATH + "\\" + roi.roiName + "_trim.jpg", roi.imgOnlySkin)


segROIAndSave('black', blackImg)
segROIAndSave('chi', chiImg)
segROIAndSave('white', whiteImg)
segROIAndSave('yellow', yellowImg)
segROIAndSave('predict1', predict1Img)
segROIAndSave('predict2', predict2Img)
segROIAndSave('predict3_white', predict3ImgWhite)
segROIAndSave('predict4_dark', predict4ImgDark)
segROIAndSave('predict_dark', predict4ImgDark)
segROIAndSave('predict_white', predict3ImgWhite)