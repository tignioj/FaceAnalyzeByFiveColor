'''
    本程序实现功能是人脸检测，并分割人脸区域，然后进行面部健康诊断
'''
import cv2 as cv
from entity.ReportEntity import ReportEntity
import math
import matplotlib.pyplot as plt

import imutils
import numpy as np
from FaceLandMark import faceDetection
import copy
# from tongueDiagnose import TongueDiagnose
from const_var import *

'''
    基于图片的人脸检测
    scaleFactor 指定在每个图像所放缩图像大小减少了多少
    minNeighbors 指定每个候选矩形应该保留多少个邻居
    minsize 最小可能的对象大小
    maxsize 最大可能的对象大小
'''


class FaceNotFoundException(Exception):
    """
    Face not found!
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def faceDetect(img, scale,username=None, gender=None):
    faces = faceDetection.faceDetectByImg(img, scale, username, gender)
    if len(faces) <= 0:
        raise FaceNotFoundException(img, "Face Not found on image.")

    return faces


def face_color(faceskinimage):
    """
    面部颜色检测
    :param faceskinimage:
    :param faceblock:
    :return:
    """
    img_lab = cv.cvtColor(faceskinimage, cv.COLOR_BGR2Lab)
    L_value, A_value, B_value = cv.split(img_lab)


def cvShowImg(img):
    img = imutils.resize(img, width=800, height=800)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    img = cv.imread("../faces/7.jpeg")
    # cvShowImg(img)
    faceDetect(img)

# if __name__ == "__main__":
#     # 输入检测人员信息
#     filename = "faceDetectResults.txt"
#     wordfile = "D:\GitDoc\FaceHealthDetect\FaceDiagnoseResults.docx"
#     pdffile = "D:\GitDoc\FaceHealthDetect" + "\FaceDiagnoseResults.pdf"
#     name = "孙悦"
#     gender = 1
#
#     # 加载图片
#     # img = cv.imread('testYao.jpg', cv.IMREAD_COLOR)
#     # img = cv.imread('testOfC.jpg', cv.IMREAD_COLOR)
#     img = cv.imread('selfieOfSun.jpg', cv.IMREAD_COLOR)
#     # img = cv.imread('InkedselfieOfSun.jpg', cv.IMREAD_COLOR)
#
#     # 显示原始图片
#     # cv.imshow('origin', img)
#     # cv.waitKey(0)
#
#     # 进行人脸检测
#     faceColor, faceGloss, img = faceDetect(img, 1)
#
#     # 根据人脸检测情况和人员信息，生成诊断结果
#     SkinResults, GlossResults = CreateDetectResults(faceColor, faceGloss)
#     # pdfCreate(filename, name, sex, faceColor, faceGloss, SkinResults, GlossResults)
#     image = "DiagnoseResult.jpg"
#     wordCreate(name, gender, faceColor, faceGloss, SkinResults, GlossResults, image)
#     word2pdf(wordfile, pdffile)
