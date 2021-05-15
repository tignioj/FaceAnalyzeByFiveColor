from collections import OrderedDict

import imutils
from utils.LogUtils import LogUtils
import os
from entity.ROIEntity import ROIEntity
from entity.FaceEntity import FaceEntity
from utils.SkinUtils import SkinUtils
from utils.ImageUtils import ImgUtils
import cv2
import dlib
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from service.ROIService import ROIService
from core.const_var import *


class __FaceDetect:
    def __init__(self):

        self.__scale = 1

        # 3. 调用人脸检测器
        self._detector = dlib.get_frontal_face_detector()

        self.__ROIService = ROIService()

        # 4. 加载预测关键点模型(68个点)
        self._predictor = dlib.shape_predictor(
            BASE_PATH + "\\model\\face_landmark\\shape_predictor_68_face_landmarks.dat")

    def _shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def _putTextCN(self, clone, coord, name_CN, face):
        img = clone
        faceW = face.bottom() - face.top()
        # fontpath = "simsun.ttc"  # <== 这里是宋体路径
        font_size = int(round(faceW * 0.1))
        font = ImageFont.truetype("simsun.ttc", font_size)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        # 使得中文居中
        Xdistance = (font_size / 2) * len(name_CN)
        Ydistance = (font_size / 2)
        text_x = int(coord[0] - Xdistance)
        text_y = int(coord[1] - Ydistance)
        if text_x <= 5:
            text_x = 5
        if text_y <= 5:
            text_y = 5
        coord = (text_x, text_y)
        draw.text(coord, name_CN, font=font, fill=COLORDICT['green'])
        img = np.array(img_pil)
        return img

    def skinDetect(self, img, scale):
        # small_img = cv2.resize(img, (0, 0), fx=1 / scale, fy=1 / scale)
        frame = SkinUtils.trimSkinRealTime(img)
        return frame

    def faceDetectRealTime(self, img, scale=1):
        self.__scale = scale
        copy = img.copy()
        small_img = cv2.resize(copy, (0, 0), fx=1 / scale, fy=1 / scale)
        # small_img = imutils.resize(img, 300, 300)
        # print(small_img.shape)
        # return small_img
        # 6. 人脸检测
        faces = self._detector(small_img, 1)  # 1代表将图片放大1倍数
        print(len(faces))
        # 7. 循环，遍历每一张人脸， 绘制矩形框和关键点
        for (i, face) in enumerate(faces):
            shape = self._predictor(small_img, face)
            shape = self._shape_to_np(shape)
            for (nameKey, name_CN) in FACIAL_LANDMARKS_NAME_DICT.items():
                # LogUtils.log("FaceLandMark", "提取ROI..., 图像是否为空:" + str(len(img)))
                roiEntity = self.__ROIService.getROIRealTime(nameKey, shape, img, face, scale)
                # 根据点画出折线
                path = [roiEntity.roiRectanglePoints.reshape((-1, 1, 2))]
                cv2.polylines(copy, path, True, (0, 255, 0), 4)
                # 加上中文文字: 这个方法特别卡！
                # copy = self._putTextCN(copy, roiEntity.centerPoint, name_CN, face)
                # cv2.putText(copy, nameKey[0], (roiEntity.centerPoint[0], roiEntity.centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_4)
                # roi = imutils.resize(roiEntity.img, width=200, inter=cv2.INTER_CUBIC)
                # cv2.imshow(nameKey, roiEntity.img)
            # 画出人脸矩形
            cv2.rectangle(copy, (face.left() * scale, face.top() * scale),
                          (face.right() * scale, face.bottom() * scale), (255, 0, 0), 4, cv2.LINE_AA)
        return copy

    # def faceDetectRealTime(self, img, scale=1):
    #     self.__scale = scale
    #     copy = img.copy()
    #     small_img = cv2.resize(copy, (0, 0), fx=1 / scale, fy=1 / scale)
    #     # 6. 人脸检测
    #     faces = self._detector(small_img, 1)  # 1代表将图片放大1倍数
    #     # 7. 循环，遍历每一张人脸， 绘制矩形框和关键点
    #     for (i, face) in enumerate(faces):
    #         shape = self._predictor(small_img, face)
    #         shape = self._shape_to_np(shape)
    #         for (nameKey, name_CN) in FACIAL_LANDMARKS_NAME_DICT.items():
    #             # LogUtils.log("FaceLandMark", "提取ROI..., 图像是否为空:" + str(len(img)))
    #             roiEntity = self.__ROIService.getROIRealTime(nameKey, shape, img, face, scale)
    #             # 根据点画出折线
    #             path = [roiEntity.roiRectanglePoints.reshape((-1, 1, 2))]
    #             cv2.polylines(copy, path, True, (0, 255, 0), 4)
    #             # 加上中文文字: 这个方法特别卡！
    #             # copy = self._putTextCN(copy, roiEntity.centerPoint, name_CN, face)
    #             # cv2.putText(copy, nameKey[0], (roiEntity.centerPoint[0], roiEntity.centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_4)
    #             # roi = imutils.resize(roiEntity.img, width=200, inter=cv2.INTER_CUBIC)
    #             # cv2.imshow(nameKey, roiEntity.img)
    #         # 画出人脸矩形
    #         cv2.rectangle(copy, (face.left() * scale, face.top() * scale),
    #                       (face.right() * scale, face.bottom() * scale), (255, 0, 0), 4, cv2.LINE_AA)
    #     return copy

    def faceDetectByImg(self, img, scale=1, username=None, gender=None):
        # todayPath = OUTPUT_PATH + "\\" + getTodayYearMonthDayHourMinSec()
        # if not os.path.isdir(todayPath):
        #     os.mkdir(todayPath)
        LogUtils.log("FaceLandMark", "开始分析图片", progress=15)

        detectedFaces = []
        copy = img.copy()
        self.__scale = scale
        p = 1 / scale
        small_img = cv2.resize(copy, (0, 0), fx=1 / scale, fy=1 / scale)
        # 6. 人脸检测
        faces = self._detector(small_img, 1)  # 1代表将图片放大1倍数
        # 7. 循环，遍历每一张人脸， 绘制矩形框和关键点
        for (i, face) in enumerate(faces):
            # (1. 创建一个实体类
            fe = FaceEntity(i)
            fe.gender = gender
            fe.name = username
            # (2. 原始图像
            fe.srcImg = img
            shape = self._predictor(small_img, face)
            # (3. 64个关键点
            fe.landMark64 = shape
            shape = self._shape_to_np(shape)
            for (nameKey, name_CN) in FACIAL_LANDMARKS_NAME_DICT.items():
                LogUtils.log("FaceLandMark", "提取ROI'" + name_CN + "...", progress=20)
                roiEntity = self.__ROIService.getROI(nameKey, shape, img, face)
                # (4. ROI
                fe.landMarkROIDict[nameKey] = roiEntity
                # 根据点画出折线
                path = [roiEntity.roiRectanglePoints.reshape((-1, 1, 2))]
                cv2.polylines(copy, path, True, (0, 255, 0), 4)
                # 加上文字
                copy = self._putTextCN(copy, roiEntity.centerPoint, name_CN, face)
                # roi = imutils.resize(roi, width=200, inter=cv2.INTER_CUBIC)
                # cv2.imshow(nameKey, roi)

            # (5. 人脸矩形
            left = face.left() * scale
            top = face.top() * scale
            right = face.right() * scale
            bottom = face.bottom() * scale
            width = right - left
            height = bottom - top

            fe.xywh = (left, top, width, height)

            fe.rectanglePoint = ((left, top), (right, bottom))

            # (6. 人脸区域图片
            fe.facePart = fe.srcImg[top:top + height, left:left + width]
            # cv2.imwrite(todayPath + "\\" + str(fe.num) + ".jpg", fe.facePart)

            fe.facePartOnlySkin = SkinUtils.trimSkin(fe.facePart)
            # 画出人脸矩形
            cv2.rectangle(copy, fe.rectanglePoint[0],
                          fe.rectanglePoint[1], COLORDICT['white'], 3, cv2.LINE_AA)
            # (7. 添加画出线条的图像
            fe.drawImg = copy

            detectedFaces.append(fe)

        LogUtils.log("FaceLandMark", "detected finish! found:", len(detectedFaces))
        return detectedFaces


# 单例模式
faceDetection = __FaceDetect()


def _testImage():
    img = cv2.imread("../four_color_face_sample/black.png")
    f = faceDetection
    faces = f.faceDetectByImg(img)
    for face in faces:
        # resizedImg = imutils.resize(face.drawImg, width=800, height=800)
        resizedImg = face.drawImg
        cv2.imshow("face", resizedImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _testVideo():
    video_capture = cv2.VideoCapture(1)
    f = faceDetection
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if frame is not None:
            detected_img = f.faceDetectRealTime(frame, scale=5)
            cv2.imshow("face", detected_img)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    # _testImage()
    _testVideo()
