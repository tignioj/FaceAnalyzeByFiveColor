import numpy as np
from entity.ROIEntity import ROIEntity
from core.const_var import *
import cv2
from utils import SkinUtils

"""
负责封装和处理ROI
"""


class ROIService:
    def analyzeROI(self):
        pass

    def getROIsReport(self):
        pass

    def _getRegionFromCenter(self, centerPoint, size):
        width = size[0]
        height = size[1]
        """
        从中心点出发, 根据长和宽获得一个矩形坐标
        :param centerPoint:
        :param width:
        :param height:
        :return:
        """
        pts = []
        width = int(width / 2)
        height = int(height / 2)
        pts.append((centerPoint[0] - width, centerPoint[1] + height))  # 左上角
        pts.append((centerPoint[0] + width, centerPoint[1] + height))  # 右上角
        pts.append((centerPoint[0] + width, centerPoint[1] - height))  # 右下角
        pts.append((centerPoint[0] - width, centerPoint[1] - height))  # 左下角
        return pts

    def _getFaceWH(self, face):
        return face.bottom() - face.top(), face.right() - face.left()

    # 获取ROI
    def getROI(self, name, shape, image, face, scale=1):
        pts = []
        center_point = ()
        faceW, faceH = self._getFaceWH(face)
        queSize = (faceW * 0.13, faceH * 0.13)
        tingSize = (faceW * 0.3, faceH * 0.13)
        quanSize = (faceW * 0.13, faceH * 0.13)
        jiaSize = (faceW * 0.13, faceH * 0.13)
        mingTangSize = (faceW * 0.11, faceH * 0.11)
        keSize = (faceW * 0.24, faceH * 0.09)

        # cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['rand'], 2)
        if name == KEY_ting:
            # 对于庭，我们只获取阙上面50个像素的位置为中心，大小为W:100, H50
            # 获取阙的中心，并且上移动x像素
            que_coord = shape[21:23].mean(axis=0)
            center_point = (int(que_coord[0]), int(que_coord[1] - faceH * 0.15))
            pts = self._getRegionFromCenter(center_point, tingSize)
        elif name == KEY_que:  # 阙
            # 对于阙，我们需要获取21和22点的坐标中点，再截取50*50的像素
            coord = shape[21:23].mean(axis=0)
            center_point = (int(coord[0]), int(coord[1]))
            pts = self._getRegionFromCenter(center_point, queSize)
            # 构造一个矩形
        elif name == KEY_quan_left:
            # 对于左边的颧骨(注意是指实际左边)，需要获取45和14的均值
            center_point = np.array([shape[45], shape[14]]).mean(axis=0)
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = self._getRegionFromCenter(center_point, quanSize)
            pass
        elif name == KEY_quan_right:
            # 对于右边边的颧骨(注意是指实际右边)，需要获取36和2的均值
            center_point = np.array([shape[36], shape[2]]).mean(axis=0)
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = self._getRegionFromCenter(center_point, quanSize)
            pass
        elif name == KEY_ming_tang:
            # 明堂：30点即可
            center_point = shape[30]
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = self._getRegionFromCenter(center_point, mingTangSize)
        elif name == KEY_jia_left:  # 脸颊左边
            # 左边脸颊：y取值为鼻头30的y，x取左边眼睛46的x
            x = shape[46][0]
            y = shape[30][1]
            center_point = (int(x), int(y))
            pts = self._getRegionFromCenter(center_point, jiaSize)
        elif name == KEY_jia_right:  # 脸颊右边
            # 左边脸颊：y取值为鼻头30的y，x取右边眼睛41的x
            x = shape[41][0]
            y = shape[30][1]
            center_point = (int(x), int(y))
            pts = self._getRegionFromCenter(center_point, jiaSize)
        elif name == KEY_chun:  # 唇部
            # 唇部直接获取范围 48:68, 这是固定的范围
            center_point = np.array(shape[48:68]).mean(axis=0)
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = shape[48:68]
        elif name == KEY_ke:  # 颏
            # 对于颏(ke 四声), 获取57和8的中点
            center_point = np.array([shape[57], shape[8]]).mean(axis=0)
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = self._getRegionFromCenter(center_point, keSize)
        else:
            print("unknown name:", name)

        # extract the ROI of the face region as a separate image
        # 提取点的矩形
        roiEntity = ROIEntity()
        # (1. ROI名字
        roiEntity.roiName = name
        roiEntity.nameCN = FACIAL_LANDMARKS_NAME_DICT[name]
        roiEntity.centerPoint = scale * np.array(center_point)
        roiEntity.roiRectanglePoints = scale * np.array(pts)
        (x, y, w, h) = cv2.boundingRect(roiEntity.roiRectanglePoints)
        roi = image[y:y + h, x:x + w]
        roiEntity.img = roi
        roiEntity.imgOnlySkin = SkinUtils.SkinUtils.trimSkin(roi)

        return roiEntity
