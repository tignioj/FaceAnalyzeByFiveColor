import cv2
from PyQt5.QtCore import *
import imutils
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from utils.ImageUtils import nparrayToQPixMap, cvshow
import sys
from designer.ReportPage import Ui_MainWindow as ReportPage
from core.const_var import *


class ReportPageImpl(QMainWindow, ReportPage):
    reportPageSignal = pyqtSignal()

    def __init__(self):
        super(ReportPageImpl, self).__init__()
        self.setupUi(self)
        self.pushButton_back_home.clicked.connect(self.closeSelf)
        self.initDict()

    def initDict(self):
        self.label_ting_user_dict = {
            KEY_ting: self.label_ting_user,
            KEY_que: self.label_que_user,
            KEY_ming_tang: self.label_ming_tang_user,
            KEY_quan_left: self.label_quan_user,
            KEY_quan_right: self.label_quan_user,
            KEY_chun: self.label_chun_user,
            KEY_jia_left: self.label_jia_user,
            KEY_jia_right: self.label_jia_user,
            KEY_ke: self.label_ke_user
        }
        self.label_ting_rgb_dict = {
            KEY_ting: self.label_ting_rgb,
            KEY_que: self.label_que_rgb,
            KEY_ming_tang: self.label_ming_tang_rgb,
            KEY_quan_left: self.label_quan_rgb,
            KEY_quan_right: self.label_quan_rgb,
            KEY_chun: self.label_chun_rgb,
            KEY_jia_left: self.label_jia_rgb,
            KEY_jia_right: self.label_jia_rgb,
            KEY_ke: self.label_ke_rgb
        }
        self.label_ting_hsv_dict = {
            KEY_ting: self.label_ting_hsv,
            KEY_que: self.label_que_hsv,
            KEY_ming_tang: self.label_ming_tang_hsv,
            KEY_quan_left: self.label_quan_hsv,
            KEY_quan_right: self.label_quan_hsv,
            KEY_chun: self.label_chun_hsv,
            KEY_jia_left: self.label_jia_hsv,
            KEY_jia_right: self.label_jia_hsv,
            KEY_ke: self.label_ke_hsv
        }
        self.label_ting_ycrcb_dict = {
            KEY_ting: self.label_ting_ycrcb,
            KEY_que: self.label_que_ycrcb,
            KEY_ming_tang: self.label_ming_tang_ycrcb,
            KEY_quan_left: self.label_quan_ycrcb,
            KEY_quan_right: self.label_quan_ycrcb,
            KEY_chun: self.label_chun_ycrcb,
            KEY_jia_left: self.label_jia_ycrcb,
            KEY_jia_right: self.label_jia_ycrcb,
            KEY_ke: self.label_ke_ycrcb
        }
        self.label_ting_lab_dict = {
            KEY_ting: self.label_ting_lab,
            KEY_que: self.label_que_lab,
            KEY_ming_tang: self.label_ming_tang_lab,
            KEY_quan_left: self.label_quan_lab,
            KEY_quan_right: self.label_quan_lab,
            KEY_chun: self.label_chun_lab,
            KEY_jia_left: self.label_jia_lab,
            KEY_jia_right: self.label_jia_lab,
            KEY_ke: self.label_ke_lab
        }
        self.label_ting_result_dict = {
            KEY_ting: self.label_ting_result,
            KEY_que: self.label_que_result,
            KEY_ming_tang: self.label_ming_tang_result,
            KEY_quan_left: self.label_quan_result,
            KEY_quan_right: self.label_quan_result,
            KEY_chun: self.label_chun_result,
            KEY_jia_left: self.label_jia_result,
            KEY_jia_right: self.label_jia_result,
            KEY_ke: self.label_ke_result
        }

    def closeSelf(self):
        self.close()

    def loadReports(self, reports):
        for r in reports:
            sz = self.label_face_drawed.size()
            img = imutils.resize(r.drawImg, width=sz.width(), height=sz.height())
            self.label_face_drawed.setPixmap(nparrayToQPixMap(img))
            self.label_username.setText(str(r.username))
            self.label_sex.setText(str(r.gender))

            self.loadROIImage(self.label_ting_user_dict, r.roiDict)
            """
            拿到分割的ROI，将其加载到GUI中，
            步骤：1. 获取分割的ROIDict
            步骤：2. 获取GUI对应的labelDict
            步骤：3. 遍历ROIDict， 在拿到ROIDict的名称过程可以同时拿到对应的label
            步骤：4. 将对应的label设置pixMap
            
            """

            # RGB
            # self.label_ting_rgb
            # self.label_que_rgb
            # self.label_ming_tang_rgb
            # self.label_jia_rgb
            # self.label_chun_rgb
            # self.label_ke_rgb

        self.reportPageSignal.emit()

    def loadROIImage(self, labelDict, roiDict):
        for (name, roi) in roiDict.items():
            self.setCvImgToLabel(labelDict[name], roi.img)

    @staticmethod
    def cvshow(img, title="0x123"):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    def setCvImgToLabel(self, label, cvImg):
        label.setPixmap(nparrayToQPixMap(cvImg))
