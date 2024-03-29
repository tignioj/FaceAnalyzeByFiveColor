import cv2
from PyQt5.QtCore import *
import imutils
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from utils.ImageUtils import ImgUtils
import sys
from designer.ReportPage import Ui_MainWindow as ReportPage
from core.const_var import *
from utils.LogUtils import LogUtils


class ReportPageImpl(QMainWindow, ReportPage):
    reportPageSignal = pyqtSignal()

    def __init__(self):
        super(ReportPageImpl, self).__init__()
        self.setupUi(self)

        # width of window
        self.w_width = 1800

        # height of window
        self.w_height = 900

        self.resultText = ""

        q = QDesktopWidget()
        sc = q.screen()
        sc.width()
        sc.height()
        self.setGeometry(sc.width() / 2 - self.w_width / 2, (sc.height() / 2 - self.w_height / 2), self.w_width,
                         self.w_height)

        self.pushButton_back_home.clicked.connect(self.closeSelf)

        self.initDict()


    def initDict(self):
        self.label_user_dict = {
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
        self.label_rgb_dict = {
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
        self.label_hsv_dict = {
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
        self.label_ycrcb_dict = {
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
        self.label_lab_dict = {
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
        self.label_result_dict = {
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
        self.label_statistics = {
            KEY_ting: self.label_ting_statistics,
            KEY_que: self.label_que_statistics,
            KEY_ming_tang: self.label_ming_tang_statistics,
            KEY_quan_left: self.label_quan_statistics,
            KEY_quan_right: self.label_quan_statistics,
            KEY_chun: self.label_chun_statistics,
            KEY_jia_left: self.label_jia_statistics,
            KEY_jia_right: self.label_jia_statistics,
            KEY_ke: self.label_ke_statistics
        }

    def closeSelf(self):
        self.close()

    def loadReports(self, reports):
        for r in reports:
            sz = self.label_face_drawed.size()
            prevDrawImg = imutils.resize(r.drawImg, width=sz.width(), height=sz.height())
            pixMapPrevDrawImg = ImgUtils.nparrayToQPixMap(prevDrawImg)
            pixMapSrcDrawImg = ImgUtils.nparrayToQPixMap(r.drawImg)
            self.label_face_drawed.setPixmap(pixMapPrevDrawImg)
            self.label_face_drawed.largePixMap = pixMapSrcDrawImg
            self.label_username.setText(str(r.username))
            self.label_sex.setText(str(r.gender))

            self.loadROIImage(self.label_user_dict, r.roiDict)
            self.loadRoiHist(self.label_rgb_dict, r.roiRGBDict)
            self.loadRoiHist(self.label_ycrcb_dict, r.roiYCrCbDict)
            self.loadRoiHist(self.label_hsv_dict, r.roiHSVDict)
            self.loadRoiHist(self.label_lab_dict, r.roiLabDict)
            self.loadRoiHist(self.label_statistics, r.roiHistograms)
            self.loadResult(self.label_result_dict, r.roiColorResults)

            self.textEdit_result.setText(self.resultText)

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

    def loadResult(self, labelDict, roiDict):
        """
        Load result,
        :param labelDict:
        :param roiDict:
        :return:
        """
        for (name, result) in roiDict.items():
            # labelDict[name].setText(result['text'])
            resultText = result['text']
            resultColor = result['color']
            diffImg = result['img']

            predictImg = diffImg['predict']
            predictImgTrim = diffImg['predictTrim']
            sampleImg = diffImg['sample']
            sampleImgTrim = diffImg['sampleTrim']

            # p, s = ImgUtils.keepSameShape(predictImg, sampleImg)
            p = cv2.resize(predictImg, (sampleImg.shape[1], sampleImg.shape[0]))
            pt = cv2.resize(predictImgTrim, (sampleImgTrim.shape[1], sampleImgTrim.shape[0]))

            LogUtils.log("ReportPageImpl", "组合图片中..")
            p = ImgUtils.putTextCN(p, "预测:" + ImgUtils.COLOR_SAMPLE_CN_NAME_BY_KEY[resultColor], (1, 1), COLORDICT['blue'])
            s = ImgUtils.putTextCN(sampleImg, "样本", (1, 1), COLORDICT['green'])

            pt = ImgUtils.putTextCN(pt, "提纯后", (1, 1), COLORDICT['blue'])
            st = ImgUtils.putTextCN(sampleImgTrim, "提纯后", (1, 1), COLORDICT['green'])

            combineImgRow1 = np.hstack([p, pt])
            combineImgRow2 = np.hstack([s, st])
            # combineImgRow1 = cv2.resize(combineImgRow1, (combineImgRow2.shape[1], combineImgRow2.shape[0]))
            combineImg = np.vstack([combineImgRow1, combineImgRow2])
            LogUtils.log("ReportPageImpl", "组合图片完毕..")

            LogUtils.log("ReportPageImpl", "正在加载'" + FACIAL_LANDMARKS_NAME_DICT[name] + "'的结果, 对比图大小" + str(combineImg.shape))

            self.resultText += resultText + "\n"

            labelDict[name].setToolTip(resultText)
            self.setCvImgToLabel(labelDict[name], combineImg)

    def loadROIImage(self, labelDict, roiDict):
        """
        封装了ROI对象
        :param labelDict:
        :param roiDict:
        :return:
        """
        for (name, roi) in roiDict.items():
            LogUtils.log("ReportPageImpl", "正在加载" + FACIAL_LANDMARKS_NAME_DICT[name] + "的图片")
            self.setCvImgToLabel(labelDict[name], roi.img)

    def loadRoiHist(self, labelDict, roiDict):
        """
        封装ROI的时候只是封装了图像，而不是ROIEntity对象
        :param labelDict:
        :param roiDict:
        :return:
        """
        for (name, roi) in roiDict.items():
            LogUtils.log("ReportPageImpl", "正在加载'" + FACIAL_LANDMARKS_NAME_DICT[name] + "的直方图")
            self.setCvImgToLabel(labelDict[name], roi, 200, 200)

    @staticmethod
    def cvshow(img, title="0x123"):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    def setCvImgToLabel(self, label, cvImg, width=None, height=None):
        label.largePixMap = ImgUtils.nparrayToQPixMap(cvImg)
        label.setPixmap(ImgUtils.nparrayToQPixMap(cvImg, width, height))
