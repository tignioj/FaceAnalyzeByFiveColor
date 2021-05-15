import re
import sys
import time

import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import *
import imutils
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic.properties import QtWidgets

from utils.ImageUtils import ImgUtils
from utils.SkinTrimUtlis import SkinTrimUtils

from designer.SkinDetect import Ui_MainWindow
from utils import ImageUtils
from utils.LogUtils import LogUtils


class SkinDetectImplGUI(QMainWindow, Ui_MainWindow):
    __IMAGE_LABEL_SIZE = (800, 600)
    "显示图像区域大小"

    __VIDEO_RGB = 0
    __VIDEO_Lab = 1
    __VIDEO_HSV = 2
    __VIDEO_YCrCb = 3
    __VIDEO_Melt = 4
    "视频模式"

    def __init__(self):
        super(SkinDetectImplGUI, self).__init__()
        self.setupUi(self)
        self.CAM_NUM = 1
        self.videoCapture = cv2.VideoCapture(self.CAM_NUM)
        self.cameraTimerAfter = QTimer()
        self.cameraTimerBefore = QTimer()
        self.prev_before_frame_time = 0
        self.new_before_frame_time = 0
        self.prev_after_frame_time = 0
        self.new_after_frame_time = 0
        self.skinMode = self.__VIDEO_Melt
        self.initRange()
        self.initSlot()

    def initRange(self):
        self.min_melt = np.array([0, 0, 0], np.uint8)
        self.max_melt = np.array([255, 255, 255], np.uint8)

        self.min_rgb = SkinTrimUtils.min_rgb
        self.max_rgb = SkinTrimUtils.max_rgb

        self.min_Lab = SkinTrimUtils.min_Lab
        self.max_Lab = SkinTrimUtils.max_Lab

        self.min_HSV = SkinTrimUtils.min_HSV
        self.max_HSV = SkinTrimUtils.max_HSV

        self.min_YCrCb = SkinTrimUtils.min_YCrCb
        self.max_YCrCb = SkinTrimUtils.max_YCrCb

        self.currentMinRange = self.min_rgb
        self.currentMaxRange = self.max_rgb

    def releaseCamera(self):
        LogUtils.log("GUI", "尝试释放相机")
        if self.cameraTimerBefore.isActive():
            self.cameraTimerBefore.stop()
            LogUtils.log("GUI", "成功关闭BeforeFrame的计时器")
        if self.cameraTimerAfter.isActive():
            self.cameraTimerAfter.stop()
            LogUtils.log("GUI", "成功关闭AfterFrame的计时器")

        self.videoCapture.release()
        LogUtils.log("GUI", "成功关闭摄像头")

        self.label_before.clear()
        self.label_after.clear()

    def closeCamera(self):
        LogUtils.log("GUI", "尝试关闭相机")
        self.appendInfo("尝试关闭摄像头..")
        if not self.videoCapture.isOpened():
            self.appendError("你没有打开摄像头!")
            return
        else:
            self.releaseCamera()
            self.appendInfo("关闭成功!")
            # self.label_ShowCamera.setPixmap(QtGui.QPixmap("../images/process1.png"))
        self.appendInfo("已关闭摄像头!")

    @staticmethod
    def changeFrameByLableSizeKeepRatio(frame, fixW, fixH):
        # 如果视频的宽：高 > 显示区域的宽：高，说明应该以视频的宽度作为基准，计算出新的高度
        return imutils.resize(frame, width=fixW, height=fixH)

    def maskOptionChange(self):
        ks = self.horizontalSlider_kernelSize.value()
        it = self.horizontalSlider_iterations.value()
        s = "ks:" + str(ks) + ", its:" + str(it)
        self.lineEdit_option.setText(s)

    def initSlot(self):
        self.pushButtonOpenCamera.clicked.connect(self.openCamera)
        self.pushButtonpushButtonCloseCamera.clicked.connect(self.closeCamera)

        self.horizontalSlider_kernelSize.setMaximum(50)
        self.horizontalSlider_kernelSize.setMinimum(0)
        self.horizontalSlider_kernelSize.valueChanged.connect(self.maskOptionChange)

        self.horizontalSlider_iterations.setMaximum(20)
        self.horizontalSlider_iterations.setMinimum(0)
        self.horizontalSlider_iterations.valueChanged.connect(self.maskOptionChange)

        self.radioButton_melt.toggled.connect(self.radioButton_modeChange)
        self.radioButton_melt.skinMode = self.__VIDEO_Melt
        self.radioButton_melt.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_melt.maxRange = np.array([255, 255, 255], np.uint8)
        self.radioButton_melt.currentMinRangeValue = self.min_melt
        self.radioButton_melt.currentMaxRangeValue = self.max_melt
        self.radioButton_melt.labelA = "A"
        self.radioButton_melt.labelB = "B"
        self.radioButton_melt.labelC = "C"

        self.radioButton_HSV.toggled.connect(self.radioButton_modeChange)
        self.radioButton_HSV.skinMode = self.__VIDEO_HSV
        self.radioButton_HSV.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_HSV.maxRange = np.array([180, 255, 255], np.uint8)
        self.radioButton_HSV.currentMinRangeValue = self.min_HSV
        self.radioButton_HSV.currentMaxRangeValue = self.max_HSV
        self.radioButton_HSV.labelA = "H"
        self.radioButton_HSV.labelB = "S"
        self.radioButton_HSV.labelC = "V"

        self.radioButton_RGB.toggled.connect(self.radioButton_modeChange)
        self.radioButton_RGB.skinMode = self.__VIDEO_RGB
        self.radioButton_RGB.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_RGB.maxRange = np.array([255, 255, 255], np.uint8)
        self.radioButton_RGB.currentMinRangeValue = self.min_rgb
        self.radioButton_RGB.currentMaxRangeValue = self.max_rgb
        self.radioButton_RGB.labelA = "R"
        self.radioButton_RGB.labelB = "G"
        self.radioButton_RGB.labelC = "B"

        self.radioButton_Lab.toggled.connect(self.radioButton_modeChange)
        self.radioButton_Lab.skinMode = self.__VIDEO_Lab
        self.radioButton_Lab.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_Lab.maxRange = np.array([255, 255, 255], np.uint8)
        self.radioButton_Lab.currentMinRangeValue = self.min_Lab
        self.radioButton_Lab.currentMaxRangeValue = self.max_Lab
        self.radioButton_Lab.labelA = "L"
        self.radioButton_Lab.labelB = "a"
        self.radioButton_Lab.labelC = "b"

        self.radioButton_YCrCb.toggled.connect(self.radioButton_modeChange)
        self.radioButton_YCrCb.skinMode = self.__VIDEO_YCrCb
        self.radioButton_YCrCb.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_YCrCb.maxRange = np.array([255, 255, 255], np.uint8)
        self.radioButton_YCrCb.currentMinRangeValue = self.min_YCrCb
        self.radioButton_YCrCb.currentMaxRangeValue = self.max_YCrCb
        self.radioButton_YCrCb.labelA = "Y"
        self.radioButton_YCrCb.labelB = "Cr"
        self.radioButton_YCrCb.labelC = "Cb"

        self.horizontalSlider_min1.valueChanged.connect(self.sliderRangeChangeMin)
        self.horizontalSlider_min1.index = 0
        self.horizontalSlider_min2.valueChanged.connect(self.sliderRangeChangeMin)
        self.horizontalSlider_min2.index = 1
        self.horizontalSlider_min3.valueChanged.connect(self.sliderRangeChangeMin)
        self.horizontalSlider_min3.index = 2

        self.horizontalSlider_max1.valueChanged.connect(self.sliderRangeChangeMax)
        self.horizontalSlider_max1.index = 0
        self.horizontalSlider_max2.valueChanged.connect(self.sliderRangeChangeMax)
        self.horizontalSlider_max2.index = 1
        self.horizontalSlider_max3.valueChanged.connect(self.sliderRangeChangeMax)
        self.horizontalSlider_max3.index = 2

        self.cameraTimerBefore.timeout.connect(self.showCameraBefore)  # 每次倒计时溢出，调用函数刷新页面
        self.cameraTimerAfter.timeout.connect(self.showCameraAfter)  # 每次倒计时溢出，调用函数刷新页面

    def sliderRangeChangeMin(self):
        slider = self.sender()
        self.currentMinRange[slider.index] = slider.value()
        self.lineEdit_minRange.setText(str(self.currentMinRange))

    def sliderRangeChangeMax(self):
        slider = self.sender()
        self.currentMaxRange[slider.index] = slider.value()
        self.lineEdit_maxRange.setText(str(self.currentMaxRange))

    def radioButton_modeChange(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.skinMode = radioButton.skinMode
            self.label_a.setText(
                radioButton.labelA + ":" + str(radioButton.minRange[0]) + "-" + str(radioButton.maxRange[0]))
            self.label_b.setText(
                radioButton.labelB + ":" + str(radioButton.minRange[1]) + "-" + str(radioButton.maxRange[1]))
            self.label_c.setText(
                radioButton.labelC + ":" + str(radioButton.minRange[2]) + "-" + str(radioButton.maxRange[2]))

            # self.label_b.setText(radioButton.labelB + ":" + str(radioButton.))
            # self.label_c.setText(radioButton.labelC)

            self.horizontalSlider_min1.setMinimum(radioButton.minRange[0])
            self.horizontalSlider_min1.setMaximum(radioButton.maxRange[0])

            self.horizontalSlider_min2.setMinimum(radioButton.minRange[1])
            self.horizontalSlider_min2.setMaximum(radioButton.maxRange[1])

            self.horizontalSlider_min3.setMinimum(radioButton.minRange[2])
            self.horizontalSlider_min3.setMaximum(radioButton.maxRange[2])

            self.horizontalSlider_max1.setMinimum(radioButton.minRange[0])
            self.horizontalSlider_max1.setMaximum(radioButton.maxRange[0])

            self.horizontalSlider_max2.setMinimum(radioButton.minRange[1])
            self.horizontalSlider_max2.setMaximum(radioButton.maxRange[1])

            self.horizontalSlider_max3.setMinimum(radioButton.minRange[2])
            self.horizontalSlider_max3.setMaximum(radioButton.maxRange[2])

            self.currentMinRange = radioButton.currentMinRangeValue
            self.currentMaxRange = radioButton.currentMaxRangeValue

            self.lineEdit_minRange.setText(str(radioButton.currentMinRangeValue))
            self.lineEdit_maxRange.setText(str(radioButton.currentMaxRangeValue))

            self.horizontalSlider_min1.setValue(self.currentMinRange[0])
            self.horizontalSlider_min2.setValue(self.currentMinRange[1])
            self.horizontalSlider_min3.setValue(self.currentMinRange[2])

            self.horizontalSlider_max1.setValue(self.currentMaxRange[0])
            self.horizontalSlider_max2.setValue(self.currentMaxRange[1])
            self.horizontalSlider_max3.setValue(self.currentMaxRange[2])

    def showCameraBefore(self):
        currentFrame = self.readCamera()
        currentFrame = SkinDetectImplGUI.changeFrameByLableSizeKeepRatio(currentFrame, self.__IMAGE_LABEL_SIZE[0],
                                                                         self.__IMAGE_LABEL_SIZE[1])
        self.new_before_frame_time = time.time()
        fps = 1 / (self.new_before_frame_time - self.prev_before_frame_time)
        self.prev_before_frame_time = self.new_before_frame_time
        s = str(currentFrame.shape[1]) + "x" + str(currentFrame.shape[0]) + ",FPS:" + re.sub(r'(.*\.\d{2}).*', r'\1',
                                                                                             str(fps))
        cv2.putText(currentFrame, s, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
        # 将图像转换为pixmap
        showImage = ImgUtils.nparrayToQPixMap(currentFrame)
        self.label_before.setPixmap(showImage)

    def readCamera(self):
        flag, currentFrame = self.videoCapture.read()
        if not flag:
            if self.cameraTimer.isActive():
                self.cameraTimer.stop()
            self.appendError("相机未能成功读取到数据")
            self.releaseCamera()
        else:
            return currentFrame

    def showCameraAfter(self):
        currentFrame = self.readCamera()
        skinMaskRange, skinMaskAfter = np.ones([self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1]]), np.ones(
            [self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1]])
        currentFrame = SkinDetectImplGUI.changeFrameByLableSizeKeepRatio(currentFrame, self.__IMAGE_LABEL_SIZE[0],
                                                                         self.__IMAGE_LABEL_SIZE[1])

        ks, it = self.horizontalSlider_kernelSize.value(), self.horizontalSlider_iterations.value()
        if self.skinMode == self.__VIDEO_Melt:
            currentFrame, skinMaskRange, skinMaskAfter = SkinTrimUtils.rgb_hsv_ycbcr(currentFrame, ks, it)
        elif self.skinMode == self.__VIDEO_RGB:
            currentFrame ,skinMaskRange, skinMaskAfter = SkinTrimUtils.rgb(currentFrame, self.currentMinRange, self.currentMaxRange, ks, it)
        elif self.skinMode == self.__VIDEO_HSV:
            currentFrame, skinMaskRange, skinMaskAfter = SkinTrimUtils.hsv(currentFrame, self.currentMinRange,
                                                                           self.currentMaxRange, ks, it)
        elif self.skinMode == self.__VIDEO_Lab:
            currentFrame, skinMaskRange, skinMaskAfter = SkinTrimUtils.Lab(currentFrame, self.currentMinRange,
                                                                           self.currentMaxRange, ks, it)
        elif self.skinMode == self.__VIDEO_YCrCb:
            currentFrame, skinMaskRange, skinMaskAfter = SkinTrimUtils.YCrCb(currentFrame, self.currentMinRange,
                                                                             self.currentMaxRange, ks, it)

        # 计算FPS
        self.new_before_frame_time = time.time()
        fps = 1 / (self.new_before_frame_time - self.prev_before_frame_time)
        self.prev_before_frame_time = self.new_before_frame_time
        s = str(currentFrame.shape[1]) + "x" + str(currentFrame.shape[0]) + ",FPS:" + re.sub(r'(.*\.\d{2}).*', r'\1',
                                                                                             str(fps))
        cv2.putText(currentFrame, s, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # 将图像转换为pixmap
        showImage = ImgUtils.nparrayToQPixMap(currentFrame)
        showSkinMaskRange = ImgUtils.nparrayToQPixMap(skinMaskRange)
        showSkinMaskAfter = ImgUtils.nparrayToQPixMap(skinMaskAfter)
        self.label_after.setPixmap(showImage)
        self.label_mask1.setPixmap(showSkinMaskRange)
        self.label_mask2.setPixmap(showSkinMaskAfter)

    def appendError(self, text):
        "追加错误信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        originalText = self.textEditLog.toHtml()
        self.textEditLog.setHtml("<span style='color:red'>[" + s + "]<br/>" + text + "</span>" + originalText)

    def showError(self, text):
        "显示错误信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        self.textEditLog.setHtml("<span style='color:red'>[" + s + "]<br/>" + text + "</span>")

    def appendInfo(self, text):
        "显示常规信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        originalText = self.textEditLog.toHtml()
        self.textEditLog.setHtml("<span style='color:black'>[" + s + "]<br/>" + text + "</span>" + originalText)

    def showInfo(self, text):
        "显示常规信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        self.textEditLog.clear()
        self.textEditLog.toHtml()
        self.textEditLog.setHtml("<span style='color:black'>[" + s + "]<br/>" + text + "</span>")

    def openCamera(self):  # 打开摄像头，启动倒计时
        LogUtils.log("GUI-openCamera", "准备打开摄像头, 更新UI的计时器状态：", self.cameraTimerBefore.isActive())
        self.appendInfo("尝试打开摄像头")
        if not self.cameraTimerBefore.isActive():
            flag = self.videoCapture.open(self.CAM_NUM)
            if not flag:
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查摄像头与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
                self.appendError("摄像头未能成功打开！")
            else:
                # self.cameraTimerBefore.start(20)
                self.cameraTimerAfter.start(20)
                self.cameraTimerBefore.start(20)
                self.appendInfo("摄像头成功开启！")
                LogUtils.log("GUI-openCamera", "开启更新UI的计时器：", self.cameraTimerBefore.isActive())
        else:
            self.showError("摄像头已经开启了！")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = SkinDetectImplGUI()
    myWin.setWindowTitle("肤色检测")
    myWin.show()
    sys.exit(app.exec_())
