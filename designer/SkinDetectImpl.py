import json
import re
import sys
import time

import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import *
import imutils
from PyQt5.QtGui import *
from core.const_var import *
from PyQt5.QtWidgets import *
from PyQt5.uic.properties import QtWidgets

from core.const_var import SKIN_PARAM_PATH, OUTPUT_PATH
from utils.ImageUtils import ImgUtils
from utils.SkinTrimUtlis import SkinTrimUtils

from designer.SkinDetect import Ui_MainWindow
from utils import ImageUtils
from utils.LogUtils import LogUtils


class SkinDetectImplGUI(QMainWindow, Ui_MainWindow):
    __IMAGE_LABEL_SIZE = (800, 400)
    "显示图像区域大小"

    "视频模式"

    "信号"
    saveSignal = pyqtSignal(dict)

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
        self.skinMode = VIDEO_Melt
        self.initUI()
        self.initRange()
        self.initSlot()

    def setBeforeImg(self, image):
        if image is not None:
            qpixMap = ImgUtils.nparrayToQPixMap(image, self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1])
            largePixMap = ImgUtils.nparrayToQPixMap(image)
            self.label_show_before.setLargePixMap(largePixMap)
            self.label_show_before.setSrcImg(image)
            self.releaseCamera()
            self.label_show_before.setPixmap(qpixMap)
            self.analyze()

    def initUI(self):
        def setWH(label):
            label.setFixedWidth(self.__IMAGE_LABEL_SIZE[0])
            label.setFixedHeight(self.__IMAGE_LABEL_SIZE[1])

        setWH(self.label_show_before)
        setWH(self.label_show_after)
        setWH(self.label_show_mask1)
        setWH(self.label_show_mask2)

    def initRange(self):
        self.min_melt = [0, 0, 0]
        self.max_melt = [255, 255, 255]

        self.min_rgb = SkinTrimUtils.min_rgb.copy()
        self.max_rgb = SkinTrimUtils.max_rgb.copy()

        self.min_Lab = SkinTrimUtils.min_Lab.copy()
        self.max_Lab = SkinTrimUtils.max_Lab.copy()

        self.min_HSV = SkinTrimUtils.min_HSV.copy()
        self.max_HSV = SkinTrimUtils.max_HSV.copy()

        self.min_YCrCb = SkinTrimUtils.min_YCrCb.copy()
        self.max_YCrCb = SkinTrimUtils.max_YCrCb.copy()

        self.paramDict = {
            VIDEO_Melt: {
                KEY_THRESHOLD_RANGE_MIN: self.min_melt,
                KEY_THRESHOLD_RANGE_MAX: self.max_melt,
                KEY_MIN_RANGE: [0, 0, 0],
                KEY_MAX_RANGE: [255, 255, 255],
                KEY_KernelSize: 7,
                KEY_Iterations: 2
            },
            VIDEO_RGB: {
                KEY_THRESHOLD_RANGE_MIN: self.min_rgb,
                KEY_THRESHOLD_RANGE_MAX: self.max_rgb,
                KEY_MIN_RANGE: [0, 0, 0],
                KEY_MAX_RANGE: [255, 255, 255],
                KEY_KernelSize: 7,
                KEY_Iterations: 2
            },
            VIDEO_Lab: {
                KEY_THRESHOLD_RANGE_MIN: self.min_Lab,
                KEY_THRESHOLD_RANGE_MAX: self.max_Lab,
                KEY_MIN_RANGE: [0, 1, 1],
                KEY_MAX_RANGE: [255, 255, 255],
                KEY_KernelSize: 7,
                KEY_Iterations: 2

            },
            VIDEO_HSV: {
                KEY_THRESHOLD_RANGE_MIN: self.min_HSV,
                KEY_THRESHOLD_RANGE_MAX: self.max_HSV,
                KEY_MIN_RANGE: [0, 0, 0],
                KEY_MAX_RANGE: [180, 255, 255],
                KEY_KernelSize: 7,
                KEY_Iterations: 2

            },
            VIDEO_YCrCb: {
                KEY_THRESHOLD_RANGE_MIN: self.min_YCrCb,
                KEY_THRESHOLD_RANGE_MAX: self.max_YCrCb,
                KEY_MIN_RANGE: [0, 0, 0],
                KEY_MAX_RANGE: [255, 255, 255],
                KEY_KernelSize: 7,
                KEY_Iterations: 2
            }
        }

        self.initSliderValue()

    def initSliderValue(self):
        self.horizontalSlider_iterations.setValue(2)
        self.horizontalSlider_kernelSize.setValue(7)
        self.radioButton_melt.click()

    def changeParamDict(self, colorMode, key, value=None, index=None):
        if index is not None:
            self.paramDict[colorMode][key][index] = value
        else:
            self.paramDict[colorMode][key] = value

        self.analyze()

    def getParamDict(self):
        return self.paramDict

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

        self.label_show_before.clear()
        self.label_show_after.clear()

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
        sender = self.sender()
        ks = self.horizontalSlider_kernelSize.value()
        it = self.horizontalSlider_iterations.value()
        self.changeParamDict(sender.colorMode, KEY_KernelSize, ks)
        self.changeParamDict(sender.colorMode, KEY_Iterations, it)
        s = "ks:" + str(ks) + ", its:" + str(it)

        self.lineEdit_option.setText(s)

    def saveParam(self):
        saveDict = {
            'currentMode': self.skinMode,
            'paramDict': self.paramDict
        }
        with open(SKIN_PARAM_PATH, 'w') as outfile:
            json.dump(saveDict, outfile)
        self.appendInfo("保存成功！")

        path = OUTPUT_PATH + "\\trimSkin.jpg"
        if self.label_show_after.srcImg is not None:
            cv2.imwrite(path, self.label_show_after.srcImg)
            saveDict['img'] = self.label_show_after.srcImg
        else:
            saveDict['img'] = None

        self.saveSignal.emit(saveDict)

    def loadParam(self):
        with open(SKIN_PARAM_PATH, 'r') as infile:
            j = json.load(infile)

        self.paramDict = j['paramDict']
        self.skinMode = j['currentMode']

        self.appendInfo("读取成功！")
        m = self.skinMode
        if m == VIDEO_Lab:
            self.radioButton_Lab.click()
        elif m == VIDEO_HSV:
            self.radioButton_HSV.click()
        elif m == VIDEO_RGB:
            self.radioButton_RGB.click()
        elif m == VIDEO_YCrCb:
            self.radioButton_YCrCb.click()
        else:
            self.radioButton_melt.click()

    def resetParam(self):
        self.initRange()

    def openFile(self):
        self.appendInfo("准备打开文件")
        curPath = QDir.currentPath()
        imagePath, imgType = QFileDialog.getOpenFileName(self, "打开图片", curPath,
                                                         " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        LogUtils.log("SkinDetect", "准备打开文件" + imagePath)
        if imgType == "" or imagePath == "":
            self.appendInfo("取消选择文件")
            return
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        qpixMap = ImgUtils.nparrayToQPixMap(image, self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1])
        largePixMap = ImgUtils.nparrayToQPixMap(image)
        self.label_show_before.setLargePixMap(largePixMap)
        self.label_show_before.setSrcImg(image)
        self.releaseCamera()
        self.label_show_before.setPixmap(qpixMap)
        self.analyze()

    def analyze(self):
        self.showCameraAfter()

    def initSlot(self):
        # 按钮区
        self.pushButtonOpenCamera.clicked.connect(self.openCamera)
        self.pushButtonpushButtonCloseCamera.clicked.connect(self.closeCamera)
        self.pushButton_save.clicked.connect(self.saveParam)
        self.pushButton_load.clicked.connect(self.loadParam)
        self.pushButton_reset.clicked.connect(self.resetParam)
        self.pushButton_import.clicked.connect(self.openFile)

        # ===================== 设置 Radio Button ==========================
        self.radioButton_melt.colorMode = VIDEO_Melt
        self.radioButton_melt.toggled.connect(self.radioButton_modeChange)
        self.radioButton_melt.skinMode = VIDEO_Melt
        self.radioButton_melt.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_melt.maxRange = np.array([255, 255, 255], np.uint8)
        self.radioButton_melt.currentMinRangeValue = self.min_melt
        self.radioButton_melt.currentMaxRangeValue = self.max_melt
        self.radioButton_melt.labelA = "A"
        self.radioButton_melt.labelB = "B"
        self.radioButton_melt.labelC = "C"

        self.radioButton_HSV.colorMode = VIDEO_HSV
        self.radioButton_HSV.toggled.connect(self.radioButton_modeChange)
        self.radioButton_HSV.skinMode = VIDEO_HSV
        self.radioButton_HSV.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_HSV.maxRange = np.array([180, 255, 255], np.uint8)
        self.radioButton_HSV.currentMinRangeValue = self.min_HSV
        self.radioButton_HSV.currentMaxRangeValue = self.max_HSV
        self.radioButton_HSV.labelA = "H"
        self.radioButton_HSV.labelB = "S"
        self.radioButton_HSV.labelC = "V"

        self.radioButton_RGB.colorMode = VIDEO_RGB
        self.radioButton_RGB.toggled.connect(self.radioButton_modeChange)
        self.radioButton_RGB.skinMode = VIDEO_RGB
        self.radioButton_RGB.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_RGB.maxRange = np.array([255, 255, 255], np.uint8)
        self.radioButton_RGB.currentMinRangeValue = self.min_rgb
        self.radioButton_RGB.currentMaxRangeValue = self.max_rgb
        self.radioButton_RGB.labelA = "R"
        self.radioButton_RGB.labelB = "G"
        self.radioButton_RGB.labelC = "B"

        self.radioButton_Lab.colorMode = VIDEO_Lab
        self.radioButton_Lab.toggled.connect(self.radioButton_modeChange)
        self.radioButton_Lab.skinMode = VIDEO_Lab
        self.radioButton_Lab.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_Lab.maxRange = np.array([255, 255, 255], np.uint8)
        self.radioButton_Lab.currentMinRangeValue = self.min_Lab
        self.radioButton_Lab.currentMaxRangeValue = self.max_Lab
        self.radioButton_Lab.labelA = "L"
        self.radioButton_Lab.labelB = "a"
        self.radioButton_Lab.labelC = "b"

        self.radioButton_YCrCb.colorMode = VIDEO_YCrCb
        self.radioButton_YCrCb.toggled.connect(self.radioButton_modeChange)
        self.radioButton_YCrCb.skinMode = VIDEO_YCrCb
        self.radioButton_YCrCb.minRange = np.array([0, 0, 0], np.uint8)
        self.radioButton_YCrCb.maxRange = np.array([255, 255, 255], np.uint8)
        self.radioButton_YCrCb.currentMinRangeValue = self.min_YCrCb
        self.radioButton_YCrCb.currentMaxRangeValue = self.max_YCrCb
        self.radioButton_YCrCb.labelA = "Y"
        self.radioButton_YCrCb.labelB = "Cr"
        self.radioButton_YCrCb.labelC = "Cb"

        # ======================= 设置Slider ==================================
        self.sliderThresholdLock = False
        "当为True的时候，不允许设置阈值，为了防止更改颜色模式最值的时候出发sliderValueChange事件"

        # 选项
        self.horizontalSlider_kernelSize.setMaximum(50)
        self.horizontalSlider_kernelSize.setMinimum(0)
        self.horizontalSlider_kernelSize.valueChanged.connect(self.maskOptionChange)
        self.horizontalSlider_kernelSize.colorMode = None

        self.horizontalSlider_iterations.setMaximum(20)
        self.horizontalSlider_iterations.setMinimum(0)
        self.horizontalSlider_iterations.valueChanged.connect(self.maskOptionChange)
        self.horizontalSlider_iterations.colorMode = None

        # 三通道参数
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

        self.setSliderMode(VIDEO_Melt)

        # =================== 设置定时器 ===========================
        self.cameraTimerBefore.timeout.connect(self.showCameraBefore)  # 每次倒计时溢出，调用函数刷新页面
        self.cameraTimerAfter.timeout.connect(self.showCameraAfter)  # 每次倒计时溢出，调用函数刷新页面

    def setSliderMode(self, mode):
        self.horizontalSlider_min1.colorMode = mode
        self.horizontalSlider_min2.colorMode = mode
        self.horizontalSlider_min3.colorMode = mode
        self.horizontalSlider_max1.colorMode = mode
        self.horizontalSlider_max2.colorMode = mode
        self.horizontalSlider_max3.colorMode = mode
        self.horizontalSlider_kernelSize.colorMode = mode
        self.horizontalSlider_iterations.colorMode = mode

    def sliderRangeChangeMin(self):
        slider = self.sender()
        if not self.sliderThresholdLock:
            self.changeParamDict(slider.colorMode, KEY_THRESHOLD_RANGE_MIN, index=slider.index,
                                 value=slider.value())
            self.lineEdit_minRange.setText(str(self.getParamDict()[slider.colorMode][KEY_THRESHOLD_RANGE_MIN]))
            print("sliderChangeMin:", slider.colorMode, slider.value())

    def sliderRangeChangeMax(self):
        if not self.sliderThresholdLock:
            slider = self.sender()
            print("sliderChangeMax:", slider.colorMode, slider.value())
            self.changeParamDict(slider.colorMode, KEY_THRESHOLD_RANGE_MAX, index=slider.index,
                                 value=slider.value())
            self.lineEdit_maxRange.setText(str(self.getParamDict()[slider.colorMode][KEY_THRESHOLD_RANGE_MAX]))

    def radioButton_modeChange(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.skinMode = radioButton.skinMode
            self.setSliderMode(self.skinMode)

            self.sliderThresholdLock = True
            "当为True的时候，不允许设置阈值，为了防止更改颜色模式最值的时候出发sliderValueChange事件"

            param = self.getParamDict()[self.skinMode]
            print(self.skinMode, param)

            self.label_a.setText(
                radioButton.labelA + ":" + str(param[KEY_MIN_RANGE][0]) + "-" + str(
                    param[KEY_MAX_RANGE][0]))
            self.label_b.setText(
                radioButton.labelB + ":" + str(param[KEY_MIN_RANGE][1]) + "-" + str(
                    param[KEY_MAX_RANGE][1]))
            self.label_c.setText(
                radioButton.labelC + ":" + str(param[KEY_MIN_RANGE][2]) + "-" + str(
                    param[KEY_MAX_RANGE][2]))

            # self.label_show_b.setText(radioButton.labelB + ":" + str(radioButton.))
            # self.label_c.setText(radioButton.labelC)
            self.sliderThreadSold = True

            self.horizontalSlider_min1.setMinimum(param[KEY_MIN_RANGE][0])
            self.horizontalSlider_min1.setMaximum(param[KEY_MAX_RANGE][0])

            self.horizontalSlider_min2.setMinimum(param[KEY_MIN_RANGE][1])
            self.horizontalSlider_min2.setMaximum(param[KEY_MAX_RANGE][1])

            self.horizontalSlider_min3.setMinimum(param[KEY_MIN_RANGE][2])
            self.horizontalSlider_min3.setMaximum(param[KEY_MAX_RANGE][2])

            self.horizontalSlider_max1.setMinimum(param[KEY_MIN_RANGE][0])
            self.horizontalSlider_max1.setMaximum(param[KEY_MAX_RANGE][0])

            self.horizontalSlider_max2.setMinimum(param[KEY_MIN_RANGE][1])
            self.horizontalSlider_max2.setMaximum(param[KEY_MAX_RANGE][1])

            self.horizontalSlider_max3.setMinimum(param[KEY_MIN_RANGE][2])
            self.horizontalSlider_max3.setMaximum(param[KEY_MAX_RANGE][2])

            self.sliderThresholdLock = False

            # 这两个不需要设置范围，只需要设置值就可以，所以无需加锁
            self.horizontalSlider_kernelSize.setValue(param[KEY_KernelSize])
            self.horizontalSlider_iterations.setValue(param[KEY_Iterations])

            cmr = param[KEY_THRESHOLD_RANGE_MIN]
            cmx = param[KEY_THRESHOLD_RANGE_MAX]

            self.lineEdit_minRange.setText(str(cmr))
            self.lineEdit_maxRange.setText(str(cmx))

            self.horizontalSlider_min1.setValue(cmr[0])
            self.horizontalSlider_min2.setValue(cmr[1])
            self.horizontalSlider_min3.setValue(cmr[2])

            self.horizontalSlider_max1.setValue(cmx[0])
            self.horizontalSlider_max2.setValue(cmx[1])
            self.horizontalSlider_max3.setValue(cmx[2])

    def showCameraBefore(self):
        currentFrame = self.readCamera()
        currentFrame = SkinDetectImplGUI.changeFrameByLableSizeKeepRatio(currentFrame, self.__IMAGE_LABEL_SIZE[0],
                                                                         self.__IMAGE_LABEL_SIZE[1])
        self.label_show_before.srcImg = currentFrame

        self.new_before_frame_time = time.time()
        fps = 1 / (self.new_before_frame_time - self.prev_before_frame_time)
        self.prev_before_frame_time = self.new_before_frame_time
        s = str(currentFrame.shape[1]) + "x" + str(currentFrame.shape[0]) + ",FPS:" + re.sub(r'(.*\.\d{2}).*', r'\1',
                                                                                             str(fps))
        cv2.putText(currentFrame, s, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
        # 将图像转换为pixmap
        showImage = ImgUtils.nparrayToQPixMap(currentFrame, self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1])
        self.label_show_before.setPixmap(showImage)
        self.label_show_before.setLargePixMap(showImage)

    def readCamera(self):
        flag, currentFrame = self.videoCapture.read()
        if not flag:
            if self.cameraTimerBefore.isActive():
                self.cameraTimerBefore.stop()
            if self.cameraTimerAfter.isActive():
                self.cameraTimerAfter.stop()
            self.appendError("相机未能成功读取到数据")
            self.releaseCamera()
        else:
            return currentFrame

    def showCameraAfter(self):
        currentFrame = self.label_show_before.srcImg
        if currentFrame is None:
            currentFrame = self.readCamera()

        skinMaskRange, skinMaskAfter = np.ones([self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1]]), np.ones(
            [self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1]])
        currentFrame = SkinDetectImplGUI.changeFrameByLableSizeKeepRatio(currentFrame, self.__IMAGE_LABEL_SIZE[0],
                                                                         self.__IMAGE_LABEL_SIZE[1])

        if self.skinMode == VIDEO_Melt:
            currentFrame, skinMaskRange, skinMaskAfter = \
                SkinTrimUtils.rgb_hsv_ycbcr(currentFrame,
                                            kernelSize=self.getParamDict()[VIDEO_Melt][
                                                KEY_KernelSize],
                                            iteration=self.getParamDict()[VIDEO_Melt][
                                                KEY_Iterations]
                                            )
        elif self.skinMode == VIDEO_RGB:

            currentFrame, skinMaskRange, skinMaskAfter = \
                SkinTrimUtils.rgb(currentFrame,
                                  self.getParamDict()[VIDEO_RGB][KEY_THRESHOLD_RANGE_MIN],
                                  self.getParamDict()[VIDEO_RGB][KEY_THRESHOLD_RANGE_MAX],
                                  self.getParamDict()[VIDEO_RGB][KEY_KernelSize],
                                  self.getParamDict()[VIDEO_RGB][KEY_Iterations],
                                  )
        elif self.skinMode == VIDEO_HSV:
            currentFrame, skinMaskRange, skinMaskAfter = \
                SkinTrimUtils.hsv(currentFrame,
                                  self.getParamDict()[VIDEO_HSV][KEY_THRESHOLD_RANGE_MIN],
                                  self.getParamDict()[VIDEO_HSV][KEY_THRESHOLD_RANGE_MAX],
                                  self.getParamDict()[VIDEO_HSV][KEY_KernelSize],
                                  self.getParamDict()[VIDEO_HSV][KEY_Iterations],
                                  )
        elif self.skinMode == VIDEO_Lab:
            currentFrame, skinMaskRange, skinMaskAfter = \
                SkinTrimUtils.Lab(currentFrame,
                                  self.getParamDict()[VIDEO_Lab][KEY_THRESHOLD_RANGE_MIN],
                                  self.getParamDict()[VIDEO_Lab][KEY_THRESHOLD_RANGE_MAX],
                                  self.getParamDict()[VIDEO_Lab][KEY_KernelSize],
                                  self.getParamDict()[VIDEO_Lab][KEY_Iterations]
                                  )
        elif self.skinMode == VIDEO_YCrCb:
            currentFrame, skinMaskRange, skinMaskAfter = \
                SkinTrimUtils.YCrCb(currentFrame,
                                    self.getParamDict()[VIDEO_YCrCb][KEY_THRESHOLD_RANGE_MIN],
                                    self.getParamDict()[VIDEO_YCrCb][KEY_THRESHOLD_RANGE_MAX],
                                    self.getParamDict()[VIDEO_YCrCb][KEY_KernelSize],
                                    self.getParamDict()[VIDEO_YCrCb][KEY_Iterations],
                                    )

        trimedImg = currentFrame.copy()
        # 计算FPS
        self.new_before_frame_time = time.time()
        fps = 1 / (self.new_before_frame_time - self.prev_before_frame_time)
        self.prev_before_frame_time = self.new_before_frame_time
        s = str(currentFrame.shape[1]) + "x" + str(currentFrame.shape[0]) + ",FPS:" + re.sub(r'(.*\.\d{2}).*', r'\1',
                                                                                             str(fps))
        cv2.putText(currentFrame, s, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # 将图像转换为pixmap
        showImage = ImgUtils.nparrayToQPixMap(currentFrame, self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1])
        showSkinMaskRange = ImgUtils.nparrayToQPixMap(skinMaskRange, self.__IMAGE_LABEL_SIZE[0],
                                                      self.__IMAGE_LABEL_SIZE[1])
        showSkinMaskAfter = ImgUtils.nparrayToQPixMap(skinMaskAfter, self.__IMAGE_LABEL_SIZE[0],
                                                      self.__IMAGE_LABEL_SIZE[1])
        self.label_show_after.setPixmap(showImage)
        self.label_show_after.setSrcImg(trimedImg)
        self.label_show_after.setLargePixMap(ImgUtils.nparrayToQPixMap(currentFrame))

        self.label_show_mask1.setPixmap(showSkinMaskRange)
        self.label_show_mask1.setLargePixMap(ImgUtils.nparrayToQPixMap(skinMaskRange))

        self.label_show_mask2.setPixmap(showSkinMaskAfter)
        self.label_show_mask2.setLargePixMap(ImgUtils.nparrayToQPixMap(skinMaskAfter))

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
