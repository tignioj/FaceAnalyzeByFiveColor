import re
import sys
import time

import cv2
from PyQt5 import QtGui
from PyQt5.QtCore import *
import imutils
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic.properties import QtWidgets

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
        self.videoMode = self.__VIDEO_Melt
        self.initSlot()

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

    def initSlot(self):
        self.pushButtonOpenCamera.clicked.connect(self.openCamera)
        self.pushButtonpushButtonCloseCamera.clicked.connect(self.closeCamera)

        self.cameraTimerBefore.timeout.connect(self.showCameraBefore)  # 每次倒计时溢出，调用函数刷新页面
        self.cameraTimerAfter.timeout.connect(self.showCameraAfter)  # 每次倒计时溢出，调用函数刷新页面



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
        showImage = ImageUtils.nparrayToQPixMap(currentFrame)
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
        currentFrame = SkinDetectImplGUI.changeFrameByLableSizeKeepRatio(currentFrame, self.__IMAGE_LABEL_SIZE[0],
                                                                         self.__IMAGE_LABEL_SIZE[1])
        if self.videoMode == self.__VIDEO_Melt:
            currentFrame == SkinTrimUtils.Lab()
        elif self.videoMode == self.__VIDEO_RGB:
            pass
        elif self.videoMode == self.__VIDEO_HSV:
            pass
        elif self.videoMode == self.__VIDEO_Lab:
            pass
        elif self.videoMode == self.__VIDEO_YCrCb:
            pass

        # 计算FPS
        self.new_before_frame_time = time.time()
        fps = 1 / (self.new_before_frame_time - self.prev_before_frame_time)
        self.prev_before_frame_time = self.new_before_frame_time
        s = str(currentFrame.shape[1]) + "x" + str(currentFrame.shape[0]) + ",FPS:" + re.sub(r'(.*\.\d{2}).*', r'\1',
                                                                                             str(fps))
        cv2.putText(currentFrame, s, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # 将图像转换为pixmap
        showImage = ImageUtils.nparrayToQPixMap(currentFrame)
        self.label_after.setPixmap(showImage)

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
