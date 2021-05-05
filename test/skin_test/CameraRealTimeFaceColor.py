import re

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import *
from core.FaceLandMark import faceDetection
from PyQt5.QtGui import *
from PyQt5 import *
import time
from PyQt5.QtWidgets import *
import sys
import cv2

'''
使用摄像机调用Dlib的HOG检测算法和特征检测算法
'''


class CameraDemo(QMainWindow):
    def __init__(self):
        super(CameraDemo, self).__init__()
        self.scale = 8
        "缩放倍数，建议是2"
        self.resize(1920, 1080)
        self.setWindowTitle('CameraDemo')
        self.initUI()

    def faceDetect(self, img):
        # return self.fd.faceDetectRealTime(img, self.scale)
        return self.fd.faceDetectRealTime(img, self.scale)
        # return faceDetectFR(img)

    def initUI(self):
        self.fd = faceDetection
        self.btnOpen = QPushButton('开启摄像头')
        self.btnOpen.clicked.connect(self.openCamera)
        self.CAM_NUM = 1
        self.video_capture = cv2.VideoCapture(self.CAM_NUM)
        self.cameraTimer = QTimer()
        self.cameraTimer.timeout.connect(self.updateLabel)  # 每次倒计时溢出，调用函数刷新页面
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.btnClose = QPushButton('关闭摄像头')
        self.btnClose.clicked.connect(self.closeCamera)

        self.label = QLabel("标签")

        vBox = QVBoxLayout()
        vBox.addWidget(self.label)
        vBox.addWidget(self.btnOpen)
        vBox.addWidget(self.btnClose)

        mainFrame = QWidget()
        mainFrame.setLayout(vBox)
        self.setCentralWidget(mainFrame)

        hBox = QHBoxLayout()
        buttonRGB  = QHBoxLayout(hBox)
        buttonHSV = QHBoxLayout(hBox)
        buttonYCrCb = QHBoxLayout(hBox)
        buttonH = QHBoxLayout(hBox)

        # 计算帧速率用
        self.count = 0
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.MODE_RGB = 0
        self.MODE_YCbCr = 0
        self.MODE_RGB = 0
        self.MODE_RGB = 0

        self.openCamera()

    def updateLabel(self):
        self.count += 1
        # print('更新次数：', self.count)


        if self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret is None:
                print('没有检测到视频')
                self.cameraTimer.stop()

            # Resize frame of video to 1/4 size for faster face recognition processing
            # small_frame = cv2.resize(frame, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
            detected_img = self.faceDetect(frame)
            # detected_img = fra
            # detected_img = frame
            self.new_frame_time = time.time()
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            s = str(frame.shape[1]) + "x" + str(frame.shape[0]) + ",FPS:" + re.sub(r'(.*\.\d{2}).*', r'\1', str(fps))
            cv2.putText(detected_img, s, (7, 30), self.font, 1, (100, 255, 0), 3, cv2.LINE_AA)
            self.label.setPixmap(self.nparrayToQPixMap(detected_img))

    def nparrayToQPixMap(self, ShowVideo):
        """
        OpenCV的BGR的数组转换成QPixMap
        :param ShowVideo:
        :return:
        """
        ShowVideo = cv2.cvtColor(ShowVideo, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(ShowVideo.data, ShowVideo.shape[1], ShowVideo.shape[0], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(showImage)

    def closeCamera(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
            self.label.clear()
            self.label.setText("已关闭摄像头")
            self.count = 0
            self.cameraTimer.stop()

    def openCamera(self):  # 打开摄像头，启动倒计时
        # self.cameraTimer = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 后一个参数用来消一个奇怪的warn
        # self.cameraTimer = cv2.VideoCapture(1)  # 后一个参数用来消一个奇怪的warn
        if not self.cameraTimer.isActive():
            flag = self.video_capture.open(self.CAM_NUM)
            if not flag:
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查摄像头与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.cameraTimer.start(20)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = CameraDemo()
    main.show()
    sys.exit(app.exec_())
