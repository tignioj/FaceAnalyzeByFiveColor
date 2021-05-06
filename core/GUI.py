import re
import time
from utils.ImageUtils import *
from utils.LogUtils import LogUtils

import cv2
import sys
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from service.ReportService import ReportService
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QDir, pyqtSignal, QThread, QSize, QDateTime, QRunnable
from designer.ReportPageImpl import ReportPageImpl
# from GUIDesign import *
from designer.GUIDesigner import *
from core.faceHealthDetect import *
from FaceLandMark import faceDetection
from utils import ImageUtils
from utils.ImageUtils import nparrayToQPixMap


# from faceTest import

class BackendThread:
    class InnerThread(QThread):
        """
        匿名内部类线程
        """
        """定义一个信号"""
        signal = pyqtSignal(dict)

        def __init__(self,
                     invokeFun,  # 让线程执行的函数，通常是大量计算的函数
                     callbackFun,  # 回调函数
                     invokeParam={},  # 线程执行函数需要传的参数，需要传入字典
                     callbackParam={}  # 回调函数的参数，需要传入字典
                     ):
            # 一定要调用父类，不然这个线程可能执行不了
            super(BackendThread.InnerThread, self).__init__()
            self.callBackFun = callbackFun
            self.invokeParam = invokeParam
            self.callBackParam = callbackParam
            self.invokeFun = invokeFun

        def run(self):
            """ 线程调用start()之后就会调用这个方法 """

            # 绑定槽函数，也就是回调函数
            self.signal.connect(self.callBackFun)
            try:
                # 调用需要大量计算的函数
                res = self.invokeFun(self.invokeParam)
                # 执行回调函数
                # self.signal.emit({'res': res, 'param': self.callBackParam})
                self.signal.emit({'res': res, 'param': self.callBackParam})
            except Exception as err:
                # 有可能出现异常
                # self.signal.emit({'res': err, 'param': self.callBackParam})
                self.signal.emit({'res': err, 'param': self.callBackParam})

    # 需要执行大量计算的函数
    @staticmethod
    def faceDetectInBackground(paramMap):
        detectedFaces = faceDetect(paramMap['image'], 1, paramMap['name'], paramMap['gender'])
        reports = ReportService.generateReports(detectedFaces)
        LogUtils.log("GUI", 'report generate finished!', len(reports))
        return {'reports': reports}

    # 生成报告，有IO操作
    @staticmethod
    def generateReport(paramMap):
        report = paramMap['reports']
        faces = report.faces
        for face in faces:
            ReportService.wordCreate(face)
            report.wordCreate()
        return {'report', report}

    @staticmethod
    def fakeFunctionForUpdateUI(paramMap):
        pass


class MyWindow(QMainWindow, Ui_MainWindow):
    __IMAGE_LABEL_STATE_NONE = 0
    "表示显示板块没有图像"

    __IMAGE_LABEL_STATE_USING_CAMERA = 1
    "表示正在显示摄像头的画面"

    __IMAGE_LABEL_STATE_USING_FILE = 2
    "表示正在使用本地图像"

    "显示图像区域大小"
    __IMAGE_LABEL_SIZE = (800, 600)

    __VIDEO_MODE_NORMAL = 0  # ->图像输出模式
    "图像输出模式为普通"

    __VIDEO_MODE_FACE = 1
    "图像输出模式为脸部检测"

    __VIDEO_MODE_EDGE = 2
    "图像输出模式为边缘检测"

    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)

        self.faceDetector = faceDetection
        self.cameraTimer = QtCore.QTimer()
        # 摄像头
        self.CAM_NUM = 1
        self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE  # 0表示无图像，1表示开启摄像头读取图像，2表示打开图像文件

        # 信息区
        self.detectedFaces = None
        self.reportUtils = None
        self.gender = '男'

        # 图像区
        self.videoMode = MyWindow.__VIDEO_MODE_NORMAL  # 定义图像输出模式
        self.movie = QMovie("../images/face_scanning.gif")
        self.defaultImg = QtGui.QPixmap("../images/process1.png")
        self.img = None
        self.EdgeTractThrehold1 = 50
        self.EdgeTractThrehold2 = self.EdgeTractThrehold1 + 200
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.videoCapture = cv2.VideoCapture(self.CAM_NUM)

        # UI初始化
        self.setupUi(self)
        self.slot_init()

        # 其它页面
        self.reportPage = None

        # 数据初始化
        self.lineEdit_UserName.setText("张三")

    def slot_init(self):
        self.button_CaptureAnalyse.clicked.connect(self.analyze)
        self.button_SaveReport.clicked.connect(self.SaveReport)
        self.cameraTimer.timeout.connect(self.showCamera)  # 每次倒计时溢出，调用函数刷新页面
        self.actionOpenImage.triggered.connect(self.OpenImage)
        self.actionOpenCamera.triggered.connect(self.openCamera)
        self.actionCloseCamera.triggered.connect(self.closeCamera)
        self.actionReset.triggered.connect(self.WorkSpaceReset)
        self.horizontalSlider_EdgeTract.valueChanged.connect(self.SliderChangeValue)

        # 监听视频模式
        self.radioButton_NormalImage.toggled.connect(self.radioButtonVideoModeChange)
        self.radioButton_NormalImage.videoMode = self.__VIDEO_MODE_NORMAL
        self.radioButton_EdgeTract.toggled.connect(self.radioButtonVideoModeChange)
        self.radioButton_EdgeTract.videoMode = self.__VIDEO_MODE_EDGE
        self.radioButton_FaceTract.toggled.connect(self.radioButtonVideoModeChange)
        self.radioButton_FaceTract.videoMode = self.__VIDEO_MODE_FACE

        # 监听性别
        self.radioButton_Male.toggled.connect(self.radioButtonGenderChange)
        self.radioButton_Male.gender = '男'
        self.radioButton_Female.toggled.connect(self.radioButtonGenderChange)
        self.radioButton_Female.gender = '女'

    def radioButtonGenderChange(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.gender = radioButton.gender

    def radioButtonVideoModeChange(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.videoMode = radioButton.videoMode

    def openCamera(self):  # 打开摄像头，启动倒计时
        LogUtils.log("GUI-openCamera", "准备打开摄像头, 更新UI的计时器状态：", self.cameraTimer.isActive())
        if not self.cameraTimer.isActive():
            flag = self.videoCapture.open(self.CAM_NUM)
            if not flag:
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查摄像头与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.cameraTimer.start(20)
                LogUtils.log("GUI-openCamera", "开启更新UI的计时器：", self.cameraTimer.isActive())
                self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_USING_CAMERA
        else:
            self.showError("摄像头已经开启了！")

    def showCamera(self):
        flag, self.image = self.videoCapture.read()
        if not flag:
            self.cameraTimer.stop()
            self.showError("相机未能成功读取到数据")
            self.releaseCamera()
            self.Flag_Image = self.__IMAGE_LABEL_STATE_NONE

        currentFrame = self.image

        currentFrame = changeFrameByLableSizeKeepRatio(currentFrame, self.__IMAGE_LABEL_SIZE[0],
                                                       self.__IMAGE_LABEL_SIZE[1])
        if self.videoMode == self.__VIDEO_MODE_FACE:
            currentFrame = self.faceDetector.faceDetectRealTime(currentFrame, 4)
        elif self.videoMode == self.__VIDEO_MODE_EDGE:
            currentFrame = cv2.Canny(currentFrame, self.EdgeTractThrehold1, self.EdgeTractThrehold2)

        # 计算FPS
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        s = str(currentFrame.shape[1]) + "x" + str(currentFrame.shape[0]) + ",FPS:" + re.sub(r'(.*\.\d{2}).*', r'\1',
                                                                                             str(fps))

        cv2.putText(currentFrame, s, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # 将图像转换为pixmap
        showImage = ImageUtils.nparrayToQPixMap(currentFrame)
        self.label_ShowCamera.setPixmap(showImage)

    def releaseCamera(self):
        LogUtils.log("GUI", "尝试释放相机")
        if self.cameraTimer is not None:
            if self.cameraTimer.isActive():
                self.cameraTimer.stop()
                LogUtils.log("GUI", "你没有打开相机")

        if self.videoCapture is not None:
            if self.videoCapture.isOpened():
                self.videoCapture.release()
                LogUtils.log("GUI", "释放成功")

        self.label_ShowCamera.clear()

    def closeCamera(self):
        LogUtils.log("GUI", "尝试关闭相机")
        if self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_NONE:
            self.showError("你没有打开摄像头!")
            return
        else:
            self.releaseCamera()
            self.showError("关闭成功!")
            self.label_ShowCamera.setPixmap(QtGui.QPixmap("../images/process1.png"))
            self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE
        self.showInfo("已关闭摄像头!")

    def handleFaceDetectResult(self, paramMap):
        self.cameraTimer.stop()
        """
        后台线程分析完毕，执行这个方法
        :return:
        """
        res = paramMap['res']
        LogUtils.log("GUI", "拿到结果", res)

        self.showInfo("诊断完成!")
        if type(res) is FaceNotFoundException:
            self.showloadingGIF(False)
            img = changeFrameByLableSizeKeepRatio(res.expression, self.__IMAGE_LABEL_SIZE[0],
                                                  self.__IMAGE_LABEL_SIZE[1])
            self.label_ShowCamera.setPixmap(nparrayToQPixMap(img))
            self.showError("未能识别到面孔！请重置工作区再试试看。" + str(res.message))
            return

        reports = res['reports']
        r = reports[0]

        img = r.drawImg
        qimg = changeFrameByLableSizeKeepRatio(img, self.__IMAGE_LABEL_SIZE[0],
                                               self.__IMAGE_LABEL_SIZE[1])
        qimg = nparrayToQPixMap(qimg)

        self.label_ShowCamera.setPixmap(qimg)
        self.button_CaptureAnalyse.setEnabled(True)
        self.showloadingGIF(False)
        self.reports = reports
        if self.reportPage is None:
            self.reportPage = ReportPageImpl()
            self.reportPage.reportPageSignal.connect(self.reportPageLoaded)

        self.reportPage.loadReports(self.reports)
        self.reportPage.show()

    def reportPageLoaded(self):
        self.setEnableButton()

    def messageBoxWarning(self, msg):
        QtWidgets.QMessageBox.warning(self, 'warning', msg, buttons=QtWidgets.QMessageBox.Ok)
        self.button_CaptureAnalyse.setEnabled(True)

    def setEnableButton(self):
        self.showloadingGIF(False)
        self.button_CaptureAnalyse.setEnabled(True)

    def analyze(self):  # 要思考未打开摄像头时按下“拍照”的问题
        """
        面容分析gg
        :return:
        """
        LogUtils.log("log", "分析前预处理...")
        self.button_CaptureAnalyse.setEnabled(False)
        userName = self.lineEdit_UserName.text()

        if len(userName.strip()) == 0:
            self.messageBoxWarning("请输入姓名")
            return

        if self.Flag_Image == self.__IMAGE_LABEL_STATE_NONE:
            self.messageBoxWarning("无图像输入")
            return

        gender = ""
        if self.gender is None:
            gender = "男"
        else:
            gender = self.gender

        #  开始分析
        LogUtils.log("GUI", "当前模式为:", self.Flag_Image)
        if self.Flag_Image == self.__IMAGE_LABEL_STATE_USING_CAMERA:
            flag, self.image = self.videoCapture.read()
            self.closeCamera()
            try:
                self.showInfo("正在为您诊断，请耐心等待...")
                self.showloadingGIF(True)
                BackendThread.InnerThread(BackendThread.faceDetectInBackground, self.handleFaceDetectResult,
                                          {'image': self.image, 'name': userName, 'gender': gender}
                                          ).start()
            except FaceNotFoundException as err:
                self.button_CaptureAnalyse.setEnabled(True)
                self.showError(err.message)

        elif self.Flag_Image == self.__IMAGE_LABEL_STATE_USING_FILE:
            self.Flag_Image = self.__IMAGE_LABEL_STATE_NONE
            # faseDetect过程比较长，需要多线程执行, 加载过渡动画动画
            self.showInfo("正在为您诊断，请耐心等待...")
            BackendThread.InnerThread(BackendThread.faceDetectInBackground, self.handleFaceDetectResult,
                                      {'image': self.image, 'name': userName, 'gender': gender}
                                      ).start()
            self.showloadingGIF(True)
            #  开启后台线程检测

    def showError(self, text):
        "显示错误信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        self.textEdit_Report.setHtml("<span style='color:red'>[" + s + "]<br/>" + text + "</span>")

    def showInfo(self, text):
        "显示常规信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        # self.textEdit_Report.setHtml("<b style='color:black'></b>")
        self.textEdit_Report.clear()
        # self.textEdit_Report.setHtml("<b>[" + s + "]</b><p style='color:black'><pre>" + text + "</pre></p>")
        self.textEdit_Report.setPlainText("[" + s + "]" + "\n" + text)

    def showloadingGIF(self, isShow):
        self.cameraTimer.stop()

        if isShow:
            self.movie.setScaledSize(QSize(800, 600))
            self.label_ShowCamera.setMovie(self.movie)
            self.movie.start()
        else:
            self.movie.stop()

    def OpenImage(self):  # 打开已有文件
        curPath = QDir.currentPath()
        imagePath, imgType = QFileDialog.getOpenFileName(self, "打开图片", curPath,
                                                         " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        LogUtils.log("GUI", "openImage:" + imagePath)
        img = QtGui.QPixmap(imagePath).scaled(self.label_ShowCamera.width(), self.label_ShowCamera.height())
        self.image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        self.label_ShowCamera.setPixmap(img)
        self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_USING_FILE

    def WorkSpaceReset(self):
        self.releaseCamera()
        self.videoMode = self.__VIDEO_MODE_NORMAL
        self.Flag_Image = self.__IMAGE_LABEL_STATE_NONE
        self.reports = None
        # self.lineEdit_UserName.setText("")
        self.gender = '男'
        self.reportPage = None
        self.label_ShowCamera.clear()
        self.reportUtils = None
        self.radioButton_Male.click()
        self.radioButton_NormalImage.click()
        self.textEdit_Report.clear()
        self.button_CaptureAnalyse.setEnabled(True)
        self.showInfo("工作区重置成功！")
        self.label_ShowCamera.setPixmap(self.defaultImg)

    def SliderChangeValue(self):
        self.EdgeTractThrehold1 = self.horizontalSlider_EdgeTract.value()

    def SaveReport(self):
        if self.reports is None:
            self.showError("尚未生成报告,请先进行诊断")
            return

        self.showInfo("正在保存报告，请稍等...")
        self.button_SaveReport.setEnabled(False)
        BackendThread.InnerThread(BackendThread.generateReport, self.handleSaveReport,
                                  {'reports': self.reports}
                                  ).start()

    def handleSaveReport(self, paramMap):
        self.button_SaveReport.setEnabled(True)
        self.showInfo("保存成功！")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.setWindowTitle("人脸像素统计与分析软件")
    myWin.show()
    sys.exit(app.exec_())
