import json
import random
import re
import time

from core.BackGroundThread import BackendThread
from designer.SkinDetectImpl import SkinDetectImplGUI
from utils.ImageUtils import ImgUtils
import cv2
import sys
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from service.ReportService import ReportService
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QDir, pyqtSignal, QThread, QSize, QDateTime, QRunnable, QTimer
from designer.ReportPageImpl import ReportPageImpl
# from GUIDesign import *
from designer.GUIDesigner import *
from core.faceHealthDetect import *
from FaceLandMark import faceDetection
from utils.LogUtils import LogUtils
from utils.SkinUtils import SkinUtils


class MainGUI(QMainWindow, Ui_MainWindow):
    __IMAGE_LABEL_STATE_NONE = 0
    "表示显示板块没有图像"
    progressSignalForNotFreeze = pyqtSignal()
    "为了表示并没有停止GUI的Signal，不断更新进度条"

    __IMAGE_LABEL_STATE_USING_CAMERA = 1
    "表示正在显示摄像头的画面"

    __IMAGE_LABEL_STATE_USING_FILE = 2
    "表示正在使用本地图像"

    __IMAGE_LABEL_STATE_USING_SKINTRIMED = 3
    "表示正在使用皮肤提取后的图像"

    "显示图像区域大小"
    __IMAGE_LABEL_SIZE = (1200, 800)

    __VIDEO_MODE_NORMAL = 0  # ->图像输出模式
    "图像输出模式为普通"

    __VIDEO_MODE_FACE = 1
    "图像输出模式为脸部检测"

    __VIDEO_MODE_SKIN = 2
    "图像输出模式为肤色检测"

    EDIT_TEXT_TYPE_UPDATE = 0
    EDIT_TEXT_TYPE_APPEND = 1
    EDIT_TEXT_TYPE_CLEAN = 2
    EDIT_TEXT_TYPE_ERROR = 3

    def __init__(self, parent=None):
        super(MainGUI, self).__init__(parent)

        self.faceDetector = faceDetection
        self.cameraTimer = QtCore.QTimer()
        self.loadingTimer = QTimer()
        self.timerLoadingPrevTime = 0
        self.timerLoadingNextTime = 0

        # 摄像头
        self.CAMERA_NUMBER = 1
        self.labelImageState = MainGUI.__IMAGE_LABEL_STATE_NONE  # 0表示无图像，1表示开启摄像头读取图像，2表示打开图像文件

        # 信息区
        self.detectedFaces = None
        self.reportUtils = None
        self.gender = '男'
        self.reports = None

        with open(SKIN_PARAM_PATH, 'r') as paramStr:
            self.skinParamDict = json.load(paramStr)

        # 图像区
        self.videoMode = MainGUI.__VIDEO_MODE_NORMAL  # 定义图像输出模式
        self.movie_loading = QMovie("../images/face_scanning.gif")
        self.preocessImagePath = "../images/process2.png"
        self.image = None
        self.skinImage = None
        self.prevFrameTime = 0
        self.newFrameTime = 0
        # self.videoCapture = cv2.VideoCapture(self.CAMERA_NUMBER, cv2.CAP_DSHOW)
        self.videoCapture = cv2.VideoCapture(self.CAMERA_NUMBER)

        # UI初始化
        self.setupUi(self)
        self.initSlot()
        self.initUI()

        # 其它页面
        self.reportPage = None
        self.skinParamPage = None

        # 数据初始化
        self.lineEdit_userName.setText("张三")

    def initUI(self):
        self.label_showCamera.setFixedWidth(self.__IMAGE_LABEL_SIZE[0])
        self.label_showCamera.setFixedHeight(self.__IMAGE_LABEL_SIZE[1])

    def handleReportText(self, param):
        text = param['text']
        type = param['type']

        try:
            progress = param['progress']
            if progress >= 100:
                progress = 100
            self.progressBar.setValue(progress)
        except:
            pass

        if type == self.EDIT_TEXT_TYPE_APPEND:
            self.appendReportText(text)
        elif type == self.EDIT_TEXT_TYPE_UPDATE:
            self.updateReportText(text)
        elif type == self.EDIT_TEXT_TYPE_CLEAN:
            self.textEdit_result.clear()
        elif type == self.EDIT_TEXT_TYPE_ERROR:
            self.updateReportForError(text)

    def updateReportForError(self, text):
        self.appendReportText(text)

    def appendReportText(self, text):
        self.appendInfo(text)

    def updateReportText(self, text):
        self.showInfo(text)

    def fakeProgress(self):
        v = self.progressBar.value()
        randomNumber = 2
        self.timerLoadingNextTime = time.time()
        if self.timerLoadingNextTime - self.timerLoadingPrevTime > randomNumber + 1:
            self.timerLoadingPrevTime = self.timerLoadingNextTime
            if v <= 99:
                # time.sleep(rnnum)
                randomNumber = random.randint(1, 3)
                v += randomNumber
                if v > 99: v = 99
                self.progressBar.setValue(v)

    def initSlot(self):
        self.button_analyze.clicked.connect(self.analyze)
        # self.button_seeReport.clicked.connect(self.SaveReport)
        self.button_seeReport.clicked.connect(self.openReportPage)
        self.pushButton_skinParamPage.clicked.connect(self.openSkinParamPage)
        self.cameraTimer.timeout.connect(self.showCamera)  # 每次倒计时溢出，调用函数刷新页面
        self.loadingTimer.timeout.connect(self.fakeProgress)

        self.actionOpenImage.triggered.connect(self.openImage)
        self.actionOpenCamera.triggered.connect(self.openCamera)
        self.actionCloseCamera.triggered.connect(self.closeCamera)
        self.actionReset.triggered.connect(self.workSpaceReset)
        self.horizontalSlider_Speed.setMaximum(12)
        self.horizontalSlider_Speed.setMinimum(5)
        self.horizontalSlider_Speed.setValue(8)

        # 监听视频模式
        self.radioButton_NormalImage.toggled.connect(self.radioButtonVideoModeChange)
        self.radioButton_NormalImage.skinMode = self.__VIDEO_MODE_NORMAL
        self.radioButton_SkinDetect.toggled.connect(self.radioButtonVideoModeChange)
        self.radioButton_SkinDetect.skinMode = self.__VIDEO_MODE_SKIN
        self.radioButton_FaceTract.toggled.connect(self.radioButtonVideoModeChange)
        self.radioButton_FaceTract.skinMode = self.__VIDEO_MODE_FACE

        # 监听性别
        self.radioButton_male.toggled.connect(self.radioButtonGenderChange)
        self.radioButton_male.gender = '男'
        self.radioButton_female.toggled.connect(self.radioButtonGenderChange)
        self.radioButton_female.gender = '女'

    def openSkinParamPage(self):
        if self.skinParamPage is None:
            self.skinParamPage = SkinDetectImplGUI()
            self.skinParamPage.setBeforeImg(self.image)
            self.skinParamPage.saveSignal.connect(self.skinParamSaved)

        if self.image is not None:
            self.radioButton_SkinDetect.click()
            self.skinParamPage.setBeforeImg(self.image)
            self.skinParamPage.analyze()

        self.appendInfo("打开肤色参数设置界面")
        self.releaseCamera()
        self.skinParamPage.show()

    def skinParamSaved(self, param):
        self.skinParamDict = param
        image = param['img']
        # 清空图像
        param['img'] = None
        self.appendInfo("已加载皮肤参数" + str(param))
        if image is not None:
            self.appendInfo("检测到返回的肤色提取照片, 已经加载到Label中, 大小:" + str(image.shape))
            self.labelImageState = self.__IMAGE_LABEL_STATE_USING_SKINTRIMED
            self.radioButton_SkinDetect.click()
            self.skinImage = image
            self.setNumpyArrayImgToImageShowLabel(image)
        # else:
        #     prevImg = ImgUtils.nparrayToQPixMap(self.image, self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1])
        #     self.setPixMapToImageShowLabel(prevImg)

    def openReportPage(self):
        if self.reports is None:
            self.appendError("尚未有报告！请先产生报告后再点击此按钮")
        else:
            if self.reportPage is None:
                self.reportPage = ReportPageImpl()
                self.reportPage.reportPageSignal.connect(self.reportPageLoaded)
                self.reportPage.loadReports(self.reports)
            self.appendInfo("打开人脸像素统计分析界面")
            self.reportPage.show()

    def radioButtonGenderChange(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.gender = radioButton.gender

    def radioButtonVideoModeChange(self):
        radioButton = self.sender()
        self.horizontalSlider_Speed.setEnabled(False)
        self.pushButton_skinParamPage.setEnabled(False)
        if radioButton.isChecked():
            self.videoMode = radioButton.skinMode
            if self.videoMode == self.__VIDEO_MODE_NORMAL:
                self.appendInfo("选择了普通模式")
                self.setNumpyArrayImgToImageShowLabel(self.image)
            elif self.videoMode == self.__VIDEO_MODE_FACE:
                self.appendInfo("选择了人脸捕捉模式")
                img = faceDetection.faceDetectForOneImage(self.image)
                if self.videoCapture.isOpened() and self.cameraTimer.isActive(): self.horizontalSlider_Speed.setEnabled(
                    True)
                self.setNumpyArrayImgToImageShowLabel(img)
            elif self.videoMode == self.__VIDEO_MODE_SKIN:
                self.appendInfo("选择了肤色分割模式")
                self.labelImageState = self.__IMAGE_LABEL_STATE_USING_SKINTRIMED
                self.pushButton_skinParamPage.setEnabled(True)
                self.skinImage = SkinUtils.trimSkinRealTime(self.image, self.skinParamDict)
                self.setNumpyArrayImgToImageShowLabel(self.skinImage)

    def openCamera(self):  # 打开摄像头，启动倒计时
        LogUtils.log("GUI-openCamera", "准备打开摄像头, 更新UI的计时器状态：", self.cameraTimer.isActive())
        self.appendInfo("尝试打开摄像头")
        if not self.cameraTimer.isActive():
            flag = self.videoCapture.open(self.CAMERA_NUMBER)
            if not flag:
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查摄像头与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
                self.appendError("摄像头未能成功打开！")
            else:
                self.cameraTimer.start(20)
                self.appendInfo("摄像头成功开启！")
                LogUtils.log("GUI-openCamera", "开启更新UI的计时器：", self.cameraTimer.isActive())
                self.labelImageState = MainGUI.__IMAGE_LABEL_STATE_USING_CAMERA
        else:
            self.showError("摄像头已经开启了！")

    def showCamera(self):
        flag, self.image = self.videoCapture.read()
        if not flag:
            if self.cameraTimer.isActive():
                self.cameraTimer.stop()
            self.appendError("相机未能成功读取到数据")
            self.releaseCamera()
            self.labelImageState = self.__IMAGE_LABEL_STATE_NONE

        currentFrame = self.image
        currentFrame = ImgUtils.changeFrameByLableSizeKeepRatio(currentFrame, self.__IMAGE_LABEL_SIZE[0],
                                                                self.__IMAGE_LABEL_SIZE[1])

        if self.videoMode == self.__VIDEO_MODE_FACE:
            sp = self.horizontalSlider_Speed.value()
            currentFrame = self.faceDetector.faceDetectRealTime(currentFrame, sp)
        elif self.videoMode == self.__VIDEO_MODE_SKIN:
            # currentFrame = cv2.Canny(currentFrame, self.EdgeTractThrehold1, self.EdgeTractThrehold2)
            sp = self.horizontalSlider_Speed.value()
            currentFrame = self.faceDetector.skinDetect(currentFrame, sp, self.skinParamDict)

        # 计算FPS
        self.newFrameTime = time.time()
        fps = 1 / (self.newFrameTime - self.prevFrameTime)
        self.prevFrameTime = self.newFrameTime
        s = str(currentFrame.shape[1]) + "x" + str(currentFrame.shape[0]) + ",FPS:" + re.sub(r'(.*\.\d{2}).*', r'\1',
                                                                                             str(fps))
        cv2.putText(currentFrame, s, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # 将图像转换为pixmap
        self.setNumpyArrayImgToImageShowLabel(currentFrame)

    def setNumpyArrayImgToImageShowLabel(self, nparrayImg):
        if nparrayImg is None:
            return
        showImage = ImgUtils.nparrayToQPixMap(nparrayImg, self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1])
        self.label_showCamera.setPixmap(showImage)

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

        # self.label_showCamera.clear()

    def closeCamera(self):
        LogUtils.log("GUI", "尝试关闭相机")
        self.appendInfo("尝试关闭摄像头..")
        if self.labelImageState == MainGUI.__IMAGE_LABEL_STATE_NONE:
            self.appendError("你没有打开摄像头!")
            return
        else:
            self.releaseCamera()
            self.appendInfo("关闭成功!")
            self.loadProcessImg()
            self.labelImageState = MainGUI.__IMAGE_LABEL_STATE_NONE
        self.appendInfo("已关闭摄像头!")

    def loadProcessImg(self):
        """
        加载原始流程图片
        :return:
        """
        self.label_showCamera.setPixmap(QtGui.QPixmap(self.preocessImagePath))

    def handleFaceDetectResult(self, paramMap):
        self.cameraTimer.stop()
        """
        后台线程分析完毕，执行这个方法
        :return:
        """
        res = paramMap['res']
        LogUtils.log("GUI", "拿到结果", res)

        self.appendInfo("诊断完成!")
        if type(res) is FaceNotFoundException:
            self.showloadingGIF(False)
            self.setNumpyArrayImgToImageShowLabel(res.expression)
            self.showError("未能识别到面孔！请重置工作区再试试看。" + str(res.message))
            return

        reports = res['reports']
        r = reports[0]

        img = r.drawImg
        # qimg = changeFrameByLableSizeKeepRatio(img, self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1])

        self.setNumpyArrayImgToImageShowLabel(img)
        self.buttonEnable(True)
        self.showloadingGIF(False)
        self.reports = reports
        self.reportPage = None
        self.openReportPage()

    def reportPageLoaded(self):
        self.setEnableButton()

    def messageBoxWarning(self, msg):
        QtWidgets.QMessageBox.warning(self, 'warning', msg, buttons=QtWidgets.QMessageBox.Ok)
        self.buttonEnable(True)

    def setEnableButton(self):
        self.showloadingGIF(False)
        self.buttonEnable(True)

    def analyze(self):
        """
        面容分析gg
        :return:
        """
        if self.reports is not None:
            self.openReportPage()

        LogUtils.log("log", "分析前预处理...")
        self.buttonEnable(False)
        userName = self.lineEdit_userName.text()

        if len(userName.strip()) == 0:
            self.messageBoxWarning("请输入姓名")
            return

        if self.labelImageState == self.__IMAGE_LABEL_STATE_NONE:
            self.messageBoxWarning("无图像输入")
            return

        if self.gender is None:
            gender = "男"
        else:
            gender = self.gender

        #  开始分析
        LogUtils.log("GUI", "当前模式为:", self.labelImageState)
        if self.labelImageState == self.__IMAGE_LABEL_STATE_USING_CAMERA:
            flag, self.image = self.videoCapture.read()
            LogUtils.log("GUI", "获得一帧图片", self.image.shape)
            self.closeCamera()
            try:
                LogUtils.log("GUI", "正在诊断...")
                self.showInfo("正在为您诊断，请耐心等待...")
                self.showloadingGIF(True)
                BackendThread.InnerThread(self, BackendThread.faceDetectInBackground, self.handleFaceDetectResult,
                                          {'image': self.image, 'name': userName, 'gender': gender},
                                          progressFun=self.handleReportText
                                          ).start()
                LogUtils.log("GUI", "后台已经发起请求", userName + "," + gender)
            except FaceNotFoundException as err:
                LogUtils.error("GUI", "没有找到人脸！", err)
                self.buttonEnable(True)
                self.showError(err.message)
            except Exception as e:
                LogUtils.error("GUI", "未知错误:", e)

        elif self.labelImageState == self.__IMAGE_LABEL_STATE_USING_FILE:
            LogUtils.log("GUI", "使用图片检测中...", self.image.shape)

            # faseDetect过程比较长，需要多线程执行, 加载过渡动画动画
            self.showInfo("正在为您诊断，请耐心等待...")
            self.showloadingGIF(True)
            LogUtils.log("GUI", "发起后台线程...", self.image.shape)
            BackendThread.InnerThread(self, BackendThread.faceDetectInBackground, self.handleFaceDetectResult,
                                      {'image': self.image, 'name': userName, 'gender': gender},
                                      progressFun=self.handleReportText
                                      ).start()

            LogUtils.log("GUI", "后台已经发起请求", userName + "," + gender)
            #  开启后台线程检测
        elif self.labelImageState == self.__IMAGE_LABEL_STATE_USING_SKINTRIMED:
            LogUtils.log("GUI", "使用提纯后的图片图片检测中...", self.skinImage.shape)

            # faseDetect过程比较长，需要多线程执行, 加载过渡动画动画
            self.showInfo("正在为您诊断，请耐心等待...")
            self.showloadingGIF(True)
            LogUtils.log("GUI", "发起后台线程...", self.skinImage.shape)
            BackendThread.InnerThread(self, BackendThread.faceDetectInBackground, self.handleFaceDetectResult,
                                      {'image': self.skinImage, 'name': userName, 'gender': gender},
                                      progressFun=self.handleReportText
                                      ).start()

            LogUtils.log("GUI", "后台已经发起请求", userName + "," + gender)
            #  开启后台线程检测

    def appendError(self, text):
        "追加错误信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        originalText = self.textEdit_Report.toHtml()
        self.textEdit_Report.setHtml("<span style='color:red'>[" + s + "]<br/>" + text + "</span>" + originalText)

    def showError(self, text):
        "显示错误信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        self.textEdit_Report.setHtml("<span style='color:red'>[" + s + "]<br/>" + text + "</span>")

    def appendInfo(self, text):
        "显示常规信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        originalText = self.textEdit_Report.toHtml()
        self.textEdit_Report.setHtml("<span style='color:black'>[" + s + "]<br/>" + text + "</span>" + originalText)

    def showInfo(self, text):
        "显示常规信息"
        d = QDateTime.currentDateTime()
        s = d.toString("yyyy-MM-dd hh:mm:ss")
        self.textEdit_Report.clear()
        self.textEdit_Report.toHtml()
        self.textEdit_Report.setHtml("<span style='color:black'>[" + s + "]<br/>" + text + "</span>")

    def showloadingGIF(self, isShow):
        if self.cameraTimer.isActive():
            self.cameraTimer.stop()
        if isShow:
            self.loadingTimer.start()
            self.movie_loading.setScaledSize(QSize(self.__IMAGE_LABEL_SIZE[0], self.__IMAGE_LABEL_SIZE[1]))
            self.label_showCamera.setMovie(self.movie_loading)
            self.movie_loading.start()
        else:
            self.loadingTimer.stop()
            self.movie_loading.stop()

    def openImage(self):  # 打开已有文件
        self.appendInfo("准备打开文件")
        curPath = QDir.currentPath()
        imagePath, imgType = QFileDialog.getOpenFileName(self, "打开图片", curPath,
                                                         " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        LogUtils.log("GUI", "准备打开文件" + imagePath)
        if imgType == "" or imagePath == "":
            self.appendInfo("取消选择文件")
            return
        # img = QtGui.QPixmap(imagePath).scaled(self.label_showCamera.width(), self.label_showCamera.height())
        self.image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        self.labelImageState = MainGUI.__IMAGE_LABEL_STATE_USING_FILE
        self.setNumpyArrayImgToImageShowLabel(self.image)
        self.appendInfo("成功加载文件:" + imagePath)

    def buttonEnable(self, b):
        self.button_analyze.setEnabled(b)
        self.actionOpenImage.setEnabled(b)
        self.actionOpenCamera.setEnabled(b)
        self.actionCloseCamera.setEnabled(b)
        self.actionReset.setEnabled(b)
        self.button_seeReport.setEnabled(b)
        if self.videoMode == self.__VIDEO_MODE_SKIN and b:
            self.pushButton_skinParamPage.setEnabled(b)

    def workSpaceReset(self):
        self.progressBar.setValue(0)
        self.videoMode = self.__VIDEO_MODE_NORMAL
        self.labelImageState = self.__IMAGE_LABEL_STATE_NONE
        self.releaseCamera()
        self.reports = None
        # self.lineEdit_UserName.setText("")
        self.gender = '男'
        self.reportPage = None
        self.label_showCamera.clear()
        self.reportUtils = None
        self.radioButton_male.click()
        self.radioButton_NormalImage.click()
        self.textEdit_Report.clear()
        self.buttonEnable(True)
        self.showInfo("工作区重置成功！")
        self.loadProcessImg()
        self.image = None
        self.skinImage = None

    # def SaveReport(self):
    #     if self.reports is None:
    #         self.showError("尚未生成报告,请先进行诊断")
    #         return
    #
    #     self.showInfo("正在保存报告，请稍等...")
    #     self.button_seeReport.setEnabled(False)
    #     BackendThread.InnerThread(self, BackendThread.generateReport, self.handleSaveReport,
    #                               {'reports': self.reports}
    #                               ).start()
    #
    # def handleSaveReport(self, paramMap):
    #     self.button_seeReport.setEnabled(True)
    #     self.showInfo("保存成功！")
    #


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainGUI()
    myWin.setWindowTitle("人脸像素统计与分析软件")
    myWin.show()
    sys.exit(app.exec_())
