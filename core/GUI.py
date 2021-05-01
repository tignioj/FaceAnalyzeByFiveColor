import re

import cv2
import imutils
import sys
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from ReportUtils import ReportUtils
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QDir, pyqtSignal, QThread, QSize, QDateTime
# from GUIDesign import *
from designer.GUIDesigner import *
from core.faceHealthDetect import *
from FaceLandMark import FaceDetect


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
                self.signal.emit({'res': res, 'param': self.callBackParam})
            except Exception as err:
                # 有可能出现异常
                self.signal.emit({'res': err, 'param': self.callBackParam})

    # 需要执行大量计算的函数
    @staticmethod
    def faceDetectInBackground(paramMap):
        # faceColor, faceGloss, image = faceDetect(self.inputImage, self.videoFlag)
        color, gloss, image = faceDetect(paramMap['image'], paramMap['flag'])
        return {'color': color, 'gloss': gloss, 'image': image}

    # 生成报告，有IO操作
    @staticmethod
    def generateReport(paramMap):
        report = paramMap['reportUtils']
        report.wordCreate()
        report.word2pdf()
        return {'report', report}


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

        self.faceDetector = FaceDetect()
        self.CameraTimer = QtCore.QTimer()
        # 摄像头
        self.CAM_NUM = 1
        self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE  # 0表示无图像，1表示开启摄像头读取图像，2表示打开图像文件
        # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 后一个参数用来消一个奇怪的warn
        self.cap = None  # 后一个参数用来消一个奇怪的warn

        # 信息区
        self.UserName = ""
        self.Gender = -1
        self.reportText = "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
        self.faceColor = ""
        self.faceGloss = ""
        self.reportUtils = None

        # 图像区
        self.VideoMode = MyWindow.__VIDEO_MODE_NORMAL  # 定义图像输出模式
        self.EdgeTractThrehold1 = 50
        self.EdgeTractThrehold2 = self.EdgeTractThrehold1 + 200
        self.VideoFlag = MyWindow.__VIDEO_MODE_NORMAL

        self.setupUi(self)
        self.slot_init()

    def slot_init(self):
        self.button_CaptureAnalyse.clicked.connect(self.Analyze)
        self.button_SaveReport.clicked.connect(self.SaveReport)
        self.CameraTimer.timeout.connect(self.ShowCamera)  # 每次倒计时溢出，调用函数刷新页面
        self.actionOpenImage.triggered.connect(self.OpenImage)
        self.actionOpenCamera.triggered.connect(self.OpenCamera)
        self.actionCloseCamera.triggered.connect(self.CloseCamera)
        self.actionReset.triggered.connect(self.WorkSpaceReset)
        self.horizontalSlider_EdgeTract.valueChanged.connect(self.SliderChangeValue)

    def OpenCamera(self):  # 打开摄像头，启动倒计时
        self.showInfo("开启摄像头")
        if self.CameraTimer.isActive() == False:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            flag = self.cap.open(self.CAM_NUM, cv2.CAP_DSHOW)
            if flag == False:
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查摄像头与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.CameraTimer.start(30)
                self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_USING_CAMERA
        else:
            self.showError("摄像头已经开启了")


    def ShowCamera(self):
        self.VideoFlag = MyWindow.__VIDEO_MODE_NORMAL
        if self.radioButton_NormalImage.isChecked():
            self.VideoMode = MyWindow.__VIDEO_MODE_NORMAL
        elif self.radioButton_EdgeTract.isChecked():
            self.VideoMode = MyWindow.__VIDEO_MODE_EDGE
        elif self.radioButton_FaceTract.isChecked():
            self.VideoMode = MyWindow.__VIDEO_MODE_FACE

        flag, self.image = self.cap.read()

        if self.VideoMode == MyWindow.__VIDEO_MODE_NORMAL:
            currentFrame = cv2.resize(self.image, MyWindow.__IMAGE_LABEL_SIZE)

        elif self.VideoMode == MyWindow.__VIDEO_MODE_FACE:
            if self.VideoFlag == MyWindow.__VIDEO_MODE_NORMAL:
                self.image = cv.resize(self.image, self.__IMAGE_LABEL_SIZE)
                currentFrame = self.faceDetector.faceDetectByImg(self.image, 2)

        elif self.VideoMode == MyWindow.__VIDEO_MODE_EDGE:
            self.edge = cv2.Canny(self.image, self.EdgeTractThrehold1, self.EdgeTractThrehold2)
            currentFrame = cv2.resize(self.edge, MyWindow.__IMAGE_LABEL_SIZE)

        showImage = self.nparrayToQPixMap(currentFrame)
        self.label_ShowCamera.setPixmap(showImage)

    def nparrayToQPixMap(self, ShowVideo):
        """
        OpenCV的BGR的数组转换成QPixMap
        :param ShowVideo:
        :return:
        """
        ShowVideo = cv2.cvtColor(ShowVideo, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(ShowVideo.data, ShowVideo.shape[1], ShowVideo.shape[0], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(showImage)

    def releaseCamera(self):
        if self.CameraTimer is not None:
            if self.CameraTimer.isActive():
                self.CameraTimer.stop()

        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
                cv2.destroyAllWindows()

        self.label_ShowCamera.clear()

    def CloseCamera(self):
        if self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_NONE:
            self.showError("你没有打开摄像头!")
            return
        else:
            self.releaseCamera()
            self.label_ShowCamera.setPixmap(QtGui.QPixmap("../images/process1.png"))
            self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE
        self.showInfo("已关闭摄像头!")

    def handleFaceDetectResult(self, paramMap):
        self.CameraTimer.stop()
        """
        后台线程分析完毕，执行这个方法
        :return:
        """
        res = paramMap['res']
        if type(res) is FaceNotFoundException:
            self.showloadingGIF(False)
            self.label_ShowCamera.setPixmap(self.nparrayToQPixMap(res.expression))
            self.showError("未能识别到面孔！请重置工作区再试试看。" + str(res.message))
            return

        self.faceColor = res['color']
        self.faceGloss = res['gloss']
        self.image = res['image']

        img_resize = cv2.resize(self.image, self.__IMAGE_LABEL_SIZE)
        ShowCapture = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(ShowCapture.data, ShowCapture.shape[1], ShowCapture.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.label_ShowCamera.setPixmap(QtGui.QPixmap.fromImage(showImage))

        # 根据人脸检测情况和人员信息，生成诊断结果
        skinResults, glossResults = ReportUtils.CreateDetectResults(self.faceColor, self.faceGloss)

        if self.Gender == 0: gender = "女"
        if self.Gender == 1: gender = "男"

        self.reportText = "姓名: %s \n" % self.UserName + "性别: %s \n" % gender + \
                          "您的面部诊断结果: \n" + "    肤色诊断结果: %s \n" % self.faceColor + "    皮肤光泽诊断结果: %s \n" % self.faceGloss + \
                          "诊断结果分析: \n %s \n" % skinResults + "    %s" % glossResults
        self.showInfo(self.reportText)
        self.reportUtils = ReportUtils(self.UserName, gender, self.faceColor, skinResults, glossResults, img_resize)
        self.button_CaptureAnalyse.setEnabled(True)
        self.showloadingGIF(False)

    def messageBoxWarning(self, msg):
        QtWidgets.QMessageBox.warning(self, 'warning', msg, buttons=QtWidgets.QMessageBox.Ok)
        self.button_CaptureAnalyse.setEnabled(True)

    def Analyze(self):  # 要思考未打开摄像头时按下“拍照”的问题
        """
        面容分析
        :return:
        """
        self.button_CaptureAnalyse.setEnabled(False)
        self.VideoFlag = 1
        self.UserName = self.lineEdit_UserName.text()
        if self.radioButton_Male.isChecked():
            self.Gender = 1
        elif self.radioButton_Female.isChecked():
            self.Gender = 0

        if self.Gender == -1 and self.UserName == "":
            self.messageBoxWarning("请输入姓名和选择性别")
        elif self.Gender == -1 and self.UserName != "":
            self.messageBoxWarning("请选择性别")
        elif self.Gender != -1 and self.UserName == "":
            self.messageBoxWarning("请输入姓名")
        else:
            if self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_NONE:
                self.messageBoxWarning("无图像输入")
            else:
                if self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_USING_CAMERA:
                    flag, self.image = self.cap.read()
                    self.CloseCamera()
                    try:
                        self.showInfo("正在为您诊断，请耐心等待...")
                        self.showloadingGIF(True)
                        BackendThread.InnerThread(BackendThread.faceDetectInBackground, self.handleFaceDetectResult,
                                                  {'image': self.image, 'flag': self.VideoFlag}
                                                  ).start()
                    except FaceNotFoundException as err:
                        self.button_CaptureAnalyse.setEnabled(True)
                        self.showError(err.message)

                elif self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_USING_FILE:
                    self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE
                    # faseDetect过程比较长，需要多线程执行, 加载过渡动画动画
                    self.showInfo("正在为您诊断，请耐心等待...")
                    BackendThread.InnerThread(BackendThread.faceDetectInBackground, self.handleFaceDetectResult,
                                              {'image': self.image, 'flag': self.VideoFlag}
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
        self.CameraTimer.stop()
        self.movie = QMovie("../images/face_scanning.gif")
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
        print(imagePath)
        img = QtGui.QPixmap(imagePath).scaled(self.label_ShowCamera.width(), self.label_ShowCamera.height())
        self.image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        self.label_ShowCamera.setPixmap(img)
        self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_USING_FILE

    def WorkSpaceReset(self):
        self.releaseCamera()
        self.label_ShowCamera.clear()
        self.reportUtils = None
        self.textEdit_Report.clear()
        self.button_CaptureAnalyse.setEnabled(True)
        self.showInfo("工作区重置成功！")
        self.label_ShowCamera.setPixmap(QtGui.QPixmap("../images/process1.png"))

    def SliderChangeValue(self):
        self.EdgeTractThrehold1 = self.horizontalSlider_EdgeTract.value()

    def SaveReport(self):
        # pdfCreate(filename, name, sex, faceColor, faceGloss, SkinResults, GlossResults)
        if self.reportUtils is None:
            self.showError("尚未生成报告,请先进行诊断")
            return

        self.showInfo("正在保存报告，请稍等...")
        self.button_SaveReport.setEnabled(False)
        BackendThread.InnerThread(BackendThread.generateReport, self.handleSaveReport,
                                  {'reportUtils': self.reportUtils}
                                  ).start()

    def handleSaveReport(self, paramMap):
        self.button_SaveReport.setEnabled(True)
        self.showInfo("保存成功！")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.setWindowTitle("人脸像素检测分析-在中医面诊的应用")
    myWin.show()
    sys.exit(app.exec_())
