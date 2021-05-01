import cv2

from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QDir, pyqtSignal, QThread, QSize
# from GUIDesign import *
from designer.GUIDesigner import *
from core.faceHealthDetect import *


# from faceTest import

class BackendThread(QThread):
    update_report_signal = pyqtSignal()

    def __init__(self, inputImage, videoFlag):
        super(BackendThread, self).__init__()
        self.inputImage = inputImage
        self.videoFlag = videoFlag
        self.image = None
        self.gloss = None
        self.color = None

    def run(self):
        # faceColor, faceGloss, image = faceDetect(self.inputImage, self.videoFlag)
        self.color, self.gloss, self.image = faceDetect(self.inputImage, self.videoFlag)
        self.update_report_signal.emit()


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

        self.CameraTimer = QtCore.QTimer()
        # 摄像头
        self.CAM_NUM = 0
        self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE  # 0表示无图像，1表示开启摄像头读取图像，2表示打开图像文件

        # 信息区
        self.UserName = ""
        self.Gender = -1
        self.report = "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
        self.faceColor = ""
        self.faceGloss = ""

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
        self.actionClearImage.triggered.connect(self.ClearImage)
        self.horizontalSlider_EdgeTract.valueChanged.connect(self.SliderChangeValue)

    def OpenCamera(self):  # 打开摄像头，启动倒计时
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 后一个参数用来消一个奇怪的warn
        if self.CameraTimer.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查摄像头与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.CameraTimer.start(30)
                self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_USING_CAMERA
        else:
            self.CameraTimer.stop()
            self.cap.release()
            self.label_ShowCamera.clear()

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
            ShowVideo = cv2.resize(self.image, MyWindow.__IMAGE_LABEL_SIZE)

        elif self.VideoMode == MyWindow.__VIDEO_MODE_FACE:
            if self.VideoFlag == 0:
                ShowVideo = faceDetect(self.image, self.VideoFlag)

        elif self.VideoMode == MyWindow.__VIDEO_MODE_EDGE:
            self.edge = cv2.Canny(self.image, self.EdgeTractThrehold1, self.EdgeTractThrehold2)
            ShowVideo = cv2.resize(self.edge, MyWindow.__IMAGE_LABEL_SIZE)

        ShowVideo = cv2.cvtColor(ShowVideo, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(ShowVideo.data, ShowVideo.shape[1], ShowVideo.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.label_ShowCamera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def CloseCamera(self):
        if self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_NONE:
            return
        else:
            self.CameraTimer.stop()
            self.cap.release()
            self.label_ShowCamera.clear()
            self.label_ShowCamera.setPixmap(QtGui.QPixmap("../images/process1.png"))
            self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE

    def handleFaceDetectResult(self):
        """
        后台线程分析完毕，执行这个方法
        :return:
        """
        self.showloadingGIF(False)
        self.faceColor = self.backgroundThread.color
        self.faceGloss = self.backgroundThread.gloss
        self.image = self.backgroundThread.image

        ShowCapture = cv2.resize(self.image, (660, 495))
        ShowCapture = cv2.cvtColor(ShowCapture, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(ShowCapture.data, ShowCapture.shape[1], ShowCapture.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.label_ShowCamera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.CameraTimer.stop()

        # 根据人脸检测情况和人员信息，生成诊断结果
        SkinResults, GlossResults = CreateDetectResults(self.faceColor, self.faceGloss)
        # pdfCreate(filename, name, sex, faceColor, faceGloss, SkinResults, GlossResults)
        image = OUTPUT_PATH + "\\DiagnoseResult.jpg"
        wordCreate(self.UserName, self.Gender, self.faceColor, self.faceGloss, SkinResults, GlossResults, image)
        # 生成PDF报告
        wordfile = OUTPUT_PATH + "\\FaceDiagnoseResults.docx"
        pdffile = OUTPUT_PATH + "\\FaceDiagnoseResults.pdf"
        word2pdf(wordfile, pdffile)

        if self.Gender == 0: gender = "女"
        if self.Gender == 1: gender = "男"
        self.report = "姓名: %s \n" % self.UserName + "性别: %s \n" % gender + \
                      "您的面部诊断结果: \n" + "    肤色诊断结果: %s \n" % self.faceColor + "    皮肤光泽诊断结果: %s \n" % self.faceGloss + \
                      "诊断结果分析: \n %s \n" % SkinResults + "    %s" % GlossResults
        self.textEdit_Report.setPlainText(self.report)

    def Analyze(self):  # 要思考未打开摄像头时按下“拍照”的问题
        self.VideoFlag = 1
        self.UserName = self.lineEdit_UserName.text()
        if self.radioButton_Male.isChecked():
            self.Gender = 1
        elif self.radioButton_Female.isChecked():
            self.Gender = 0

        if self.Gender == -1 and self.UserName == "":
            QtWidgets.QMessageBox.warning(self, 'warning', "请输入姓名和选择性别", buttons=QtWidgets.QMessageBox.Ok)
        elif self.Gender == -1 and self.UserName != "":
            QtWidgets.QMessageBox.warning(self, 'warning', "请选择性别", buttons=QtWidgets.QMessageBox.Ok)
        elif self.Gender != -1 and self.UserName == "":
            QtWidgets.QMessageBox.warning(self, 'warning', "请输入姓名", buttons=QtWidgets.QMessageBox.Ok)
        else:
            if self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_NONE:
                QtWidgets.QMessageBox.warning(self, 'warning', "无图像输入")
            else:
                if self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_USING_CAMERA:
                    self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE
                    # flag, self.image = self.cap.read()

                    self.faceColor, self.faceGloss, self.image = faceDetect(self.image, self.VideoFlag)
                    self.backgroundThread = BackendThread(self.image, self.VideoFlag)
                    self.backgroundThread.update_report_signal.connect(self.handleFaceDetectResult)
                    self.backgroundThread.start()
                elif self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_USING_FILE:
                    self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE
                    # faseDetect过程比较长，需要多线程执行, 加载过渡动画动画
                    self.showloadingGIF(True)
                    #  开启后台线程检测
                    self.backgroundThread = BackendThread(self.image, self.VideoFlag)
                    self.backgroundThread.start()
                    #  接受后台线程执行完毕的信号
                    self.backgroundThread.update_report_signal.connect(self.handleFaceDetectResult)

    def showloadingGIF(self, isShow):
        if isShow:
            self.movie = QMovie("../images/face_scanning.gif")
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

    def ClearImage(self):
        if self.Flag_Image == MyWindow.__IMAGE_LABEL_STATE_USING_CAMERA:
            self.CameraTimer.stop()
            self.cap.release()
            self.Flag_Image = MyWindow.__IMAGE_LABEL_STATE_NONE
        else:
            self.textEdit_Report.clear()

        self.label_ShowCamera.clear()
        self.textEdit_Report.clear()

    def SliderChangeValue(self):
        self.EdgeTractThrehold1 = self.horizontalSlider_EdgeTract.value()

    def SaveReport(self):
        self.textEdit_Report.setPlainText("已保存")
