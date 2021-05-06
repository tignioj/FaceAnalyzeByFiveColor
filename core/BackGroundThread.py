from PyQt5.QtCore import QThread, pyqtSignal

from core.faceHealthDetect import faceDetect
from service.ReportService import ReportService
from utils.LogUtils import LogUtils


class BackendThread:
    class InnerThread(QThread):
        """
        匿名内部类线程
        """
        """定义一个信号"""
        signal = pyqtSignal(dict)

        def __init__(self,
                     parent, # 一定要传入这个，调试的时候不传没问题，正式运行的时候，没有parent一定会异常退出！
                     invokeFun,  # 让线程执行的函数，通常是大量计算的函数
                     callbackFun,  # 回调函数
                     invokeParam={},  # 线程执行函数需要传的参数，需要传入字典
                     callbackParam={}  # 回调函数的参数，需要传入字典
                     ):
            # 一定要调用父类，不然这个线程可能执行不了
            super(BackendThread.InnerThread, self).__init__(parent) # 一定要传入这个！！！
            LogUtils.log("BackendThread", "正在生成UI线程...")
            self.callBackFun = callbackFun
            self.invokeParam = invokeParam
            self.callBackParam = callbackParam
            self.invokeFun = invokeFun
            LogUtils.log("BackendThread", "后台线程初始化完毕！", (invokeFun, invokeParam , callbackParam,))

        def run(self):
            """ 线程调用start()之后就会调用这个方法 """

            # 绑定槽函数，也就是回调函数
            LogUtils.log("BackendThread-run", "正在执行run", self.signal)
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
                LogUtils.error("BackendThread-run", "出现异常！", err)
                self.signal.emit({'res': err, 'param': self.callBackParam})

    # 需要执行大量计算的函数
    @staticmethod
    def faceDetectInBackground(paramMap):
        LogUtils.log("BackendThread", "正在进行人脸检测...")
        detectedFaces = faceDetect(paramMap['image'], 1, paramMap['name'], paramMap['gender'])
        LogUtils.log("BackendThread", "检测到..." + str(len(detectedFaces)) + "张人脸, 准备生成报告...")
        reports = ReportService.generateReports(detectedFaces)
        LogUtils.log("BackendThread", '共生成' + str(len(reports)) + "份报告")
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
