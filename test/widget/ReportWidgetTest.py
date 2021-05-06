from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from service.ReportService import ReportService
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QDir, pyqtSignal, QThread, QSize, QDateTime
from designer.ReportPageImpl import ReportPageImpl as ReportPage
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ReportPage()
    w.setupUi(w)
    w.setWindowTitle("人脸像素统计与分析软件")
    w.show()
    sys.exit(app.exec_())
