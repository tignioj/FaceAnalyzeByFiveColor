from PyQt5.QtCore import *
import imutils
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from utils.ImageUtils import nparrayToQPixMap
import sys
from designer.ReportPage import Ui_MainWindow as ReportPage


class ReportPageImpl(QMainWindow, ReportPage):
    def __init__(self):
        super(ReportPageImpl, self).__init__()
        self.setupUi(self)
        self.pushButton_back_home.clicked.connect(self.closeSelf)

    def closeSelf(self):
        self.close()

    def loadReports(self, reports):
        for r in reports:
            sz = self.label_face_drawed.size()
            img = imutils.resize(r.drawImg, width=sz.width(), height=sz.height())
            self.label_face_drawed.setPixmap(nparrayToQPixMap(img))
            self.label_username.setText(str(r.username))

