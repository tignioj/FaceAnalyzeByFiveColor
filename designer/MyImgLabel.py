# https://www.programmersought.com/article/36971069089/
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:49:32 2019
@author: Tiny
"""
# =============================================================================
from PyQt5.QtWidgets import QDialog, QGridLayout, QLabel
from numpy.ma import angle

''' Mouse event, each action response event can be customized '''

''' Reference: 1. https://blog.csdn.net/richenyunqi/article/details/80554257
                          Pyqt determines the mouse click event - left button press, middle button press, right button press, left and right button press, etc.;
         2. https://fennbk.com/8065
                          Pyqt5 mouse (introduction to events and methods)
         3. https://blog.csdn.net/leemboy/article/details/80462632
                          PyQt5 Programming - Mouse Events
         4. https://doc.qt.io/qtforpython/PySide2/QtGui/QWheelEvent.html#PySide2.QtGui.PySide2.QtGui.QWheelEvent.delta
            QWheelEvent'''
# =============================================================================
# =============================================================================
''' PyQt4 and PyQt5 difference: '''
#   PySide2.QtGui.QWheelEvent.delta()
#   Return type:	int
#   This function has been deprecated, use pixelDelta() or angleDelta() instead.
# =============================================================================
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

'''Custom QLabel class'''


class PopUpDLG(QDialog):
    def __init__(self, pixMap=None):
        super(PopUpDLG, self).__init__()
        self.setObjectName("self")
        self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.resize(pixMap.width(), pixMap.height())
        self.setMinimumSize(QSize(50, 50))
        self.setMaximumSize(QSize(1920, 1080))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../images/icons8_photo_gallery_100px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.labelImg = QLabel("图像")
        self.labelImg.setPixmap(pixMap)
        self.srcPixMap = pixMap
        self.gridLayout.addWidget(self.labelImg, 1, 1, 1, 1)
        self.retranslateUi(self)
        self.retrunVal = None
        self.xd = 0
        self.yd = 0
        self.speed = 55

    def keyPressEvent(self, ev):
        print(ev)
        if ev.key() == Qt.Key_Escape:
            self.close()
        else:
            QDialog.keyPressEvent(self, ev)

    def wheelEvent(self, event):
        # if event.delta() > 0: # Roller up, PyQt4
        # This function has been deprecated, use pixelDelta() or angleDelta() instead.
        angle = event.angleDelta() / 8  # Returns the QPoint object, the value of the wheel, in 1/8 degrees
        angleX = angle.x()  # The distance rolled horizontally (not used here)
        angleY = angle.y()  # The distance that is rolled vertically

        self.labelImg.setScaledContents(True)
        w = self.labelImg.width()
        h = self.labelImg.height()

        if angleY > 0:
            w += self.speed
            h += self.speed
            # self.labelImg.setMinimumSize(self.w, self.h)

        else:  # roll down
            w -= self.speed
            h -= self.speed
            # self.labelImg.setMaximumSize(self.w, self.h)

        x1 = event.x() + self.xd
        y1 = event.y() + self.yd
        self.labelImg.setPixmap(self.srcPixMap)
        # self.labelImg.move(int(x1 + self.xd - w1/2), int(y1 + self.yd - h1/2))

        self.labelImg.setFixedWidth(w)
        self.labelImg.setFixedHeight(h)
        self.labelImg.move(1000, 1000)

    def mouseDoubleClickEvent(self, a0: QtGui.QMouseEvent):
        if not self.isFullScreen():
            self.showFullScreen()
            # self.labelImg.setScaledContents(True)
            self.labelImg.setPixmap(self.srcPixMap.scaled(2000, 2000, Qt.KeepAspectRatio))
        # else:

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            srcX = self.labelImg.x()
            srcY = self.labelImg.y()
            x = event.x()
            y = event.y()
            self.xd = srcX - x
            self.yd = srcY - y

        if event.buttons() == QtCore.Qt.RightButton:  # left button pressed
            self.close()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        print(x, y)
        # if x > 0 and y > 0:
        # x -= self.labelImg.width() / 2
        # y -= self.labelImg.height() / 2
        self.labelImg.move(x + self.xd, y + self.yd)

    def retranslateUi(self, Dialog):
        _translate = QCoreApplication.translate

    def setPixMap(self, pixMap):
        self.labelImg.setPixmap(pixMap)

    def exec_(self):
        super(PopUpDLG, self).exec_()
        return self.retrunVal


class MyQImgLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(MyQImgLabel, self).__init__(parent)
        f = QFont("ZYSong18030", 10)  # Set the font, font size
        self.setFont(f)  # After the event is not defined, the two sentences are deleted or commented out.

    '''Reload the mouse click event (click) '''

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:  # left button pressed
            if self.largePixMap is not None:
                dialog = PopUpDLG(self.largePixMap)
                dialog.exec_()
            # if value:
            #     self.valText.setText(value)


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.imgLabel = MyQImgLabel()  # declare imgLabel

        qimg = QImage()
        qimg.load("../faces/7.jpeg")
        pixMap = QPixmap.fromImage(qimg.scaled(800, 800))

        self.imgLabel.largePixMap = pixMap

        self.image = QImage()  # declare new img
        if self.image.load("../faces/7.jpeg"):  # if the image is loaded, then
            self.imgLabel.setPixmap(QPixmap.fromImage(self.image.scaled(500, 500)))  # Display image

        self.gridLayout = QtWidgets.QGridLayout(self)  # Layout settings
        self.gridLayout.addWidget(self.imgLabel, 0, 0, 1,
                                  1)  # comment out these two sentences, no image will be displayed


'''Main function'''
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
