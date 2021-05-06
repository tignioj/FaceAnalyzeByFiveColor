import numpy as np
import cv2
import PyQt5.QtGui
from PyQt5 import QtGui


def getcvImgFromFigure(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def nparrayToQPixMap(ShowVideo):
    """
    OpenCV的BGR的数组转换成QPixMap
    :param ShowVideo:
    :return:
    """
    ShowVideo = cv2.cvtColor(ShowVideo, cv2.COLOR_BGR2RGB)
    showImage = QtGui.QImage(ShowVideo.data, ShowVideo.shape[1], ShowVideo.shape[0], QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(showImage)