import imutils
import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap
import PyQt5.QtGui
from PyQt5 import QtGui


def getcvImgFromFigure(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def nparrayToQPixMap(nparrayImg, width=None, height=None):
    """
    OpenCV的BGR的数组转换成QPixMap
    :param nparrayImg:
    :return:
    """
    if width is None: width = nparrayImg.shape[1]
    if height is None: height = nparrayImg.shape[0]

    nparrayImg = cv2.cvtColor(nparrayImg, cv2.COLOR_BGR2RGB)
    # showImage = QtGui.QImage(nparrayImg.data, nparrayImg.shape[1], nparrayImg.shape[0], QtGui.QImage.Format_RGB888)
    # return QtGui.QPixmap.fromImage(showImage)

    im = nparrayImg
    im = QImage(im.data, im.shape[1], im.shape[0], im.shape[1] * 3, QImage.Format_RGB888)
    # pix = QPixmap(im).scaled(a.width(), a.height())
    return QPixmap(im).scaled(width, height)


def cvshow(img, title="0x123"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def changeFrameByLableSizeKeepRatio(frame, fixW, fixH):
    # 如果视频的宽：高 > 显示区域的宽：高，说明应该以视频的宽度作为基准，计算出新的高度
    # frameHeight, frameWidth = frame.shape[0], frame.shape[1]
    # rationFrame = frameWidth / frameHeight
    # rationLabel = fixW / fixH
    #
    # if rationFrame > rationLabel:
    #     frameHeight = fixW
    #     frameWidth = frameHeight / rationFrame
    # else:
    #     frameWidth = fixH
    #     frameHeight = frameWidth * rationFrame
    #
    # return cv2.resize(frame, (int(frameHeight), int(frameWidth)))
    return imutils.resize(frame, width=fixW, height=fixH)
