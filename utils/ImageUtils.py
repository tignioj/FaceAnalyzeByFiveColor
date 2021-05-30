import imutils
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from PyQt5.QtGui import QImage, QPixmap
from core.const_var import *
from utils.LogUtils import LogUtils


class ImgUtils:
    @staticmethod
    def getImg(path, width=50, height=50):
        img = cv2.imread(path)
        return img
        # return imutils.resize(img, width, height)

    @staticmethod
    def keepSameShape(img1, img2, width=None, height=None):
        if width is None:
            h = min([img1.shape[0], img2.shape[0]])
        else:
            h = height
        if height is None:
            w = min([img1.shape[1], img2.shape[1]])
        else:
            w = width
        return cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))

    KEY_SAMPLE_YELLOW = "yellow"
    KEY_SAMPLE_RED = "chi"
    KEY_SAMPLE_BLACK = "black"
    KEY_SAMPLE_WHITE = "white"
    SAMPLE_PATH = BASE_PATH + "/four_color_face_sample"

    COLOR_SAMPLE_CN_NAME_BY_KEY = {
        KEY_SAMPLE_WHITE: '白',
        KEY_SAMPLE_RED: '赤',
        KEY_SAMPLE_YELLOW: '黄',
        KEY_SAMPLE_BLACK: '黑'
    }

    @staticmethod
    # https://stackoverflow.com/questions/50854235/how-to-draw-chinese-text-on-the-image-using-cv2-puttextcorrectly-pythonopen
    def putTextCN(image, text, coordinate=(1, 1), color=COLORDICT['white'], fontSize=15):
        fontpath = FONT_PATH
        font = ImageFont.truetype(fontpath, fontSize)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text(coordinate, text, font=font, fill=color)
        img = np.array(img_pil)
        return img
        # cv2.putText(img, text, coordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    @staticmethod
    def _getSampleDict():
        """
        返回所有ROI样本图片，图片格式是numpy, 数据格式是dict, 请用
        :return: blackSample, yellowSample, redSample, whiteSample
        """
        black_sample_dict = {}
        black_sample_dict_trim = {}

        yellow_sample_dict= {}
        yellow_sample_dict_trim = {}

        red_sample_dict = {}
        red_sample_dict_trim = {}

        white_sample_dict = {}
        white_sample_dict_trim = {}
        for roiName in FACIAL_LANDMARKS_NAME_DICT.keys():
            LogUtils.log("ImageUtils",
                         "正在加载" + FACIAL_LANDMARKS_NAME_DICT[
                             roiName] + "的样本图像数据" + ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_BLACK + "/" + roiName + ".jpg")

            black_sample_dict[roiName] = ImgUtils.getImg( ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_BLACK + "/" + roiName + ".jpg")
            black_sample_dict_trim[roiName] = ImgUtils.getImg( ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_BLACK + "/" + roiName + "_trim.jpg")

            yellow_sample_dict[roiName] = ImgUtils.getImg( ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_YELLOW + "/" + roiName + ".jpg")
            yellow_sample_dict_trim[roiName] = ImgUtils.getImg( ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_YELLOW + "/" + roiName + "_trim.jpg")

            red_sample_dict[roiName] = ImgUtils.getImg( ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_RED + "/" + roiName + ".jpg")
            red_sample_dict_trim[roiName] = ImgUtils.getImg( ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_RED + "/" + roiName + "_trim.jpg")

            white_sample_dict[roiName] = ImgUtils.getImg( ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_WHITE + "/" + roiName + ".jpg")
            white_sample_dict_trim[roiName] = ImgUtils.getImg( ImgUtils.SAMPLE_PATH + "/" + ImgUtils.KEY_SAMPLE_WHITE + "/" + roiName + "_trim.jpg")

        sampleDict = {
            ImgUtils.KEY_SAMPLE_RED: red_sample_dict,
            ImgUtils.KEY_SAMPLE_RED + "_trim": red_sample_dict_trim,

            ImgUtils.KEY_SAMPLE_WHITE: white_sample_dict,
            ImgUtils.KEY_SAMPLE_WHITE + "_trim": white_sample_dict_trim,

            ImgUtils.KEY_SAMPLE_BLACK: black_sample_dict,
            ImgUtils.KEY_SAMPLE_BLACK + "_trim": black_sample_dict_trim,

            ImgUtils.KEY_SAMPLE_YELLOW: yellow_sample_dict,
            ImgUtils.KEY_SAMPLE_YELLOW + "_trim": yellow_sample_dict_trim
        }
        return sampleDict

    # _sammpleDict = _getSampleDict()

    _sampleDict = None

    @staticmethod
    def getSampleDict():
        """
        获取样本，方法：
        :return:
        """
        if ImgUtils._sampleDict is None:
            ImgUtils._sampleDict = ImgUtils._getSampleDict()

        return ImgUtils._sampleDict

    @staticmethod
    def getcvImgFromFigure(fig):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    # def nparrayToQPixMap(nparrayImg, width=None, height=None):
    #     """
    #     OpenCV的BGR的数组转换成QPixMap
    #     :param nparrayImg:
    #     :return:
    #     """
    #     if width is None: width = nparrayImg.shape[1]
    #     if height is None: height = nparrayImg.shape[0]
    #
    #     nparrayImg = cv2.cvtColor(nparrayImg, cv2.COLOR_BGR2RGB)
    #     # showImage = QtGui.QImage(nparrayImg.data, nparrayImg.shape[1], nparrayImg.shape[0], QtGui.QImage.Format_RGB888)
    #     # return QtGui.QPixmap.fromImage(showImage)
    #
    #     im = nparrayImg
    #     im = QImage(im.data, im.shape[1], im.shape[0], im.shape[1] * 3, QImage.Format_RGB888)
    #     # pix = QPixmap(im).scaled(a.width(), a.height())
    #     return QPixmap(im).scaled(width, height)
    @staticmethod
    def nparrayToQPixMap(nparrayImg, labelWidth=None, labelHeight=None):
        """
        OpenCV的BGR的数组转换成QPixMap
        :param nparrayImg:
        :return:
        """
        """
        如果没有设置label的宽高，则认为label的宽高就是图片的宽高"""
        if labelWidth is None: labelWidth = nparrayImg.shape[1]

        if labelHeight is None: labelHeight = nparrayImg.shape[0]

        imgWidth = nparrayImg.shape[1]
        imgHeight = nparrayImg.shape[0]
        imgRatio = imgWidth / imgHeight
        labelRatio = labelWidth / labelHeight

        " 如果图片的宽度大于Label的宽度，则设置图片的宽度为label的宽度，剩下的高度按照原比例计算 "
        if imgWidth >= labelWidth:
            if imgRatio > labelRatio:
                newWidth = labelWidth
                newHeight = newWidth / imgRatio
            else:
                newHeight = labelHeight
                newWidth = newHeight * imgRatio
        else:
            if imgRatio > labelRatio:
                newWidth = labelWidth
                newHeight = newWidth / imgRatio
            else:
                newHeight = labelHeight
                newWidth = newHeight * imgRatio

        # newWidth = labelHeight * imgRatio

        nparrayImg = cv2.cvtColor(nparrayImg, cv2.COLOR_BGR2RGB)
        # showImage = QtGui.QImage(nparrayImg.data, nparrayImg.shape[1], nparrayImg.shape[0], QtGui.QImage.Format_RGB888)
        # return QtGui.QPixmap.fromImage(showImage)

        im = nparrayImg
        im = QImage(im.data, im.shape[1], im.shape[0], im.shape[1] * 3, QImage.Format_RGB888)
        # im = QImage(im.data, im.shape[1], im.shape[0], QImage.Format_RGB888)
        return QPixmap(im).scaled(newWidth, newHeight)

    @staticmethod
    def cvshow(img, title="0x123"):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    @staticmethod
    def changeFrameByLableSizeKeepRatio(frame, fixW, fixH):
        return imutils.resize(frame, width=fixW, height=fixH)

if __name__ == '__main__':
    d = ImgUtils.getSampleDict()
    print(d)
