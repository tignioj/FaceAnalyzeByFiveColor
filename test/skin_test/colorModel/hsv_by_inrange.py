# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

# Required modules

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

videoCapture = cv2.VideoCapture(1)



def hsv(image):
    """
    色相是颜色模型的颜色部分，并表示为0到360度之间的数字。在OpenCV中为0-180。定义主色[R，Y，G，C，B，M]
    饱和度是颜色中的灰色量，从0到100％。
    值与饱和度结合使用，并描述颜色的亮度或强度（0％至100％）。
    :param image:
    :return:
    """
    min_HSV = np.array([0, 58, 30], dtype="uint8")
    max_HSV = np.array([33, 255, 255], dtype="uint8")
    # Get pointer to video frames from primary device
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
    skinHSV = cv2.bitwise_and(image, image, mask=skinRegionHSV)
    return np.hstack([image, skinHSV])


def YCrCb(image):
    """
    Comparison between YCbCr Color Space and CIELab
    Color Space for Skin Color Segmentation
    编码的非线性RGB信号，通常用于视频编码和图像压缩工作。
    构造为RGB值和两个色差值Cr和Cb的加权和，这两个色差值是通过从RGB红色和蓝色分量中减去亮度来形成的
    Y为16至235，Cb和Cr为16至240
    皮肤像素在Cb-Cr平面中形成紧凑的簇。
    :param image:
    :return:
    """
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    # min_YCrCb = np.array([0, 133, 104], np.uint8)
    # max_YCrCb = np.array([235, 144, 139], np.uint8)

    imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    skinYCrCb = cv2.bitwise_and(image, image, mask=skinRegionYCrCb)

    return np.hstack([image, skinYCrCb])


def Lab(image):
    """
    编码的非线性RGB信号，通常用于视频编码和图像压缩工作。
    构造为RGB值和两个色差值Cr和Cb的加权和，这两个色差值是通过从RGB红色和蓝色分量中减去亮度来形成的
    Y为16至235，Cb和Cr为16至240
    皮肤像素在Cb-Cr平面中形成紧凑的簇。
    :param image:
    :return:
    """
    min_Lab = np.array([0, 128, 127], np.uint8)
    max_Lab = np.array([235, 143, 158], np.uint8)

    # min_YCrCb = np.array([0, 133, 104], np.uint8)
    # max_YCrCb = np.array([235, 144, 139], np.uint8)

    imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    skinRegionLab = cv2.inRange(imageLab, min_Lab, max_Lab)

    skinLab = cv2.bitwise_and(image, image, mask=skinRegionLab)

    return np.hstack([image, skinLab])


if __name__ == '__main__':
    while videoCapture.isOpened():
        # image = cv2.imread("../..")
        flag, frame = videoCapture.read()
        if not flag:
            break
        # img = hsv(frame)
        img_ycbcr = imutils.resize(YCrCb(frame), width=2000)
        cv2.imshow('ycbcr', img_ycbcr)

        img_hsv = imutils.resize(hsv(frame), width=2000)
        cv2.imshow('hsv', img_hsv)

        img_lab= imutils.resize(Lab(frame), width=2000)
        cv2.imshow('lab', img_lab)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()
