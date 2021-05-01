"""运行失败"""

import cv2
import dlib, os
import numpy as np
import matplotlib.pyplot as plt


# 方法 显示图片
def show_image(image, title):
    img_RGB = image[:, :, ::-1]
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis("off")


# 方法：绘制人脸矩形框
def plot_rectangle(image, faces, scale):
    for face in faces:
        cv2.rectangle(image,
                      (face.left() * scale, face.top() * scale),
                      (face.right() * scale, face.bottom() * scale),
                      (255, 0, 0),
                      4)

    return image


# 导入cnn模型

detector = dlib.get_frontal_face_detector()


def faceDetect(img):
    # 灰度转换
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 调用dlib库中的检测器
    dets_result = detector(img_rgb, 1)  # 1代表将图片放大1倍数

    # 给检测出的人脸绘制矩形框
    # img_result = plot_rectangle(img.copy(), dets_result, scale)

    return dets_result


if __name__ == '__main__':
    res = faceDetect(cv2.imread("../images/7.jpeg"))
    cv2.imshow("hog", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
