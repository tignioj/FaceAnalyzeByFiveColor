"""运行失败"""

import cv2
import dlib, os
import numpy as np
import matplotlib.pyplot as plt

# 方法 显示图片
from core.const_var import COLORDICT


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

predictor = dlib.shape_predictor("../model/face_landmark/shape_predictor_68_face_landmarks.dat")


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def faceDetect(img, scale=1):
    # 灰度转换
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # small_img = img
    small_img = cv2.resize(img, (0, 0), fx=1 / scale, fy=1 / scale)

    # 调用dlib库中的检测器
    faces = detector(small_img, 1)  # 1代表将图片放大1倍数

    for (i, face) in enumerate(faces):
        shape = predictor(small_img, face)
        # shape = shape_to_np(shape)
        for pt in shape.parts():
            pt_position = (pt.x * scale, pt.y * scale)
            cv2.circle(img, pt_position, 1, COLORDICT['red'],
                       2)

    # 给检测出的人脸绘制矩形框
    plot_rectangle(img, faces, scale)
    return img


if __name__ == '__main__':
    res = faceDetect(cv2.imread("../faces/7.jpeg"))
    cv2.imshow("hog", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
