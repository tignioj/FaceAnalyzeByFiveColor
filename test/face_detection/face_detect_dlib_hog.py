import time

import cv2
import dlib, os

# 方法 显示图片
from core.const_var import COLORDICT
# 方法：绘制人脸矩形框
def plot_rectangle(image, faces, scale):
    for face in faces:
        cv2.rectangle(image,
                      (face.left() * scale, face.top() * scale),
                      (face.right() * scale, face.bottom() * scale),
                      (255, 0, 0),
                      4)

    return image
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../model/face_landmark/shape_predictor_68_face_landmarks.dat")

def faceDetectHog(img, scale=1):
    small_img = cv2.resize(img, (0, 0), fx=1 / scale, fy=1 / scale)
    pt = time.time()
    # 调用dlib库中的检测器
    faces = detector(small_img, 1)  # 1代表将图片放大1倍数

    plot_rectangle(img, faces, scale)
    et = time.time()
    print("人脸检测时长:", et-pt, "图片大小:", img.shape)

    return img


if __name__ == '__main__':
    res = faceDetectHog(cv2.imread("../../faces/7.jpeg"))
    cv2.imshow("hog", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
