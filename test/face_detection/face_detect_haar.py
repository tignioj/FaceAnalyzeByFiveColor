import cv2
from core.const_var import *

from core.const_var import OUTPUT_PATH, OPENCV_CASCADE_PATH



def faceDetectHaar(img):
    cascPath = OPENCV_CASCADE_PATH + "\\haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)

    faces = faceCascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    '''
        基于图片的人脸检测
        img: 表示要检测的图片
        minNeighbors 表示每个人脸最少检查5次才确定是人脸
        scaleFactor 每次检测人脸图像缩小比例
        maxsize 人脸最大像素
        minsize 人脸最小像素
    '''
    for (x, y, w, h) in faces:
        # 在图像上绘制出人脸矩形
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)
        # 脸部区域
        face_part = img[y:y + h, x:x + w]

    return img


if __name__ == '__main__':
    img = cv2.imread("../../faces/white.jpg")
    imgHaar = faceDetectHaar(img)
    cv2.imshow("haar", imgHaar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
