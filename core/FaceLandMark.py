from collections import OrderedDict

import cv2
import dlib
import numpy as np
from PIL import ImageFont, ImageDraw, Image

colors = {'blue': (255, 0, 0),
          'green': (0, 255, 0),
          'red': (0, 0, 255),
          'yellow': (0, 255, 255),
          'magenta': (255, 0, 255),
          'cyan': (255, 255, 0),
          'white': (255, 255, 255),
          'black': (0, 0, 0),
          'gray': (125, 125, 125),
          'rand': np.random.randint(0, high=256, size=(3,)).tolist(), 'dark_gray': (50, 50, 50),
          'light_gray': (220, 220, 220)}


class FaceDetect:
    def __init__(self):
        print("new instance")
        self._FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("ting", "庭"),  # 额头上面
            ("que", "阙"),  # 眉毛中间
            ("quan_left", "颧左"),  # 眼角旁边
            ("quan_right", "颧右"),  # 眼角旁边
            ("ming_tang", "明堂"),  # 鼻头
            ("jia_left", "颊左"),  # 脸颊左
            ("jia_right", "颊右"),  # 脸颊右
            ("chun", "唇部"),  # 脸颊右
            ("ke", "颏")  # 下巴到下唇中间
        ])

        self.scale = 1

        # 3. 调用人脸检测器
        self._detector = dlib.get_frontal_face_detector()

        # 4. 加载预测关键点模型(68个点)
        self._predictor = dlib.shape_predictor("../model/face_landmark/shape_predictor_68_face_landmarks.dat")

    def _getRegionFromCenter(self, centerPoint, size):
        width = size[0]
        height = size[1]
        """
        从中心点出发, 根据长和宽获得一个矩形坐标
        :param centerPoint:
        :param width:
        :param height:
        :return:
        """
        pts = []
        width = int(width / 2)
        height = int(height / 2)
        pts.append((centerPoint[0] - width, centerPoint[1] + height))  # 左上角
        pts.append((centerPoint[0] + width, centerPoint[1] + height))  # 右上角
        pts.append((centerPoint[0] + width, centerPoint[1] - height))  # 右下角
        pts.append((centerPoint[0] - width, centerPoint[1] - height))  # 左下角
        return pts

    def _getFaceWH(self, face):
        return face.bottom() - face.top(), face.right() - face.left()

    # 获取ROI
    def _getROI(self, name, shape, image, face):
        pts = []
        center_point = ()
        clone = image
        faceW, faceH = self._getFaceWH(face)

        queSize = (faceW * 0.16, faceH * 0.16)
        tingSize = (faceW * 0.3, faceH * 0.16)
        quanSize = (faceW * 0.16, faceH * 0.16)
        jiaSize = (faceW * 0.16, faceH * 0.16)
        mingTangSize = (faceW * 0.12, faceH * 0.12)
        keSize = (faceW * 0.3, faceH * 0.15)

        # cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['rand'], 2)
        if name == "ting":
            # 对于庭，我们只获取阙上面50个像素的位置为中心，大小为W:100, H50
            # 获取阙的中心，并且上移动x像素
            que_coord = shape[21:23].mean(axis=0)
            center_point = (int(que_coord[0]), int(que_coord[1] - faceH * 0.15))
            pts = self._getRegionFromCenter(center_point, tingSize)
        elif name == "que":  # 阙
            # 对于阙，我们需要获取21和22点的坐标中点，再截取50*50的像素
            coord = shape[21:23].mean(axis=0)
            center_point = (int(coord[0]), int(coord[1]))
            pts = self._getRegionFromCenter(center_point, queSize)
            # 构造一个矩形
        elif name == "quan_left":
            # 对于左边的颧骨(注意是指实际左边)，需要获取45和14的均值
            center_point = np.array([shape[45], shape[14]]).mean(axis=0)
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = self._getRegionFromCenter(center_point, quanSize)
            pass
        elif name == "quan_right":
            # 对于右边边的颧骨(注意是指实际右边)，需要获取36和2的均值
            center_point = np.array([shape[36], shape[2]]).mean(axis=0)
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = self._getRegionFromCenter(center_point, quanSize)
            pass
        elif name == "ming_tang":
            # 明堂：30点即可
            center_point = shape[30]
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = self._getRegionFromCenter(center_point, mingTangSize)
        elif name == "jia_left":  # 脸颊左边
            # 左边脸颊：y取值为鼻头30的y，x取左边眼睛46的x
            x = shape[46][0]
            y = shape[30][1]
            center_point = (int(x), int(y))
            pts = self._getRegionFromCenter(center_point, jiaSize)
        elif name == "jia_right":  # 脸颊右边
            # 左边脸颊：y取值为鼻头30的y，x取右边眼睛41的x
            x = shape[41][0]
            y = shape[30][1]
            center_point = (int(x), int(y))
            pts = self._getRegionFromCenter(center_point, jiaSize)
        elif name == "chun":  # 唇部
            # 唇部直接获取范围 48:68, 这是固定的范围
            center_point = np.array(shape[48:68]).mean(axis=0)
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = shape[48:68]
        elif name == "ke":  # 颏
            # 对于颏(ke 四声), 获取57和8的中点
            center_point = np.array([shape[57], shape[8]]).mean(axis=0)
            center_point = (int(center_point[0]), int(center_point[1]))
            pts = self._getRegionFromCenter(center_point, keSize)
        else:
            print("unknown name:", name)

        # extract the ROI of the face region as a separate image
        # 提取点的矩形
        (x, y, w, h) = cv2.boundingRect(np.array([pts]))
        roi = image[y:y + h, x:x + w]
        return (roi, pts, center_point)

    def _shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def _putTextCN(self, clone, coord, name_CN, face):
        img = clone
        faceW, faceH = self._getFaceWH(face)
        ## Use simsum.ttc to write Chinese.
        fontpath = "simsun.ttc"  # <== 这里是宋体路径
        font_size = int(round(faceW * 0.1))
        font = ImageFont.truetype(fontpath, font_size)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        # 使得中文居中
        Xdistance = (font_size / 2) * len(name_CN)
        Ydistance = (font_size / 2)
        text_x = int(coord[0] - Xdistance)
        text_y = int(coord[1] - Ydistance)
        if text_x <= 5:
            text_x = 5
        if text_y <= 5:
            text_y = 5
        coord = (text_x, text_y)
        draw.text(coord, name_CN, font=font, fill=colors['green'])
        img = np.array(img_pil)
        return img

    def faceDetectByImg(self, img, scale):
        copy = img.copy()
        small_img = cv2.resize(copy, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
        self.scale = scale
        # 6. 人脸检测
        faces = self._detector(small_img, 1)  # 1代表将图片放大1倍数
        # 7. 循环，遍历每一张人脸， 绘制矩形框和关键点
        for (i, face) in enumerate(faces):
            shape = self._predictor(small_img, face)
            shape = self._shape_to_np(shape)

            for (nameKey, name_CN) in self._FACIAL_LANDMARKS_IDXS.items():
                (roi, pts, center_point) = self._getROI(nameKey, shape, copy, face)
                pts = self.scale * np.array(pts)
                center_point = self.scale * np.array(center_point)
                # 根据点画出折线
                path = [pts.reshape((-1, 1, 2))]
                cv2.polylines(copy, path, True, (0, 255, 0), 1)
                # 加上文字
                copy = self._putTextCN(copy, center_point, name_CN, face)
                # roi = imutils.resize(roi, width=200, inter=cv2.INTER_CUBIC)
                # cv2.imshow(nameKey, roi)

            # 画出人脸矩形
            cv2.rectangle(copy, (face.left() * scale, face.top() * scale),
                          (face.right() * scale, face.bottom() * scale), (255, 0, 0), 3, cv2.LINE_AA)
        return copy


def _testImage():
    img = cv2.imread("../faces/7.jpeg")
    f = FaceDetect()
    detected_img = f.faceDetectByImg(img)
    cv2.imshow("face", detected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _testVideo():
    video_capture = cv2.VideoCapture(1)
    f = FaceDetect()
    while True:
        ret, frame = video_capture.read()
        detected_img = f.faceDetectByImg(frame, 4)
        cv2.imshow("face", detected_img)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    # testImage()
    _testVideo()
