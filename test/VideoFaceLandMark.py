from collections import OrderedDict

import cv2
import dlib
import numpy as np
import imutils
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import time

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

scale = 4
"视频质量，可以选择1-4，1最高，4最低"


# 单个方法抽离出坐标
# 方法 显示图片
def show_image(image, title):
    img_RGB = image[:, :, ::-1]
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis("off")


FACIAL_LANDMARKS_IDXS = OrderedDict([
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


def getRegionFromCenter(centerPoint, size):
    width = size[0]
    height = size[1]
    # centerPoint = 4 * np.array(centerPoint)
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


# 获取ROI
def getROI(name, shape, image, face):
    pts = []
    center_point = ()
    clone = image
    faceH = face.bottom() - face.top()
    faceW = face.right() - face.left()
    queSize = (faceW * 0.16, faceH * 0.16)
    tingSize = (faceW * 0.3, faceH * 0.16)
    quanSize = (faceW * 0.16, faceH * 0.16)
    jiaSize = (faceW * 0.16, faceH * 0.16)
    mingTangSize = (faceW * 0.12, faceH * 0.12)
    keSize = (faceW * 0.3, faceH * 0.15)

    cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, colors['rand'], 2)
    if name == "ting":
        # 对于庭，我们只获取阙上面50个像素的位置为中心，大小为W:100, H50
        # 获取阙的中心，并且上移动x像素
        que_coord = shape[21:23].mean(axis=0)
        center_point = (int(que_coord[0]), int(que_coord[1] - faceH * 0.15))
        pts = getRegionFromCenter(center_point, tingSize)
    elif name == "que":  # 阙
        # 对于阙，我们需要获取21和22点的坐标中点，再截取50*50的像素
        coord = shape[21:23].mean(axis=0)
        center_point = (int(coord[0]), int(coord[1]))
        pts = getRegionFromCenter(center_point, queSize)
        # 构造一个矩形
    elif name == "quan_left":
        # 对于左边的颧骨(注意是指实际左边)，需要获取45和14的均值
        center_point = np.array([shape[45], shape[14]]).mean(axis=0)
        center_point = (int(center_point[0]), int(center_point[1]))
        pts = getRegionFromCenter(center_point, quanSize)
        pass
    elif name == "quan_right":
        # 对于右边边的颧骨(注意是指实际右边)，需要获取36和2的均值
        center_point = np.array([shape[36], shape[2]]).mean(axis=0)
        center_point = (int(center_point[0]), int(center_point[1]))
        pts = getRegionFromCenter(center_point, quanSize)
        pass
    elif name == "ming_tang":
        # 明堂：30点即可
        center_point = shape[30]
        center_point = (int(center_point[0]), int(center_point[1]))
        pts = getRegionFromCenter(center_point, mingTangSize)
    elif name == "jia_left":  # 脸颊左边
        # 左边脸颊：y取值为鼻头30的y，x取左边眼睛46的x
        x = shape[46][0]
        y = shape[30][1]
        center_point = (int(x), int(y))
        pts = getRegionFromCenter(center_point, jiaSize)
    elif name == "jia_right":  # 脸颊右边
        # 左边脸颊：y取值为鼻头30的y，x取右边眼睛41的x
        x = shape[41][0]
        y = shape[30][1]
        center_point = (int(x), int(y))
        pts = getRegionFromCenter(center_point, jiaSize)
    elif name == "chun":  # 唇部
        # 唇部直接获取范围 48:68, 这是固定的范围
        center_point = np.array(shape[48:68]).mean(axis=0)
        center_point = (int(center_point[0]), int(center_point[1]))
        pts = shape[48:68]
    elif name == "ke":  # 颏
        # 对于颏(ke 四声), 获取57和8的中点
        center_point = np.array([shape[57], shape[8]]).mean(axis=0)
        center_point = (int(center_point[0]), int(center_point[1]))
        pts = getRegionFromCenter(center_point, keSize)
    else:
        print("unknown name:", name)

    # extract the ROI of the face region as a separate image
    # 提取点的矩形
    (x, y, w, h) = cv2.boundingRect(np.array([pts]))
    roi = image[y:y + h, x:x + w]
    return (roi, pts, center_point)


# 方法：绘制人脸矩形框
def plot_rectangle(image, faces):
    for face in faces:
        cv2.rectangle(image,
                      (face.left(), face.top()),
                      (face.right(), face.bottom()),
                      colors['black'],
                      4)
        return image


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def putTextCN(clone, coord, name_CN):
    img = clone
    ## Use simsum.ttc to write Chinese.
    fontpath = "simsun.ttc"
    font_size = 15
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


video_capture = cv2.VideoCapture(1)


def main():
    # 3. 调用人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 4. 加载预测关键点模型(68个点)
    predictor = dlib.shape_predictor("../model/face_landmark/shape_predictor_68_face_landmarks.dat")
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # if video finished or no Video Input
        if not ret:
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / scale, fy=1 / scale)
        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(fps)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # 6. 人脸检测
        faces = detector(rgb_small_frame, 1)  # 1代表将图片放大1倍数
        # 7. 循环，遍历每一张人脸， 绘制矩形框和关键点
        for (i, face) in enumerate(faces):
            shape = predictor(rgb_small_frame, face)
            shape = shape_to_np(shape)
            for (nameKey, name_CN) in FACIAL_LANDMARKS_IDXS.items():
                (roi, pts, center_point) = getROI(nameKey, shape, frame.copy(), face)
                pts = scale * np.array(pts)
                center_point = scale * np.array(center_point)
                # 根据点画出折线
                path = [pts.reshape((-1, 1, 2))]
                cv2.polylines(frame, path, True, (0, 255, 0), 1)
                # 加上文字
                frame = putTextCN(frame, center_point, name_CN)
                # roi = imutils.resize(roi, width=200, inter=cv2.INTER_CUBIC)
                # cv2.imshow(nameKey, roi)

        # puting the FPS count on the frame
        cv2.putText(frame, "FPS:" + fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
