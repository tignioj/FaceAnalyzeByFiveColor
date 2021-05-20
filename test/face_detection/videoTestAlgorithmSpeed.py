import re

import cv2
# from face_detect_mtcnn import faceDetect as faceDetectCNN
from face_detect_dlib_hog import faceDetectHog
import time

# from face_detect_mtcnn import faceDetectMTCNN
from face_detect_haar import faceDetectHaar
# from face_detect_mtcnn import faceDetectMTCNN

scale = 6


def detectFace(img):
    return faceDetectHog(img, scale)
    # return faceDetectMTCNN(img)
    # return faceDetectHaar(img)


def _testVideo():
    video_capture = cv2.VideoCapture(0)
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX

    averageFPS = 0
    k = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        # if video finished or no Video Input
        if not ret:
            break
        # small_frame = cv2.resize(frame, (0, 0), fx=1 / scale, fy=1 / scale)
        new_frame_time = time.time()
        d = (new_frame_time - prev_frame_time)
        if d == 0: d = 1
        fps = 1 / d
        k += 1
        averageFPS += fps

        prev_frame_time = new_frame_time
        fps = str(fps)

        # facesPoints = detectFace(small_frame)
        detected_img = detectFace(frame)
        s = str(detected_img.shape[1]) + "x" + str(detected_img.shape[0]) + ",FPS:" + re.sub(r'(\d+\.\d{2})(.*)', r'\1',
                                                                                             fps)
        cv2.putText(detected_img, s, (7, 50), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("face", detected_img)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            print(averageFPS/k)
            break


if __name__ == '__main__':
    _testVideo()
