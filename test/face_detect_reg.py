import time

import face_recognition
import cv2
import numpy as np


# 3 方法：绘制LandMars关键点
def show_landmarks(image, landmarks):
    # 获取字典
    for landmarks_dict in landmarks:
        # 获取字典的键
        for landmarks_key in landmarks_dict.keys():
            # 获取字典的值
            for point in landmarks_dict[landmarks_key]:
                cv2.circle(image, (point[0] * 4, point[1] * 4), 2, (0, 0, 255), -1)
    return image


def faceDetect(frame, scale):
    ptime = time.time()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_landmark = face_recognition.face_landmarks(rgb_small_frame)

    # Display the results
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        # 8.绘制关键点
        frame = show_landmarks(frame, face_landmark)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "test", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    # cv2.imshow('Video', frame)

    ntime = time.time()
    difftime = ntime - ptime
    print('检测用时:', difftime)
    return frame

# Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
