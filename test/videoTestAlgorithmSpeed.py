import cv2
# from face_detect_mtcnn import faceDetect as faceDetectCNN
from face_detect_dlib_hog import faceDetect as faceDetectHog
import time

scale = 2


def detectFace(img):
    # return faceDetectCNN(img)
    return faceDetectHog(img)


def drawFaces(image, faces):
    for face in faces:
        cv2.rectangle(image,
                      (face.left() * scale, face.top() * scale),
                      (face.right() * scale, face.bottom() * scale),
                      (255, 0, 0),
                      4)
    return image


def _testVideo():
    video_capture = cv2.VideoCapture(1)
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        # if video finished or no Video Input
        if not ret:
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / scale, fy=1 / scale)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]

        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        # fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        facesPoints = detectFace(small_frame)
        detected_img = drawFaces(frame, facesPoints)

        # puting the FPS count on the frame
        cv2.putText(detected_img, "FPS:" + fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("face", detected_img)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    _testVideo()
