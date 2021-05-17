import time

import cv2

videoCapture = cv2.VideoCapture(1)
def main():
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while videoCapture.isOpened():
        # Grab a single frame of video
        ret, frame = videoCapture.read()

        # if video finished or no Video Input
        if not ret:
            break

        scale = 3
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1 / scale, fy=1 / scale)
        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(fps)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)


        # puting the FPS count on the frame
        cv2.putText(frame, "FPS:" + fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            videoCapture.release()
            cv2.destroyAllWindows()
            break
