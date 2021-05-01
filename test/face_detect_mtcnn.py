# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
import os
import tensorflow as tf

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# # (600-0.001) 0: 03:41.127197
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import numpy as np


# draw an image with detected objects
def draw_image_with_boxes(image, result_list):
    for result in result_list:
        x, y, width, height = result['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return image

# def pointTransform(result_list):
    # for result in result_list:



def faceDetect(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = draw_image_with_boxes(img, faces)
    return result


if __name__ == '__main__':
    res = faceDetect(cv2.imread("../images/7.jpeg"))
    cv2.imshow("mtcnn", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
