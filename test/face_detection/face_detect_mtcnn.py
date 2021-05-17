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
from mtcnn.mtcnn import MTCNN


# 默认情况下Tensorflow会一直占用所有的GPU内存，通过以下代码限制内存
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpudevices = tf.config.experimental.list_physical_devices('GPU')
if gpudevices:
    try:
        for oneGPU in gpudevices:
            tf.config.experimental.set_memory_growth(oneGPU, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpudevices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# draw an image with detected objects
def drawImageBox(image, result_list):
    for result in result_list:
        x, y, width, height = result['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return image

mtcnn_detector = MTCNN()
def faceDetectMTCNN(img):
    faces = mtcnn_detector.detect_faces(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = drawImageBox(img, faces)
    return result


if __name__ == '__main__':
    res = faceDetectMTCNN(cv2.imread("../../faces/white.jpg"))
    cv2.imshow("mtcnn", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
